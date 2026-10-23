"""Person + pose detection with YOLO26-pose, saved per sampled frame.

Run detection (results go to outputs/<video name>/detections/ by default):
    python -m endcap.detect run data/nvidia_retail/scene_071/camera_0639.mp4 --end 60
Render saved detections (optionally with zones) for a visual check:
    python -m endcap.detect render data/nvidia_retail/scene_071/camera_0639.mp4 \
        --zones configs/zones/camera_0639.json --out outputs/camera_0639/detections.mp4

Saved files (one folder per run):
    frames.parquet      one row per sampled frame: frame, t, timestamp (includes frames with no people)
    detections.parquet  one row per person per frame: frame, det, x1, y1, x2, y2, score,
                        then for each of the 17 COCO keypoints <name>_x, <name>_y, <name>_c
    meta.json           video, fps, size, model and settings used

Detection is kept separate from tracking so tracker settings can be tuned without re-running the model.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np
import pandas as pd

from endcap.config import load_config, resolve_device
from endcap.ingest import Frame, probe, read_frames

KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP = {name: i for i, name in enumerate(KEYPOINTS)}
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]
KP_COLUMNS = [f"{n}_{a}" for n in KEYPOINTS for a in ("x", "y", "c")]


@dataclass
class FrameDetections:
    frame: int
    t: float
    timestamp: datetime | None
    boxes: np.ndarray      # (N, 4) x1, y1, x2, y2 in pixels
    scores: np.ndarray     # (N,)
    keypoints: np.ndarray  # (N, 17, 3) x, y, confidence

    def __len__(self) -> int:
        return len(self.scores)


@dataclass
class DetectionRun:
    meta: dict
    frames: list[FrameDetections] = field(default_factory=list)

    # --- table conversion ---------------------------------------------------

    def frames_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"frame": [f.frame for f in self.frames],
             "t": [f.t for f in self.frames],
             "timestamp": pd.to_datetime([f.timestamp for f in self.frames])}
        )

    def detections_table(self) -> pd.DataFrame:
        rows = []
        for f in self.frames:
            if not len(f):
                continue
            n = len(f)
            block = np.column_stack([
                np.full(n, f.frame), np.arange(n), f.boxes, f.scores, f.keypoints.reshape(n, -1),
            ])
            rows.append(block)
        cols = ["frame", "det", "x1", "y1", "x2", "y2", "score", *KP_COLUMNS]
        df = pd.DataFrame(np.vstack(rows) if rows else np.empty((0, len(cols))), columns=cols)
        return df.astype({"frame": "int64", "det": "int64"})

    # --- save / load ----------------------------------------------------------

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.frames_table().to_parquet(out / "frames.parquet", index=False)
        self.detections_table().to_parquet(out / "detections.parquet", index=False)
        (out / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str), encoding="utf-8")
        return out

    @classmethod
    def load(cls, out_dir: str | Path) -> "DetectionRun":
        out = Path(out_dir)
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        frames_df = pd.read_parquet(out / "frames.parquet")
        dets = pd.read_parquet(out / "detections.parquet")
        by_frame = {k: g for k, g in dets.groupby("frame")}

        frames = []
        for row in frames_df.itertuples(index=False):
            ts = None if pd.isna(row.timestamp) else row.timestamp.to_pydatetime()
            g = by_frame.get(row.frame)
            if g is None:
                boxes, scores, kpts = np.empty((0, 4)), np.empty(0), np.empty((0, 17, 3))
            else:
                g = g.sort_values("det")
                boxes = g[["x1", "y1", "x2", "y2"]].to_numpy()
                scores = g["score"].to_numpy()
                kpts = g[KP_COLUMNS].to_numpy().reshape(-1, 17, 3)
            frames.append(FrameDetections(int(row.frame), float(row.t), ts, boxes, scores, kpts))
        return cls(meta, frames)


# --- model ----------------------------------------------------------------------

class PoseDetector:
    def __init__(self, model: str, models_dir: str | Path, conf: float = 0.25, iou: float = 0.7,
                 imgsz: int = 1280, half: bool = True, device: str = "auto"):
        from ultralytics import YOLO

        path = Path(model)
        if not path.is_absolute() and path.parent == Path("."):
            Path(models_dir).mkdir(parents=True, exist_ok=True)
            path = Path(models_dir) / path.name  # ultralytics downloads known weights to this path
        self.device = resolve_device(device)
        self.model = YOLO(str(path))
        self.kwargs = dict(conf=conf, iou=iou, imgsz=imgsz, classes=[0], verbose=False,
                           device=self.device, quantize=16 if half and self.device != "cpu" else None)

    @classmethod
    def from_config(cls, cfg: dict) -> "PoseDetector":
        d = cfg["detect"]
        return cls(d["model"], cfg["paths"]["models_dir"], d["conf"], d["iou"], d["imgsz"],
                   d["half"], cfg["device"])

    def predict(self, images: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return (boxes, scores, keypoints) per image."""
        out = []
        for r in self.model.predict(images, **self.kwargs):
            if r.boxes is None or len(r.boxes) == 0:
                out.append((np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty((0, 17, 3), np.float32)))
                continue
            out.append((
                r.boxes.xyxy.cpu().numpy().astype(np.float32),
                r.boxes.conf.cpu().numpy().astype(np.float32),
                r.keypoints.data.cpu().numpy().astype(np.float32),
            ))
        return out

    def detect_frames(self, frames: Iterable[Frame], batch: int = 8) -> Iterator[FrameDetections]:
        buf: list[Frame] = []
        for frame in frames:
            buf.append(frame)
            if len(buf) == batch:
                yield from self._flush(buf)
                buf = []
        if buf:
            yield from self._flush(buf)

    def _flush(self, buf: list[Frame]) -> Iterator[FrameDetections]:
        for f, (b, s, k) in zip(buf, self.predict([f.image for f in buf])):
            yield FrameDetections(f.index, f.t, f.timestamp, b, s, k)


def run_detection(video: str | Path, cfg: dict, start_s: float = 0.0, end_s: float | None = None,
                  log_every_s: float = 10.0) -> DetectionRun:
    info = probe(video)
    ing, det = cfg["ingest"], cfg["detect"]
    start_time = ing.get("start_time")
    if isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time)

    detector = PoseDetector.from_config(cfg)
    meta = {
        "video": str(video), "width": info.width, "height": info.height, "source_fps": info.fps,
        "sample_fps": ing.get("sample_fps"), "start_s": start_s, "end_s": end_s,
        "start_time": start_time, "clock_offset_s": ing.get("clock_offset_s", 0.0),
        "device": detector.device, **{f"detect_{k}": v for k, v in det.items()},
    }
    run = DetectionRun(meta)
    frames = read_frames(video, ing.get("sample_fps"), start_time, ing.get("clock_offset_s", 0.0), start_s, end_s)

    t0 = last_log = time.time()
    for fd in detector.detect_frames(frames, det.get("batch", 8)):
        run.frames.append(fd)
        now = time.time()
        if now - last_log >= log_every_s:
            print(f"  t={fd.t:7.1f}s  {len(run.frames)} frames  {len(run.frames) / (now - t0):.1f} fps")
            last_log = now
    elapsed = time.time() - t0
    n_det = sum(len(f) for f in run.frames)
    print(f"Detected {n_det} people in {len(run.frames)} frames in {elapsed:.1f} s "
          f"({len(run.frames) / max(elapsed, 1e-9):.1f} fps)")
    return run


# --- drawing --------------------------------------------------------------------

def draw_detections(image: np.ndarray, fd: FrameDetections, kpt_conf: float = 0.5) -> np.ndarray:
    out = image.copy()
    for box, score, kpts in zip(fd.boxes, fd.scores, fd.keypoints):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{score:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        visible = kpts[:, 2] >= kpt_conf
        for a, b in SKELETON:
            if visible[a] and visible[b]:
                cv2.line(out, tuple(map(int, kpts[a, :2])), tuple(map(int, kpts[b, :2])), (255, 200, 0), 2)
        for i, (x, y, c) in enumerate(kpts):
            if c >= kpt_conf:
                color = (0, 0, 255) if i <= KP["right_ear"] else (255, 255, 255)  # head points red
                cv2.circle(out, (int(x), int(y)), 3, color, -1)
    return out


def render(video: str | Path, run: DetectionRun, out: str | Path, zones=None, kpt_conf: float = 0.5) -> None:
    from endcap.zones import draw_zones

    m = run.meta
    by_frame = {f.frame: f for f in run.frames}
    fps = m.get("sample_fps") or m["source_fps"]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (m["width"], m["height"]))
    for frame in read_frames(video, m.get("sample_fps"), start_s=m.get("start_s") or 0.0, end_s=m.get("end_s")):
        fd = by_frame.get(frame.index)
        img = frame.image if zones is None else draw_zones(frame.image, zones)
        if fd is not None:
            img = draw_detections(img, fd, kpt_conf)
        cv2.putText(img, f"t={frame.t:.1f}s  people={len(fd) if fd else 0}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(img)
    writer.release()
    print(f"Wrote {out}")


# --- CLI ------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO26-pose person detection.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="detect people and save results")
    r.add_argument("video")
    r.add_argument("--config", default=None, help="experiment YAML merged over configs/default.yaml")
    r.add_argument("--start", type=float, default=0.0)
    r.add_argument("--end", type=float, default=None)
    r.add_argument("--fps", type=float, default=None, help="override ingest.sample_fps")
    r.add_argument("--model", default=None, help="override detect.model")
    r.add_argument("--out", default=None, help="default: outputs/<video name>/detections")

    v = sub.add_parser("render", help="draw saved detections on the video")
    v.add_argument("video")
    v.add_argument("--dets", default=None, help="default: outputs/<video name>/detections")
    v.add_argument("--zones", default=None, help="zones JSON to draw as well")
    v.add_argument("--out", default=None, help="default: <dets>/../detections.mp4")
    args = parser.parse_args()

    overrides: dict = {}
    if getattr(args, "fps", None):
        overrides.setdefault("ingest", {})["sample_fps"] = args.fps
    if getattr(args, "model", None):
        overrides.setdefault("detect", {})["model"] = args.model
    cfg = load_config(getattr(args, "config", None), overrides)
    default_dir = Path(cfg["paths"]["output_dir"]) / Path(args.video).stem / "detections"

    if args.cmd == "run":
        run = run_detection(args.video, cfg, args.start, args.end)
        print(f"Saved to {run.save(args.out or default_dir)}")
    else:
        from endcap.zones import CameraZones

        dets = Path(args.dets) if args.dets else default_dir
        run = DetectionRun.load(dets)
        zones = CameraZones.load(args.zones) if args.zones else None
        render(args.video, run, args.out or dets.parent / "detections.mp4", zones, cfg["detect"]["kpt_conf"])


if __name__ == "__main__":
    main()
