"""Video ingestion: read a video, sample frames at a target fps, attach timestamps.

Each sampled frame keeps
  - `index`: frame number in the source video
  - `t`: seconds since the start of the video (index / source fps)
  - `timestamp`: wall-clock time, if the video's start time is known (needed to join with POS later)

Usage:
    python -m endcap.ingest data/videos/clip.mp4 --fps 10
    python -m endcap.ingest data/videos/clip.mp4 --fps 1 --dump outputs/frames
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    source: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


@dataclass
class Frame:
    index: int
    t: float
    timestamp: datetime | None
    image: np.ndarray


def _open(source: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {source}")
    return cap


def probe(source: str | Path) -> VideoInfo:
    cap = _open(source)
    try:
        return VideoInfo(
            source=str(source),
            fps=cap.get(cv2.CAP_PROP_FPS),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def read_frames(
    source: str | Path,
    sample_fps: float | None = None,
    start_time: datetime | None = None,
    clock_offset_s: float = 0.0,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> Iterator[Frame]:
    """Yield frames sampled at `sample_fps` (None or >= source fps = every frame).

    `start_time` is the wall-clock time of frame 0; `clock_offset_s` corrects a known
    camera clock error (added to every timestamp). `start_s`/`end_s` limit the window read.
    """
    cap = _open(source)
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if src_fps <= 0:
            raise ValueError(f"Video reports no fps: {source}")

        step = 1.0 if not sample_fps or sample_fps >= src_fps else src_fps / sample_fps
        index = int(round(start_s * src_fps))
        if index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        end_index = int(round(end_s * src_fps)) if end_s is not None else None

        next_keep = float(index)
        while end_index is None or index < end_index:
            if not cap.grab():  # grab without decoding; only kept frames are decoded
                break
            if index >= next_keep - 1e-6:
                ok, image = cap.retrieve()
                if not ok:
                    break
                t = index / src_fps
                ts = start_time + timedelta(seconds=t + clock_offset_s) if start_time else None
                yield Frame(index=index, t=t, timestamp=ts, image=image)
                next_keep += step
            index += 1
    finally:
        cap.release()


def read_frames_from_config(source: str | Path, cfg: dict) -> Iterator[Frame]:
    ing = cfg.get("ingest", {})
    start_time = ing.get("start_time")
    if isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time)
    return read_frames(
        source,
        sample_fps=ing.get("sample_fps"),
        start_time=start_time,
        clock_offset_s=ing.get("clock_offset_s", 0.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a video and sample frames.")
    parser.add_argument("source", help="video file path (or stream URL)")
    parser.add_argument("--fps", type=float, default=None, help="sample fps (default: config ingest.sample_fps)")
    parser.add_argument("--start-time", default=None, help="wall-clock time of frame 0, ISO format")
    parser.add_argument("--start", type=float, default=0.0, help="start reading at this second")
    parser.add_argument("--end", type=float, default=None, help="stop reading at this second")
    parser.add_argument("--dump", default=None, help="save sampled frames as JPEGs to this folder")
    args = parser.parse_args()

    from endcap.config import load_config

    cfg = load_config()
    fps = args.fps if args.fps is not None else cfg["ingest"]["sample_fps"]
    start_time = datetime.fromisoformat(args.start_time) if args.start_time else None

    info = probe(args.source)
    print(f"{info.source}: {info.width}x{info.height} @ {info.fps:.2f} fps, "
          f"{info.frame_count} frames ({info.duration_s:.1f} s)")

    dump = Path(args.dump) if args.dump else None
    if dump:
        dump.mkdir(parents=True, exist_ok=True)

    n, first, last = 0, None, None
    for frame in read_frames(args.source, fps, start_time, cfg["ingest"]["clock_offset_s"], args.start, args.end):
        if dump:
            cv2.imwrite(str(dump / f"frame_{frame.index:06d}.jpg"), frame.image)
        first = first or frame
        last = frame
        n += 1

    if n == 0:
        print("No frames read.")
        return
    span = last.t - first.t
    eff = (n - 1) / span if span > 0 else float("nan")
    print(f"Sampled {n} frames at target {fps} fps (effective {eff:.2f} fps), t = {first.t:.2f}..{last.t:.2f} s")
    if last.timestamp:
        print(f"Wall clock: {first.timestamp.isoformat()} .. {last.timestamp.isoformat()}")
    if dump:
        print(f"Saved frames to {dump}")


if __name__ == "__main__":
    main()
