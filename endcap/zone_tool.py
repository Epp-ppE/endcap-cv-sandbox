"""Draw end-cap zones on a video frame and check them on the video.

Draw (opens a window):
    python -m endcap.zone_tool draw data/nvidia_retail/scene_071/camera_0639.mp4 --at 30
        -> saves configs/zones/camera_0639.json (camera id = video file name by default)

    For each end-cap:
      1. left-click the floor zone polygon corners, press ENTER to close it
      2. left-click the 2 ends of the end-cap face line (saved automatically after the 2nd click)

    Keys:  ENTER/SPACE finish polygon   U/BACKSPACE undo point   D delete last end-cap
           [ / ]  frame -/+ 10 s        S save                   Q/ESC quit

Check:
    python -m endcap.zone_tool show <video> --zones configs/zones/camera_0639.json            # play in a window
    python -m endcap.zone_tool show <video> --zones ... --out outputs/zones_check.mp4         # write a video
    python -m endcap.zone_tool show <video> --zones ... --at 30 --out outputs/zones_check.png # one frame
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from endcap.config import ROOT
from endcap.ingest import probe, read_frames
from endcap.zones import CameraZones, Endcap, draw_endcap, draw_zones

ZONES_DIR = ROOT / "configs" / "zones"
WINDOW = "endcap zones"


class ZoneEditor:
    """Click-by-click editing state, independent of any window (so it can be tested)."""

    def __init__(self, zones: CameraZones):
        self.zones = zones
        self.mode = "zone"  # "zone" -> clicking polygon, "face" -> clicking face line
        self.current = Endcap(self._next_id(), [], [])
        self.dirty = False

    def _next_id(self) -> str:
        used = {e.endcap_id for e in self.zones.endcaps}
        n = 1
        while f"endcap_{n}" in used:
            n += 1
        return f"endcap_{n}"

    def click(self, x: int, y: int) -> None:
        if self.mode == "zone":
            self.current.zone.append((int(x), int(y)))
        else:
            self.current.face.append((int(x), int(y)))
            if len(self.current.face) == 2:
                self.zones.endcaps.append(self.current)
                self.dirty = True
                self.mode = "zone"
                self.current = Endcap(self._next_id(), [], [])

    def finish_polygon(self) -> bool:
        if self.mode == "zone" and len(self.current.zone) >= 3:
            self.mode = "face"
            return True
        return False

    def undo(self) -> None:
        if self.mode == "face":
            if self.current.face:
                self.current.face.pop()
            else:
                self.mode = "zone"  # reopen the polygon
        elif self.current.zone:
            self.current.zone.pop()

    def delete_last(self) -> None:
        if self.zones.endcaps:
            self.zones.endcaps.pop()
            self.dirty = True
            if not self.current.zone and not self.current.face:
                self.current = Endcap(self._next_id(), [], [])

    def status(self) -> str:
        n = len(self.zones.endcaps)
        if self.mode == "zone":
            step = f"{self.current.endcap_id}: click zone corners ({len(self.current.zone)}), ENTER to close"
        else:
            step = f"{self.current.endcap_id}: click 2 ends of face line ({len(self.current.face)}/2)"
        return f"{n} saved | {step}"


# --- video helpers --------------------------------------------------------

def grab_frame(source: str | Path, at_s: float) -> np.ndarray:
    return next(read_frames(source, start_s=at_s, end_s=at_s + 1.0)).image


def _display_scale(width: int, height: int, max_w: int = 1600, max_h: int = 900) -> float:
    return min(1.0, max_w / width, max_h / height)


def _hud(image: np.ndarray, text: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# --- commands -------------------------------------------------------------

def draw_cmd(args: argparse.Namespace) -> None:
    info = probe(args.video)
    camera_id = args.camera_id or Path(args.video).stem
    out = Path(args.out) if args.out else ZONES_DIR / f"{camera_id}.json"

    if out.exists():
        zones = CameraZones.load(out)
        zones.check_frame_size(info.width, info.height)
        print(f"Loaded {len(zones.endcaps)} end-cap(s) from {out}")
    else:
        zones = CameraZones(camera_id, (info.width, info.height))

    editor = ZoneEditor(zones)
    at = args.at
    frame = grab_frame(args.video, at)
    scale = _display_scale(info.width, info.height)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            editor.click(round(x / scale), round(y / scale))

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    quit_armed = False

    while True:
        view = draw_zones(frame, editor.zones)
        draw_endcap(view, editor.current, alpha=0.15)
        if scale != 1.0:
            view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        hint = "unsaved changes - Q again to discard" if quit_armed else f"t={at:.0f}s | {editor.status()}"
        _hud(view, hint)
        cv2.imshow(WINDOW, view)

        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue
        if key in (ord("q"), 27):
            if editor.dirty and not quit_armed:
                quit_armed = True
                continue
            break
        quit_armed = False
        if key in (13, 32):
            editor.finish_polygon()
        elif key in (ord("u"), 8):
            editor.undo()
        elif key == ord("d"):
            editor.delete_last()
        elif key == ord("s"):
            editor.zones.save(out)
            editor.dirty = False
            print(f"Saved {len(editor.zones.endcaps)} end-cap(s) to {out}")
        elif key in (ord("["), ord("]")):
            at = max(0.0, min(info.duration_s - 1, at + (10 if key == ord("]") else -10)))
            frame = grab_frame(args.video, at)

    cv2.destroyAllWindows()


def show_cmd(args: argparse.Namespace) -> None:
    zones = CameraZones.load(args.zones)
    info = probe(args.video)
    zones.check_frame_size(info.width, info.height)
    out = Path(args.out) if args.out else None

    if out and out.suffix.lower() in (".png", ".jpg", ".jpeg"):
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), draw_zones(grab_frame(args.video, args.at), zones))
        print(f"Wrote {out}")
        return

    writer = None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (info.width, info.height))
    scale = _display_scale(info.width, info.height)
    end = args.at + args.duration if args.duration else None

    for frame in read_frames(args.video, sample_fps=args.fps, start_s=args.at, end_s=end):
        view = draw_zones(frame.image, zones)
        if writer:
            writer.write(view)
            continue
        if scale != 1.0:
            view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        _hud(view, f"t={frame.t:.1f}s | Q/ESC to quit")
        cv2.imshow(WINDOW, view)
        if (cv2.waitKey(max(1, int(1000 / args.fps))) & 0xFF) in (ord("q"), 27):
            break

    if writer:
        writer.release()
        print(f"Wrote {out}")
    else:
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw and check end-cap zones.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("draw", help="draw zones on a frame of the video")
    d.add_argument("video")
    d.add_argument("--camera-id", default=None, help="default: video file name")
    d.add_argument("--out", default=None, help="default: configs/zones/<camera_id>.json")
    d.add_argument("--at", type=float, default=0.0, help="frame time in seconds")
    d.set_defaults(func=draw_cmd)

    s = sub.add_parser("show", help="overlay zones on the video")
    s.add_argument("video")
    s.add_argument("--zones", required=True)
    s.add_argument("--at", type=float, default=0.0, help="start time in seconds")
    s.add_argument("--duration", type=float, default=None, help="seconds to show (default: to the end)")
    s.add_argument("--fps", type=float, default=10.0)
    s.add_argument("--out", default=None, help=".mp4 to write a video, .png/.jpg for one frame")
    s.set_defaults(func=show_cmd)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
