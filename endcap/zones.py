"""End-cap zones per camera: data model, JSON load/save, and drawing on frames.

One JSON file per camera, e.g. configs/zones/camera_0639.json:

{
  "camera_id": "camera_0639",
  "image_size": [1920, 1080],
  "endcaps": [
    {
      "endcap_id": "endcap_1",
      "zone": [[x, y], ...],        # floor polygon in front of the end-cap (>= 3 points)
      "face": [[x, y], [x, y]]      # line along the front of the end-cap (facing target)
    }
  ],
  "homography": null                # optional 3x3 image -> floor matrix (for later)
}

Coordinates are pixels in the original video resolution.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

Point = tuple[int, int]


@dataclass
class Endcap:
    endcap_id: str
    zone: list[Point]
    face: list[Point]

    def validate(self) -> None:
        if len(self.zone) < 3:
            raise ValueError(f"{self.endcap_id}: zone needs >= 3 points, got {len(self.zone)}")
        if len(self.face) != 2:
            raise ValueError(f"{self.endcap_id}: face needs exactly 2 points, got {len(self.face)}")


@dataclass
class CameraZones:
    camera_id: str
    image_size: tuple[int, int]  # (width, height)
    endcaps: list[Endcap] = field(default_factory=list)
    homography: list[list[float]] | None = None

    def validate(self) -> None:
        ids = [e.endcap_id for e in self.endcaps]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.camera_id}: duplicate endcap_id in {ids}")
        for e in self.endcaps:
            e.validate()

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CameraZones":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        endcaps = [
            Endcap(e["endcap_id"], [tuple(p) for p in e["zone"]], [tuple(p) for p in e["face"]])
            for e in d.get("endcaps", [])
        ]
        cz = cls(d["camera_id"], tuple(d["image_size"]), endcaps, d.get("homography"))
        cz.validate()
        return cz

    def check_frame_size(self, width: int, height: int) -> None:
        if (width, height) != tuple(self.image_size):
            raise ValueError(
                f"{self.camera_id}: zones drawn on {self.image_size[0]}x{self.image_size[1]}, "
                f"video is {width}x{height}"
            )


# --- drawing -------------------------------------------------------------

ZONE_COLOR = (0, 200, 255)   # BGR orange
FACE_COLOR = (255, 80, 255)  # BGR magenta


def draw_endcap(image: np.ndarray, endcap: Endcap, alpha: float = 0.25, thickness: int = 2) -> None:
    """Draw one end-cap in place: filled translucent zone, outline, face line, label."""
    pts = np.array(endcap.zone, dtype=np.int32)
    if len(pts) >= 3:
        overlay = image.copy()
        cv2.fillPoly(overlay, [pts], ZONE_COLOR)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)
    if len(pts) >= 2:
        cv2.polylines(image, [pts], isClosed=len(pts) >= 3, color=ZONE_COLOR, thickness=thickness)
    for p in endcap.zone:
        cv2.circle(image, tuple(map(int, p)), thickness + 2, ZONE_COLOR, -1)

    if len(endcap.face) == 2:
        a, b = (tuple(map(int, p)) for p in endcap.face)
        cv2.line(image, a, b, FACE_COLOR, thickness * 2)
    for p in endcap.face:
        cv2.circle(image, tuple(map(int, p)), thickness + 3, FACE_COLOR, -1)

    anchor = endcap.zone[0] if endcap.zone else (endcap.face[0] if endcap.face else None)
    if anchor is not None:
        x, y = map(int, anchor)
        cv2.putText(image, endcap.endcap_id, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, endcap.endcap_id, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)


def draw_zones(image: np.ndarray, zones: CameraZones, alpha: float = 0.25) -> np.ndarray:
    """Return a copy of `image` with every end-cap zone and face line drawn on it."""
    out = image.copy()
    for e in zones.endcaps:
        draw_endcap(out, e, alpha=alpha)
    return out
