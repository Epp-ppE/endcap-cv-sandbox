from datetime import datetime

import numpy as np
import pytest

from endcap.config import ROOT
from endcap.detect import KP, DetectionRun, FrameDetections, draw_detections

WEIGHTS = ROOT / "models" / "yolo26n-pose.pt"
SAMPLE_VIDEO = ROOT / "data" / "nvidia_retail" / "scene_071" / "camera_0639.mp4"


def _fd(frame, n, ts=None):
    rng = np.random.default_rng(frame)
    boxes = np.tile(np.array([10, 20, 50, 120], np.float32), (n, 1)) + np.arange(n)[:, None] * 60
    kpts = rng.uniform(0, 100, (n, 17, 3)).astype(np.float32)
    kpts[..., 2] = rng.uniform(0, 1, (n, 17))
    return FrameDetections(frame, frame / 30, ts, boxes, rng.uniform(0.3, 1, n).astype(np.float32), kpts)


def test_save_load_roundtrip_keeps_empty_frames(tmp_path):
    ts = datetime(2026, 10, 23, 9, 0, 0)
    run = DetectionRun({"video": "x.mp4", "sample_fps": 10}, [_fd(0, 2, ts), _fd(3, 0, ts), _fd(6, 3, ts)])
    run.save(tmp_path)
    loaded = DetectionRun.load(tmp_path)

    assert [f.frame for f in loaded.frames] == [0, 3, 6]
    assert [len(f) for f in loaded.frames] == [2, 0, 3]
    assert loaded.frames[0].timestamp == ts
    for a, b in zip(run.frames, loaded.frames):
        np.testing.assert_allclose(a.boxes, b.boxes, rtol=1e-6)
        np.testing.assert_allclose(a.scores, b.scores, rtol=1e-6)
        np.testing.assert_allclose(a.keypoints, b.keypoints, rtol=1e-6)
    assert loaded.meta["sample_fps"] == 10


def test_no_timestamps_roundtrip(tmp_path):
    DetectionRun({}, [_fd(0, 1)]).save(tmp_path)
    assert DetectionRun.load(tmp_path).frames[0].timestamp is None


def test_detections_table_columns():
    df = DetectionRun({}, [_fd(0, 2)]).detections_table()
    assert list(df.columns[:7]) == ["frame", "det", "x1", "y1", "x2", "y2", "score"]
    assert "left_ankle_c" in df.columns and len(df.columns) == 7 + 17 * 3
    assert df["det"].tolist() == [0, 1]


def test_draw_detections_only_draws_visible_keypoints():
    img = np.zeros((200, 200, 3), np.uint8)
    kpts = np.zeros((1, 17, 3), np.float32)
    kpts[0, KP["nose"]] = [100, 100, 0.9]
    kpts[0, KP["left_ankle"]] = [150, 150, 0.1]  # below threshold
    fd = FrameDetections(0, 0.0, None, np.array([[5, 5, 195, 195]], np.float32), np.array([0.9]), kpts)
    out = draw_detections(img, fd, kpt_conf=0.5)
    assert out[100, 100].sum() > 0
    assert out[150, 150].sum() == 0
    assert img.sum() == 0


@pytest.mark.skipif(not WEIGHTS.exists(), reason="model weights not downloaded")
def test_detector_on_blank_and_real_frame():
    from endcap.detect import PoseDetector
    from endcap.ingest import read_frames

    det = PoseDetector(str(WEIGHTS), ROOT / "models", imgsz=640)
    (b, s, k), = det.predict([np.zeros((480, 640, 3), np.uint8)])
    assert b.shape == (0, 4) and s.shape == (0,) and k.shape == (0, 17, 3)

    if SAMPLE_VIDEO.exists():
        frame = next(read_frames(SAMPLE_VIDEO, start_s=30, end_s=31))
        fds = list(det.detect_frames([frame]))
        assert fds[0].frame == frame.index
        assert len(fds[0]) >= 4  # 7 people visible at t=30 s
        assert fds[0].keypoints.shape[1:] == (17, 3)
