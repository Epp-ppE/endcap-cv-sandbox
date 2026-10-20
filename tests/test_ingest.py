from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest

from endcap.ingest import probe, read_frames, read_frames_from_config

SRC_FPS = 30
N_FRAMES = 90  # 3 s


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """3 s synthetic 30 fps video; frame i is a flat gray image of value 2*i (decodable index)."""
    path = tmp_path_factory.mktemp("vid") / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), SRC_FPS, (64, 48))
    for i in range(N_FRAMES):
        writer.write(np.full((48, 64, 3), 2 * i, dtype=np.uint8))
    writer.release()
    return path


def _decoded_index(frame):
    return int(round(frame.image.mean() / 2))


def test_probe(video):
    info = probe(video)
    assert info.fps == pytest.approx(SRC_FPS)
    assert info.frame_count == N_FRAMES
    assert (info.width, info.height) == (64, 48)
    assert info.duration_s == pytest.approx(3.0)


def test_all_frames_when_no_sampling(video):
    frames = list(read_frames(video))
    assert len(frames) == N_FRAMES
    assert [f.index for f in frames] == list(range(N_FRAMES))


def test_sample_10fps(video):
    frames = list(read_frames(video, sample_fps=10))
    assert [f.index for f in frames] == list(range(0, N_FRAMES, 3))
    assert [f.t for f in frames][:3] == pytest.approx([0.0, 0.1, 0.2])
    # the image actually belongs to the reported index (no off-by-one from grab/retrieve)
    assert all(_decoded_index(f) == f.index for f in frames)


def test_non_integer_ratio(video):
    frames = list(read_frames(video, sample_fps=7))  # 30/7 ≈ 4.29 frames per sample
    assert len(frames) == 21  # 3 s * 7 fps
    gaps = np.diff([f.index for f in frames])
    assert set(gaps) <= {4, 5}


def test_sample_fps_above_source_keeps_all(video):
    assert len(list(read_frames(video, sample_fps=60))) == N_FRAMES


def test_window(video):
    frames = list(read_frames(video, sample_fps=10, start_s=1.0, end_s=2.0))
    assert [f.index for f in frames] == list(range(30, 60, 3))
    assert frames[0].t == pytest.approx(1.0)


def test_wall_clock_timestamps(video):
    start = datetime(2026, 10, 20, 9, 0, 0)
    frames = list(read_frames(video, sample_fps=10, start_time=start, clock_offset_s=2.0))
    assert frames[0].timestamp == start + timedelta(seconds=2.0)
    assert frames[10].timestamp == start + timedelta(seconds=3.0)


def test_no_timestamp_without_start_time(video):
    assert next(read_frames(video)).timestamp is None


def test_from_config(video):
    cfg = {"ingest": {"sample_fps": 5, "start_time": "2026-10-20T09:00:00", "clock_offset_s": 0.0}}
    frames = list(read_frames_from_config(video, cfg))
    assert len(frames) == 15
    assert frames[1].timestamp == datetime(2026, 10, 20, 9, 0, 0, 200000)


def test_missing_file_raises(tmp_path):
    with pytest.raises(IOError):
        probe(tmp_path / "nope.mp4")
