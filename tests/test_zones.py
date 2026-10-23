import json

import numpy as np
import pytest

from endcap.zone_tool import ZoneEditor
from endcap.zones import ZONE_COLOR, CameraZones, Endcap, draw_zones


def _cam(n=1):
    endcaps = [
        Endcap(f"endcap_{i + 1}", [(10, 10), (50, 10), (50, 40), (10, 40)], [(10, 5), (50, 5)])
        for i in range(n)
    ]
    return CameraZones("cam_a", (100, 60), endcaps)


# --- model / JSON ---------------------------------------------------------

def test_json_roundtrip(tmp_path):
    path = tmp_path / "zones" / "cam_a.json"
    _cam(2).save(path)
    loaded = CameraZones.load(path)
    assert loaded == _cam(2)
    raw = json.loads(path.read_text())
    assert raw["camera_id"] == "cam_a" and raw["image_size"] == [100, 60]


def test_zone_needs_three_points(tmp_path):
    cam = CameraZones("cam_a", (100, 60), [Endcap("e1", [(0, 0), (1, 1)], [(0, 0), (1, 0)])])
    with pytest.raises(ValueError, match="zone needs >= 3"):
        cam.save(tmp_path / "x.json")


def test_face_needs_two_points():
    with pytest.raises(ValueError, match="face needs exactly 2"):
        Endcap("e1", [(0, 0), (1, 0), (1, 1)], [(0, 0)]).validate()


def test_duplicate_ids_rejected():
    cam = _cam(1)
    cam.endcaps.append(cam.endcaps[0])
    with pytest.raises(ValueError, match="duplicate"):
        cam.validate()


def test_frame_size_check():
    cam = _cam()
    cam.check_frame_size(100, 60)
    with pytest.raises(ValueError, match="drawn on 100x60"):
        cam.check_frame_size(1920, 1080)


# --- drawing --------------------------------------------------------------

def test_draw_zones_tints_inside_only():
    img = np.zeros((60, 100, 3), dtype=np.uint8)
    out = draw_zones(img, _cam())
    assert img.sum() == 0  # input untouched
    inside = out[25, 30].astype(int)
    assert inside.sum() > 0 and np.argmax(inside) == np.argmax(ZONE_COLOR)
    assert out[55, 90].sum() == 0  # far outside


# --- editor ---------------------------------------------------------------

def _editor():
    return ZoneEditor(CameraZones("cam_a", (100, 60)))


def test_editor_full_endcap():
    ed = _editor()
    for p in [(10, 10), (50, 10), (50, 40)]:
        ed.click(*p)
    assert ed.finish_polygon()
    assert ed.mode == "face"
    ed.click(10, 5)
    ed.click(50, 5)
    assert len(ed.zones.endcaps) == 1
    e = ed.zones.endcaps[0]
    assert e.endcap_id == "endcap_1" and len(e.zone) == 3 and e.face == [(10, 5), (50, 5)]
    assert ed.mode == "zone" and ed.current.endcap_id == "endcap_2" and ed.dirty


def test_editor_cannot_close_with_two_points():
    ed = _editor()
    ed.click(0, 0)
    ed.click(5, 5)
    assert not ed.finish_polygon()
    assert ed.mode == "zone"


def test_editor_undo_back_into_polygon():
    ed = _editor()
    for p in [(10, 10), (50, 10), (50, 40)]:
        ed.click(*p)
    ed.finish_polygon()
    ed.click(10, 5)
    ed.undo()  # removes face point
    assert ed.current.face == [] and ed.mode == "face"
    ed.undo()  # reopens polygon
    assert ed.mode == "zone"
    ed.undo()  # removes last corner
    assert ed.current.zone == [(10, 10), (50, 10)]


def test_editor_ids_skip_existing_and_delete():
    ed = ZoneEditor(_cam(2))
    assert ed.current.endcap_id == "endcap_3"
    ed.delete_last()
    assert [e.endcap_id for e in ed.zones.endcaps] == ["endcap_1"]
    assert ed.current.endcap_id == "endcap_2"
