from pathlib import Path

from endcap.config import ROOT, load_config, resolve_device


def test_default_config_loads():
    cfg = load_config()
    assert cfg["device"] == "auto"
    assert Path(cfg["paths"]["data_dir"]) == ROOT / "data"


def test_experiment_file_overrides_default(tmp_path):
    exp = tmp_path / "exp.yaml"
    exp.write_text("device: cpu\npaths:\n  output_dir: outputs/exp1\n", encoding="utf-8")
    cfg = load_config(exp)
    assert cfg["device"] == "cpu"
    assert Path(cfg["paths"]["output_dir"]) == ROOT / "outputs" / "exp1"
    assert Path(cfg["paths"]["data_dir"]) == ROOT / "data"  # untouched keys kept


def test_overrides_dict_applied_last():
    cfg = load_config(overrides={"device": "cuda"})
    assert cfg["device"] == "cuda"


def test_resolve_device():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cuda", "cpu"}
