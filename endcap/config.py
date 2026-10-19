"""Load YAML config: configs/default.yaml, optionally overridden by an experiment file."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Return default config merged with `path` (if given) and then `overrides` (if given).

    Relative paths under `paths` are resolved against the repo root.
    """
    cfg = _read_yaml(DEFAULT_CONFIG)
    if path is not None:
        cfg = _deep_merge(cfg, _read_yaml(Path(path)))
    if overrides:
        cfg = _deep_merge(cfg, overrides)

    cfg["paths"] = {k: str((ROOT / v).resolve()) for k, v in cfg.get("paths", {}).items()}
    return cfg
