from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent.parent / parent_path if path.parent.name == "configs" else path.parent / parent_path
            if not parent_path.exists():
                parent_path = Path(parent)
        base = load_config(parent_path)
        cfg = deep_update(base, cfg)
    cfg["config_path"] = str(path)
    return cfg


def get_nested(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
