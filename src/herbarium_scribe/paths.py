from __future__ import annotations

from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base or repo_root()) / p


def ensure_dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    base = repo_root()
    data_dir = resolve_path(cfg.get("paths", {}).get("data_dir", "data"), base)
    reports_dir = resolve_path(cfg.get("paths", {}).get("reports_dir", "reports"), base)
    paths = {
        "data": data_dir,
        "raw_metadata": data_dir / "raw" / "metadata",
        "raw_images": data_dir / "raw" / "images",
        "interim": data_dir / "interim",
        "crops": data_dir / "interim" / "crops",
        "ocr": data_dir / "interim" / "ocr",
        "llm": data_dir / "interim" / "llm",
        "processed": data_dir / "processed",
        "fixtures": data_dir / "fixtures",
        "reports": reports_dir,
        "figures": reports_dir / "figures",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
