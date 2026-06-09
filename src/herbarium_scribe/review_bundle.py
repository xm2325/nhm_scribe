from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

COMPONENT_BUNDLE_ROOT = "component_aware_eval10"


def bundle_names(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        return []
    with zipfile.ZipFile(path) as archive:
        return archive.namelist()


def read_bundle_csv(
    path: str | Path,
    filename: str,
    *,
    root: str = COMPONENT_BUNDLE_ROOT,
) -> pd.DataFrame:
    member = f"{root}/processed/{filename}"
    with zipfile.ZipFile(path) as archive:
        with archive.open(member) as handle:
            return pd.read_csv(handle, dtype=str).fillna("")


def read_bundle_jsonl(
    path: str | Path,
    filename: str,
    *,
    root: str = COMPONENT_BUNDLE_ROOT,
) -> list[dict]:
    member = f"{root}/processed/{filename}"
    with zipfile.ZipFile(path) as archive:
        text = archive.read(member).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_bundle_bytes(path: str | Path, member: str) -> bytes:
    with zipfile.ZipFile(path) as archive:
        return archive.read(member)
