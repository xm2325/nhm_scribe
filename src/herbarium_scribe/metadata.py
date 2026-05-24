from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .paths import repo_root, resolve_path

REQUIRED_COLUMNS = [
    "occurrenceID", "catalogNumber", "institutionCode", "scientificName",
    "recordedBy", "eventDate", "country", "stateProvince",
    "decimalLatitude", "decimalLongitude", "typeStatus", "image_url",
]


def clean_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    return "" if s.lower() in {"nan", "none", "null"} else s


def load_metadata(cfg: dict[str, Any]) -> pd.DataFrame:
    path = cfg.get("metadata", {}).get("input_csv") or cfg.get("paths", {}).get("metadata_csv")
    if not path:
        path = "data/fixtures/specimen_records.csv"
    csv_path = resolve_path(path, repo_root())
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    if "occurrenceID" not in df.columns or df["occurrenceID"].eq("").any():
        df["occurrenceID"] = [f"urn:row:{i}" for i in range(len(df))]
    for col in df.columns:
        df[col] = df[col].map(clean_str)
    return df


def save_metadata_copy(df: pd.DataFrame, paths: dict[str, Path]) -> Path:
    out = paths["raw_metadata"] / "metadata_loaded.csv"
    df.to_csv(out, index=False)
    return out
