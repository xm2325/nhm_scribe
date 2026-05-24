from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .logging_utils import get_logger
from .metadata import clean_str

logger = get_logger(__name__)


def download_one(url: str, out_path: Path, timeout: int = 20) -> tuple[bool, str]:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True, "cached"
    if not clean_str(url):
        return False, "missing_url"
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "herbarium-scribe-demo/0.1"})
        r.raise_for_status()
        out_path.write_bytes(r.content)
        return True, "downloaded"
    except Exception as e:
        return False, f"error:{type(e).__name__}"


def run_download(records: pd.DataFrame, cfg: dict[str, Any], paths: dict[str, Path]) -> pd.DataFrame:
    allow = bool(cfg.get("image", {}).get("allow_download", False))
    timeout = int(cfg.get("image", {}).get("timeout_seconds", 20))
    rows = []
    for _, row in records.iterrows():
        occ = clean_str(row.get("occurrenceID"))
        url = clean_str(row.get("image_url"))
        safe = occ.replace(":", "_").replace("/", "_")
        image_path = paths["raw_images"] / f"{safe}.jpg"
        if allow and url:
            ok, status = download_one(url, image_path, timeout=timeout)
        else:
            ok, status = (image_path.exists(), "cached" if image_path.exists() else "skipped_download")
        rows.append({"occurrenceID": occ, "image_url": url, "image_path": str(image_path) if ok else "", "image_status": status})
    out = pd.DataFrame(rows)
    out.to_csv(paths["processed"] / "image_manifest.csv", index=False)
    return out
