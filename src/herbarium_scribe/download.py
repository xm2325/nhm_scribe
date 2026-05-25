from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

from .logging_utils import get_logger
from .metadata import clean_str

logger = get_logger(__name__)


def safe_filename(value: str, fallback: str = "record") -> str:
    text = clean_str(value) or fallback
    out = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return out.strip("._") or fallback


def _extension_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".tif", ".tiff"} else ".jpg"


def download_one(url: str, out_path: Path, timeout: int = 20) -> tuple[bool, str, str]:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True, "cached", ""
    if not clean_str(url):
        return False, "missing_url", "missing_url"
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "herbarium-scribe-demo/0.1"})
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "")
        if "text/html" in content_type.lower():
            return False, "not_downloaded_html_response", f"unexpected_content_type:{content_type}"
        out_path.write_bytes(r.content)
        return True, "downloaded", ""
    except Exception as e:
        return False, f"error:{type(e).__name__}", str(e)[:500]


def run_download(records: pd.DataFrame, cfg: dict[str, Any], paths: dict[str, Path]) -> pd.DataFrame:
    allow = bool(cfg.get("image", {}).get("allow_download", False))
    timeout = int(cfg.get("image", {}).get("timeout_seconds", 20))
    subdir = clean_str(cfg.get("image", {}).get("output_subdir", ""))
    image_dir = paths["raw_images"] / subdir if subdir else paths["raw_images"]
    image_dir.mkdir(parents=True, exist_ok=True)
    force = bool(cfg.get("image", {}).get("force", False))
    rows = []
    for _, row in records.iterrows():
        occ = clean_str(row.get("occurrenceID"))
        url = clean_str(row.get("image_url"))
        safe = safe_filename(clean_str(row.get("catalogNumber")) or occ)
        image_path = image_dir / f"{safe}{_extension_from_url(url)}"
        if force and image_path.exists():
            image_path.unlink()
        if allow and url:
            ok, status, error = download_one(url, image_path, timeout=timeout)
        else:
            ok = image_path.exists()
            status = "cached" if ok else "skipped_download"
            error = "" if ok else ("download_disabled" if url else "missing_url")
        rows.append({
            "occurrenceID": occ,
            "catalogNumber": clean_str(row.get("catalogNumber")),
            "image_url": url,
            "image_path": str(image_path) if ok else "",
            "image_status": status,
            "download_error": error,
            "file_size_bytes": image_path.stat().st_size if ok and image_path.exists() else 0,
        })
    out = pd.DataFrame(rows)
    out.to_csv(paths["processed"] / "image_manifest.csv", index=False)
    return out


def download_real_images(records: pd.DataFrame, cfg: dict[str, Any], paths: dict[str, Path]) -> pd.DataFrame:
    return run_download(records, cfg, paths)
