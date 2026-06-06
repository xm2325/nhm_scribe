from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
from PIL import Image

from .logging_utils import get_logger
from .metadata import clean_str
from .paths import repo_root, resolve_path

logger = get_logger(__name__)


def safe_filename(value: str, fallback: str = "record") -> str:
    text = clean_str(value) or fallback
    out = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return out.strip("._") or fallback


def _extension_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".tif", ".tiff"} else ".jpg"


RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image_file(path: Path, min_bytes: int = 1) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing_file"
    if path.stat().st_size < min_bytes:
        return False, f"file_too_small:{path.stat().st_size}"
    try:
        with Image.open(path) as image:
            image.verify()
        return True, ""
    except Exception as exc:
        return False, f"invalid_image:{type(exc).__name__}"


def _retry_delay(response: requests.Response | None, attempt: int, backoff_seconds: float) -> float:
    retry_after = response.headers.get("Retry-After", "") if response is not None else ""
    try:
        return max(0.0, float(retry_after))
    except (TypeError, ValueError):
        return min(backoff_seconds * (2 ** attempt), 120.0)


def download_one_diagnostic(
    url: str,
    out_path: Path,
    timeout: int = 20,
    retries: int = 0,
    retry_backoff_seconds: float = 2.0,
    min_bytes: int = 1,
) -> dict[str, Any]:
    if out_path.exists():
        valid, validation_error = validate_image_file(out_path, min_bytes=min_bytes)
        if valid:
            return {
                "ok": True,
                "status": "cached",
                "error": "",
                "attempts": 0,
                "sha256": file_sha256(out_path),
            }
        out_path.unlink()
    if not clean_str(url):
        return {"ok": False, "status": "missing_url", "error": "missing_url", "attempts": 0, "sha256": ""}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix(out_path.suffix + ".part")
    last_status = "download_failed"
    last_error = ""
    attempts = 0
    for attempt in range(max(0, retries) + 1):
        attempts = attempt + 1
        response = None
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "herbarium-scribe-demo/0.2"},
            )
            if response.status_code in RETRYABLE_HTTP_STATUS and attempt < retries:
                time.sleep(_retry_delay(response, attempt, retry_backoff_seconds))
                continue
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type.lower():
                last_status = "not_downloaded_html_response"
                last_error = f"unexpected_content_type:{content_type}"
                break
            temp_path.write_bytes(response.content)
            valid, validation_error = validate_image_file(temp_path, min_bytes=min_bytes)
            if not valid:
                last_status = "invalid_download"
                last_error = validation_error
                temp_path.unlink(missing_ok=True)
                if attempt < retries:
                    time.sleep(_retry_delay(response, attempt, retry_backoff_seconds))
                    continue
                break
            temp_path.replace(out_path)
            return {
                "ok": True,
                "status": "downloaded",
                "error": "",
                "attempts": attempts,
                "sha256": file_sha256(out_path),
            }
        except requests.HTTPError as exc:
            status_code = getattr(exc.response, "status_code", "")
            last_status = f"http_error:{status_code}" if status_code else "http_error"
            last_error = str(exc)[:500]
            if status_code in RETRYABLE_HTTP_STATUS and attempt < retries:
                time.sleep(_retry_delay(exc.response, attempt, retry_backoff_seconds))
                continue
            break
        except Exception as exc:
            last_status = f"error:{type(exc).__name__}"
            last_error = str(exc)[:500]
            if attempt < retries:
                time.sleep(_retry_delay(response, attempt, retry_backoff_seconds))
                continue
            break
        finally:
            temp_path.unlink(missing_ok=True)
    return {
        "ok": False,
        "status": last_status,
        "error": last_error,
        "attempts": attempts,
        "sha256": "",
    }


def download_one(url: str, out_path: Path, timeout: int = 20) -> tuple[bool, str, str]:
    result = download_one_diagnostic(url, out_path, timeout=timeout)
    return bool(result["ok"]), str(result["status"]), str(result["error"])


def _shared_manifest(records: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame | None:
    manifest_value = clean_str(cfg.get("image", {}).get("shared_manifest_csv", ""))
    if not manifest_value:
        return None
    manifest_path = resolve_path(manifest_value, repo_root())
    if not manifest_path.exists():
        raise FileNotFoundError(f"Shared image manifest not found: {manifest_path}")
    shared = pd.read_csv(manifest_path, dtype=str).fillna("")
    required = {"occurrenceID", "image_path", "sha256"}
    missing = required - set(shared.columns)
    if missing:
        raise ValueError(f"Shared image manifest is missing columns: {sorted(missing)}")
    requested = records[["occurrenceID"]].copy()
    out = requested.merge(shared, on="occurrenceID", how="left")
    rows = []
    min_bytes = int(cfg.get("image", {}).get("min_bytes", 1024))
    for _, row in out.iterrows():
        item = row.to_dict()
        image_path = Path(clean_str(item.get("image_path", ""))) if clean_str(item.get("image_path", "")) else None
        valid = False
        error = clean_str(item.get("excluded_reason", ""))
        actual_sha = ""
        if image_path is not None:
            valid, error = validate_image_file(image_path, min_bytes=min_bytes)
            if valid:
                actual_sha = file_sha256(image_path)
                expected_sha = clean_str(item.get("sha256", ""))
                if expected_sha and actual_sha != expected_sha:
                    valid = False
                    error = "sha256_mismatch"
        item["image_path"] = str(image_path) if valid and image_path is not None else ""
        item["image_status"] = "shared_manifest" if valid else "shared_manifest_unavailable"
        item["download_error"] = "" if valid else error or "shared_image_unavailable"
        item["file_size_bytes"] = image_path.stat().st_size if valid and image_path is not None else 0
        item["sha256"] = actual_sha if valid else clean_str(item.get("sha256", ""))
        item["image_valid"] = valid
        item["paired_eligible"] = valid and str(item.get("selected_for_paired_eval", "true")).lower() not in {"false", "0", "no"}
        item["excluded_reason"] = "" if item["paired_eligible"] else item["download_error"] or clean_str(item.get("excluded_reason", ""))
        rows.append(item)
    return pd.DataFrame(rows)


def run_download(records: pd.DataFrame, cfg: dict[str, Any], paths: dict[str, Path]) -> pd.DataFrame:
    shared = _shared_manifest(records, cfg)
    if shared is not None:
        shared.to_csv(paths["processed"] / "image_manifest.csv", index=False)
        return shared

    allow = bool(cfg.get("image", {}).get("allow_download", False))
    timeout = int(cfg.get("image", {}).get("timeout_seconds", 20))
    retries = int(cfg.get("image", {}).get("retries", 0))
    retry_backoff = float(cfg.get("image", {}).get("retry_backoff_seconds", 2.0))
    min_bytes = int(cfg.get("image", {}).get("min_bytes", 1024))
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
            result = download_one_diagnostic(
                url,
                image_path,
                timeout=timeout,
                retries=retries,
                retry_backoff_seconds=retry_backoff,
                min_bytes=min_bytes,
            )
            ok = bool(result["ok"])
            status = str(result["status"])
            error = str(result["error"])
            attempts = int(result["attempts"])
            sha256 = str(result["sha256"])
        else:
            ok, validation_error = validate_image_file(image_path, min_bytes=min_bytes)
            status = "cached" if ok else "skipped_download"
            error = "" if ok else validation_error or ("download_disabled" if url else "missing_url")
            attempts = 0
            sha256 = file_sha256(image_path) if ok else ""
        rows.append({
            "occurrenceID": occ,
            "catalogNumber": clean_str(row.get("catalogNumber")),
            "image_url": url,
            "image_path": str(image_path) if ok else "",
            "image_status": status,
            "download_error": error,
            "file_size_bytes": image_path.stat().st_size if ok and image_path.exists() else 0,
            "download_attempts": attempts,
            "sha256": sha256,
            "image_valid": ok,
            "paired_eligible": ok,
            "excluded_reason": "" if ok else error or status,
        })
    out = pd.DataFrame(rows)
    out.to_csv(paths["processed"] / "image_manifest.csv", index=False)
    return out


def download_real_images(records: pd.DataFrame, cfg: dict[str, Any], paths: dict[str, Path]) -> pd.DataFrame:
    return run_download(records, cfg, paths)
