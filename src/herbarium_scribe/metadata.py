from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from .paths import repo_root, resolve_path

REQUIRED_COLUMNS = [
    "occurrenceID", "catalogNumber", "institutionCode", "scientificName",
    "recordedBy", "eventDate", "country", "stateProvince",
    "decimalLatitude", "decimalLongitude", "typeStatus", "image_url",
]
RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}


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


def _normalise_record_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    if "occurrenceID" not in df.columns or df["occurrenceID"].eq("").any():
        df["occurrenceID"] = [f"urn:row:{i}" for i in range(len(df))]
    for col in df.columns:
        df[col] = df[col].map(clean_str)
    return df


def _zenodo_csv_urls(record_id: str, explicit_url: str = "") -> list[str]:
    preferred = ["Data and Links excl extensions.csv", "Data and Links.csv"]
    urls = [clean_str(explicit_url)] if clean_str(explicit_url) else []
    for name in preferred:
        quoted = quote(name)
        urls.extend([
            f"https://zenodo.org/records/{record_id}/files/{quoted}?download=1",
            f"https://zenodo.org/api/records/{record_id}/files/{quoted}/content",
        ])
    return list(dict.fromkeys(url for url in urls if url))


def _retry_delay(response: requests.Response | None, attempt: int, backoff_seconds: float) -> float:
    retry_after = response.headers.get("Retry-After", "") if response is not None else ""
    try:
        return min(max(0.0, float(retry_after)), 120.0)
    except (TypeError, ValueError):
        return min(backoff_seconds * (2 ** attempt), 120.0)


def _download_zenodo_csv(
    urls: list[str],
    cache_path: Path,
    timeout: int,
    retries: int,
    retry_backoff_seconds: float,
) -> None:
    temp_path = cache_path.with_suffix(cache_path.suffix + ".part")
    errors: list[str] = []
    for url in urls:
        for attempt in range(max(0, retries) + 1):
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
                temp_path.write_bytes(response.content)
                header = pd.read_csv(temp_path, nrows=0)
                if "jpegURL" not in header.columns:
                    raise ValueError("downloaded CSV does not contain jpegURL")
                temp_path.replace(cache_path)
                return
            except Exception as exc:
                status_code = getattr(getattr(exc, "response", None), "status_code", "")
                errors.append(f"{url} attempt {attempt + 1}: {type(exc).__name__} {status_code}".strip())
                temp_path.unlink(missing_ok=True)
                retryable = not status_code or status_code in RETRYABLE_HTTP_STATUS
                if retryable and attempt < retries:
                    time.sleep(_retry_delay(response, attempt, retry_backoff_seconds))
                    continue
                break
    raise RuntimeError("Zenodo metadata download failed after retries: " + "; ".join(errors[-8:]))


def load_zenodo_metadata(cfg: dict[str, Any], paths: dict[str, Path] | None = None) -> pd.DataFrame:
    mcfg = cfg.get("metadata", {})
    record_id = clean_str(mcfg.get("record_id", "6372393")) or "6372393"
    timeout = int(mcfg.get("timeout_seconds", 60))
    retries = int(mcfg.get("retries", 3))
    retry_backoff_seconds = float(mcfg.get("retry_backoff_seconds", 10))
    cache_path = resolve_path(mcfg.get("cache_csv", f"data/raw/metadata/zenodo_{record_id}_links.csv"), repo_root())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists() or bool(mcfg.get("force_download", False)):
        _download_zenodo_csv(
            _zenodo_csv_urls(record_id, clean_str(mcfg.get("csv_url", ""))),
            cache_path,
            timeout=timeout,
            retries=retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
    df = pd.read_csv(cache_path, dtype=str).fillna("")
    if "jpegURL" not in df.columns:
        df["jpegURL"] = ""
    df = df[df["jpegURL"].map(clean_str).astype(bool)].copy()
    if "persistentID" in df.columns:
        df["occurrenceID"] = df["persistentID"].map(clean_str)
    elif "occurrenceID" in df.columns:
        df["occurrenceID"] = df["occurrenceID"].map(clean_str)
    else:
        df["occurrenceID"] = [f"zenodo:{record_id}:{i}" for i in range(len(df))]
    df["image_url"] = df["jpegURL"].map(clean_str)
    df["metadata_source"] = f"zenodo:{record_id}"
    df["dataset_source"] = f"Zenodo {record_id}"
    df["fixture_label_text"] = ""
    df = _normalise_record_columns(df)
    keep = list(dict.fromkeys(REQUIRED_COLUMNS + [
        "persistentID", "jpegURL", "tiffURL", "jsonURL", "pngSegAllURL", "pngSegSelURL",
        "DOI", "metadata_source", "dataset_source", "fixture_label_text",
    ]))
    return df[[col for col in keep if col in df.columns]].copy()


def resolve_rbge_image_url(cfg: dict[str, Any]) -> tuple[str, str]:
    mcfg = cfg.get("metadata", {})
    zoom_url = clean_str(mcfg.get("rbge_zoom_url", "https://data.rbge.org.uk/search/herbarium/?specimen_num=625512&cfg=zoom.cfg&filename=E00633257.zip"))
    timeout = int(mcfg.get("timeout_seconds", 30))
    try:
        resp = requests.get(zoom_url, timeout=timeout, headers={"User-Agent": "herbarium-scribe-demo/0.1"})
        resp.raise_for_status()
        text = resp.text
        match = re.search(r"https?://[^\"'\s<>]+?\.(?:jpg|jpeg|png)", text, flags=re.IGNORECASE)
        if match:
            return match.group(0), "resolved_from_rbge_zoom_html"
    except Exception as e:
        return "", f"rbge_zoom_resolution_failed:{type(e).__name__}"
    return "", "rbge_zoom_direct_image_not_found"


def load_rbge_smoke_metadata(cfg: dict[str, Any]) -> pd.DataFrame:
    image_url, image_note = resolve_rbge_image_url(cfg)
    mcfg = cfg.get("metadata", {})
    if not image_url and bool(mcfg.get("allow_zenodo_image_fallback", True)):
        image_url = "https://zenodo.org/record/1484146/files/E00633257.jpg"
        image_note = f"{image_note};using_zenodo_jpeg_for_same_barcode"
    row = {
        "occurrenceID": "http://data.rbge.org.uk/herb/E00633257",
        "persistentID": "http://data.rbge.org.uk/herb/E00633257",
        "catalogNumber": "E00633257",
        "institutionCode": "E",
        "scientificName": "Abelia forrestii (Diels.) W.W.Sm.",
        "recordedBy": "",
        "eventDate": "",
        "country": "China",
        "stateProvince": "",
        "decimalLatitude": "",
        "decimalLongitude": "",
        "typeStatus": "",
        "image_url": image_url,
        "jpegURL": image_url,
        "jsonURL": "https://zenodo.org/record/1484146/files/E00633257.json",
        "DOI": "http://dx.doi.org/10.5281/zenodo.1484146",
        "metadata_source": "rbge:E00633257",
        "dataset_source": "RBGE E00633257 smoke",
        "rbge_metadata_url": "https://data.rbge.org.uk/search/herbarium/?cfg=fulldetails.cfg&barcode=E00633257",
        "rbge_zoom_url": "https://data.rbge.org.uk/search/herbarium/?specimen_num=625512&cfg=zoom.cfg&filename=E00633257.zip",
        "rbge_image_resolution_status": image_note,
        "fixture_label_text": "",
    }
    return _normalise_record_columns(pd.DataFrame([row]))


def load_metadata(cfg: dict[str, Any]) -> pd.DataFrame:
    source = clean_str(cfg.get("metadata", {}).get("source", "")).lower()
    if source in {"zenodo", "zenodo_6372393"}:
        return load_zenodo_metadata(cfg)
    if source in {"rbge_e00633257", "rbge_smoke"}:
        return load_rbge_smoke_metadata(cfg)
    path = cfg.get("metadata", {}).get("input_csv") or cfg.get("paths", {}).get("metadata_csv")
    if not path:
        path = "data/fixtures/specimen_records.csv"
    csv_path = resolve_path(path, repo_root())
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    return _normalise_record_columns(df)


def save_metadata_copy(df: pd.DataFrame, paths: dict[str, Path]) -> Path:
    out = paths["raw_metadata"] / "metadata_loaded.csv"
    df.to_csv(out, index=False)
    return out
