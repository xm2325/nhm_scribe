from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .metadata import clean_str

COUNTRY_NORMALISATION = {
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "usa": "United States",
    "u.s.a.": "United States",
    "us": "United States",
    "united states of america": "United States",
}


def canonical_scientific_name(name: str) -> str:
    name = clean_str(name)
    name = re.sub(r"\s+", " ", name)
    # Keep a conservative binomial canonical form; do not remove authorship aggressively.
    m = re.match(r"^([A-Z][a-zA-Z\-]+\s+[a-z][a-zA-Z\-]+)", name)
    return m.group(1) if m else name


def _looks_like_binomial(name: str) -> bool:
    return bool(re.match(r"^[A-Z][a-zA-Z\-]+\s+[a-z][a-zA-Z\-]+(?:\s|$)", clean_str(name)))


def normalise_country(country: str) -> str:
    c = clean_str(country)
    return COUNTRY_NORMALISATION.get(c.lower(), c)


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _gbif_match(
    name: str,
    config: dict[str, Any],
    cache: dict[str, Any],
) -> dict[str, Any]:
    if name in cache and not (
        isinstance(cache[name], dict) and cache[name].get("_error")
    ):
        value = cache[name]
        return value if isinstance(value, dict) else {}
    base_url = clean_str(config.get("gbif_base_url", "https://api.gbif.org/v1"))
    timeout = float(config.get("gbif_timeout_seconds", 20))
    retries = int(config.get("gbif_retries", 2))
    last_error = ""
    for attempt in range(retries + 1):
        try:
            response = requests.get(
                f"{base_url.rstrip('/')}/species/match",
                params={"name": name},
                timeout=timeout,
            )
            response.raise_for_status()
            value = response.json()
            cache[name] = value if isinstance(value, dict) else {}
            return cache[name]
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= retries:
                break
    cache[name] = {"_error": last_error}
    return cache[name]


def _safe_gbif_name(
    original: str,
    match: dict[str, Any],
    min_confidence: float,
) -> tuple[str, bool]:
    accepted = clean_str(match.get("scientificName", ""))
    confidence = float(match.get("confidence", 0) or 0)
    if not accepted or confidence < min_confidence:
        return original, False
    original_canonical = canonical_scientific_name(original).lower()
    accepted_canonical = canonical_scientific_name(accepted).lower()
    if not original_canonical or original_canonical != accepted_canonical:
        return original, False
    return accepted, accepted != original


def reconcile_row(
    row: pd.Series,
    config: dict[str, Any] | None = None,
    gbif_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    verbatim = clean_str(row.get("scientificName", ""))
    resolved = verbatim
    gbif_match: dict[str, Any] = {}
    corrected = False
    allowed_methods = set(config.get("gbif_methods", []))
    method_allowed = not allowed_methods or clean_str(row.get("method", "")) in allowed_methods
    if (
        verbatim
        and _looks_like_binomial(verbatim)
        and method_allowed
        and bool(config.get("use_gbif_api", False))
    ):
        gbif_match = _gbif_match(verbatim, config, gbif_cache if gbif_cache is not None else {})
        resolved, corrected = _safe_gbif_name(
            verbatim,
            gbif_match,
            float(config.get("gbif_min_confidence", 90)),
        )
    canonical = canonical_scientific_name(resolved)
    if not verbatim:
        status = "not_provided"
    elif corrected:
        status = "gbif_authorship_corrected"
    elif gbif_match and not gbif_match.get("_error"):
        status = "gbif_checked_unchanged"
    else:
        status = "local_canonical" if canonical else "unmatched"
    warning = "" if canonical or not verbatim else "taxonomy_unmatched"
    country = normalise_country(row.get("country", ""))
    return {
        "scientificName_resolved": resolved,
        "scientificName_verbatim": verbatim,
        "scientificName_canonical": canonical,
        "taxonomy_match_status": status,
        "taxonomy_warning": warning,
        "taxonomy_match_confidence": gbif_match.get("confidence", ""),
        "taxonomy_match_type": gbif_match.get("matchType", ""),
        "taxonomy_matched_name": gbif_match.get("scientificName", ""),
        "taxonomy_correction_applied": corrected,
        "taxonomy_error": gbif_match.get("_error", ""),
        "country_verbatim": clean_str(row.get("country", "")),
        "country_normalised": country,
        "stateProvince_normalised": clean_str(row.get("stateProvince", "")),
    }


def reconcile_dataframe(
    pred_df: pd.DataFrame,
    paths: dict[str, Path] | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    config = config or {}
    rec_config = config.get("reconciliation", config)
    cache_path_value = clean_str(rec_config.get("gbif_cache_path", ""))
    cache_path = (
        Path(cache_path_value)
        if cache_path_value
        else ((paths["processed"] / "gbif_match_cache.json") if paths else Path("gbif_match_cache.json"))
    )
    gbif_cache = _load_cache(cache_path)
    rows = []
    for _, row in pred_df.iterrows():
        rows.append(reconcile_row(row, rec_config, gbif_cache))
    enrichment = pd.DataFrame(rows)
    rec = pd.concat([pred_df.reset_index(drop=True), enrichment], axis=1)
    if "scientificName_resolved" in rec.columns:
        rec["scientificName"] = rec["scientificName_resolved"]
    if bool(rec_config.get("use_gbif_api", False)):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(gbif_cache, indent=2, ensure_ascii=False), encoding="utf-8")
    if paths:
        rec.to_csv(paths["processed"] / "extractions_flat_reconciled.csv", index=False)
        columns = [
            "occurrenceID",
            "method",
            "scientificName_verbatim",
            "scientificName_resolved",
            "scientificName_canonical",
            "taxonomy_match_status",
            "taxonomy_match_confidence",
            "taxonomy_match_type",
            "taxonomy_matched_name",
            "taxonomy_correction_applied",
            "taxonomy_warning",
            "taxonomy_error",
            "country_normalised",
            "stateProvince_normalised",
        ]
        rec[[column for column in columns if column in rec.columns]].to_csv(
            paths["processed"] / "reconciliation.csv",
            index=False,
        )
    return rec
