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


def _catalog_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", clean_str(value)).upper()


def _safe_catalog_extension(
    original: str,
    candidates: list[dict[str, Any]],
    max_extension_chars: int = 4,
) -> tuple[str, bool, str]:
    original = clean_str(original)
    original_key = _catalog_key(original)
    if not original_key:
        return original, False, "not_provided"
    if not re.fullmatch(r"[A-Z]{1,4}\d{4,14}", original_key):
        return original, False, "original_not_structured"
    matches: dict[str, str] = {}
    for item in candidates:
        value = clean_str(item.get("value", ""))
        key = _catalog_key(value)
        extension_length = len(key) - len(original_key)
        if not 1 <= extension_length <= max(1, int(max_extension_chars)):
            continue
        if not key.startswith(original_key):
            continue
        if not re.fullmatch(r"[A-Z]{1,4}\d{6,14}", key):
            continue
        matches.setdefault(key, value)
    if len(matches) != 1:
        return original, False, "ambiguous_extension" if matches else "no_safe_extension"
    return next(iter(matches.values())), True, "strict_extension_applied"


def _catalog_candidates_by_occurrence(
    paths: dict[str, Path] | None,
) -> dict[str, list[dict[str, Any]]]:
    if not paths:
        return {}
    ocr_path = paths["processed"] / "ocr_by_region.csv"
    if not ocr_path.exists():
        return {}
    ocr = pd.read_csv(ocr_path, dtype=str).fillna("")
    if "ocr_engine" not in ocr.columns:
        return {}
    ensemble = ocr[
        ocr["ocr_engine"].astype(str).eq("tesseract_catalog_number_ensemble")
    ]
    by_occurrence: dict[str, dict[str, dict[str, Any]]] = {}
    for _, row in ensemble.iterrows():
        occurrence_id = clean_str(row.get("occurrenceID", ""))
        if not occurrence_id:
            continue
        items = []
        raw_json = clean_str(row.get("ocr_ensemble_candidates_json", ""))
        if raw_json:
            try:
                parsed = json.loads(raw_json)
                items = parsed if isinstance(parsed, list) else []
            except Exception:
                items = []
        if not items:
            items = [
                {"value": value}
                for value in str(row.get("ocr_text", "")).splitlines()
                if clean_str(value)
            ]
        occurrence_items = by_occurrence.setdefault(occurrence_id, {})
        for item in items:
            value = clean_str(item.get("value", ""))
            key = _catalog_key(value)
            if not key:
                continue
            existing = occurrence_items.setdefault(
                key,
                {
                    "value": value,
                    "normalised": key,
                    "votes": 0,
                    "score": 0,
                    "sources": [],
                },
            )
            existing["votes"] = max(
                int(existing.get("votes", 0) or 0),
                int(item.get("votes", 0) or 0),
            )
            existing["score"] = max(
                float(existing.get("score", 0) or 0),
                float(item.get("score", 0) or 0),
            )
            for source in item.get("sources", []) or []:
                if source not in existing["sources"]:
                    existing["sources"].append(source)
    return {
        occurrence_id: sorted(
            items.values(),
            key=lambda item: (-float(item["score"]), -int(item["votes"]), item["normalised"]),
        )
        for occurrence_id, items in by_occurrence.items()
    }


def reconcile_row(
    row: pd.Series,
    config: dict[str, Any] | None = None,
    gbif_cache: dict[str, Any] | None = None,
    catalog_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config = config or {}
    catalog_candidates = catalog_candidates or []
    catalog_verbatim = clean_str(row.get("catalogNumber", ""))
    catalog_resolved = catalog_verbatim
    catalog_corrected = False
    catalog_status = "disabled"
    catalog_methods = set(config.get("catalog_number_resolver_methods", []))
    catalog_method_allowed = (
        not catalog_methods
        or clean_str(row.get("method", "")) in catalog_methods
    )
    if bool(config.get("use_catalog_number_resolver", False)) and catalog_method_allowed:
        catalog_resolved, catalog_corrected, catalog_status = _safe_catalog_extension(
            catalog_verbatim,
            catalog_candidates,
            int(config.get("catalog_number_max_extension_chars", 4)),
        )
    elif bool(config.get("use_catalog_number_resolver", False)):
        catalog_status = "not_enabled_for_method"
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
        "catalogNumber_resolved": catalog_resolved,
        "catalogNumber_verbatim": catalog_verbatim,
        "catalog_number_resolution_status": catalog_status,
        "catalog_number_correction_applied": catalog_corrected,
        "catalog_number_candidate_count": len(catalog_candidates),
        "catalog_number_candidates": " | ".join(
            clean_str(item.get("value", ""))
            for item in catalog_candidates
            if clean_str(item.get("value", ""))
        ),
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
    catalog_candidates = (
        _catalog_candidates_by_occurrence(paths)
        if bool(rec_config.get("use_catalog_number_resolver", False))
        else {}
    )
    rows = []
    for _, row in pred_df.iterrows():
        rows.append(
            reconcile_row(
                row,
                rec_config,
                gbif_cache,
                catalog_candidates.get(clean_str(row.get("occurrenceID", "")), []),
            )
        )
    enrichment = pd.DataFrame(rows)
    rec = pd.concat([pred_df.reset_index(drop=True), enrichment], axis=1)
    if "catalogNumber_resolved" in rec.columns:
        corrected = rec["catalog_number_correction_applied"].astype(bool)
        rec.loc[corrected, "catalogNumber"] = rec.loc[corrected, "catalogNumber_resolved"]
        if "catalogNumber_evidence_span" in rec.columns:
            rec.loc[corrected, "catalogNumber_evidence_span"] = rec.loc[
                corrected,
                "catalogNumber_resolved",
            ]
        if "catalogNumber_confidence" in rec.columns:
            confidence = pd.to_numeric(
                rec.loc[corrected, "catalogNumber_confidence"],
                errors="coerce",
            ).fillna(0.0)
            resolved_confidence = confidence.clip(lower=0.85)
            if pd.api.types.is_string_dtype(rec["catalogNumber_confidence"].dtype):
                rec.loc[corrected, "catalogNumber_confidence"] = resolved_confidence.map(
                    lambda value: f"{value:g}"
                )
            else:
                rec.loc[corrected, "catalogNumber_confidence"] = resolved_confidence
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
            "catalogNumber_verbatim",
            "catalogNumber_resolved",
            "catalog_number_resolution_status",
            "catalog_number_correction_applied",
            "catalog_number_candidate_count",
            "catalog_number_candidates",
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
