from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

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


def normalise_country(country: str) -> str:
    c = clean_str(country)
    return COUNTRY_NORMALISATION.get(c.lower(), c)


def reconcile_row(row: pd.Series) -> dict[str, Any]:
    verbatim = clean_str(row.get("scientificName", ""))
    canonical = canonical_scientific_name(verbatim)
    status = "not_provided" if not verbatim else ("local_canonical" if canonical else "unmatched")
    warning = "" if canonical or not verbatim else "taxonomy_unmatched"
    country = normalise_country(row.get("country", ""))
    return {
        "scientificName_verbatim": verbatim,
        "scientificName_canonical": canonical,
        "taxonomy_match_status": status,
        "taxonomy_warning": warning,
        "country_verbatim": clean_str(row.get("country", "")),
        "country_normalised": country,
        "stateProvince_normalised": clean_str(row.get("stateProvince", "")),
    }


def reconcile_dataframe(pred_df: pd.DataFrame, paths: dict[str, Path] | None = None) -> pd.DataFrame:
    rows = []
    for _, row in pred_df.iterrows():
        rows.append(reconcile_row(row))
    rec = pd.concat([pred_df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    if paths:
        rec.to_csv(paths["processed"] / "extractions_flat_reconciled.csv", index=False)
        rec[["occurrenceID", "method", "scientificName_verbatim", "scientificName_canonical", "taxonomy_match_status", "taxonomy_warning", "country_normalised", "stateProvince_normalised"]].to_csv(paths["processed"] / "reconciliation.csv", index=False)
    return rec
