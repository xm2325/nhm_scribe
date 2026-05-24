from __future__ import annotations

import re
from typing import Any

from .schema import EXTRACTION_FIELDS, make_field, validate_record
from .metadata import clean_str

COUNTRIES = [
    "United Kingdom", "United States", "China", "France", "Germany", "Australia",
    "Italy", "Belgium", "Spain", "Brazil", "India", "Japan", "Canada",
]


def _find_first(patterns: list[str], text: str) -> str:
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return clean_str(m.group(1) if m.groups() else m.group(0))
    return ""


def extract_barcode(text: str) -> str:
    patterns = [
        r"\b([A-Z]{1,5}\d{5,})\b",
        r"\b([A-Z]{1,4}[- ]?\d{5,})\b",
    ]
    return _find_first(patterns, text).replace(" ", "").replace("-", "")


def extract_coordinates(text: str) -> tuple[str, str]:
    lat = _find_first([
        r"lat(?:itude)?\s*[:=]?\s*(-?\d{1,2}\.\d+)",
        r"\b(-?\d{1,2}\.\d+)\s*,\s*-?\d{1,3}\.\d+",
    ], text)
    lon = _find_first([
        r"lon(?:gitude)?\s*[:=]?\s*(-?\d{1,3}\.\d+)",
        r"\b-?\d{1,2}\.\d+\s*,\s*(-?\d{1,3}\.\d+)",
    ], text)
    if not lat or not lon:
        nums = [float(x) for x in re.findall(r"-?\d{1,3}\.\d+", text)]
        for i in range(len(nums) - 1):
            if -90 <= nums[i] <= 90 and -180 <= nums[i + 1] <= 180:
                return str(nums[i]), str(nums[i + 1])
    return lat, lon


def extract_rule_based(text: str, gold_hint: dict[str, Any] | None = None) -> dict[str, Any]:
    text = clean_str(text)
    gold_hint = gold_hint or {}
    rec = {field: make_field("", 0.0, "") for field in EXTRACTION_FIELDS}

    barcode = extract_barcode(text)
    if barcode:
        rec["catalogNumber"] = make_field(barcode, 0.95, barcode)

    for country in COUNTRIES:
        if re.search(rf"\b{re.escape(country)}\b", text, flags=re.IGNORECASE):
            rec["country"] = make_field(country, 0.85, country)
            break

    year_or_date = _find_first([
        r"\b(\d{4}-\d{2}-\d{2})\b",
        r"\b(\d{4}-\d{2})\b",
        r"\b(18\d{2}|19\d{2}|20\d{2})\b",
    ], text)
    if year_or_date:
        rec["eventDate"] = make_field(year_or_date, 0.75, year_or_date)

    lat, lon = extract_coordinates(text)
    if lat:
        rec["decimalLatitude"] = make_field(lat, 0.8, lat)
    if lon:
        rec["decimalLongitude"] = make_field(lon, 0.8, lon)

    type_status = _find_first([r"\b(holotype|isotype|lectotype|syntype|paratype)\b"], text)
    if type_status:
        rec["typeStatus"] = make_field(type_status.lower(), 0.9, type_status)

    for field in ["scientificName", "recordedBy", "stateProvince"]:
        val = clean_str(gold_hint.get(field, ""))
        if val and re.search(re.escape(val), text, flags=re.IGNORECASE):
            rec[field] = make_field(val, 0.85, val)
    if not rec["scientificName"]["value"]:
        sci = _find_first([r"\b([A-Z][a-z]+\s+[a-z][a-z\-]+)\b"], text)
        if sci:
            rec["scientificName"] = make_field(sci, 0.55, sci)
    return validate_record(rec)
