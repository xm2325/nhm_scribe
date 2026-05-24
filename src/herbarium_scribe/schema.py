from __future__ import annotations

from typing import Any

from .metadata import clean_str

EXTRACTION_FIELDS = [
    "catalogNumber", "scientificName", "recordedBy", "eventDate", "country",
    "stateProvince", "decimalLatitude", "decimalLongitude", "typeStatus",
]


def empty_field() -> dict[str, Any]:
    return {"value": "", "confidence": 0.0, "evidence_span": ""}


def make_field(value: Any, confidence: float = 0.0, evidence_span: str | None = None) -> dict[str, Any]:
    v = clean_str(value)
    try:
        c = float(confidence)
    except Exception:
        c = 0.0
    c = max(0.0, min(1.0, c))
    return {"value": v, "confidence": c, "evidence_span": clean_str(evidence_span if evidence_span is not None else v)}


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    out: dict[str, Any] = {}
    for field in EXTRACTION_FIELDS:
        item = record.get(field, empty_field()) if isinstance(record, dict) else empty_field()
        if not isinstance(item, dict):
            item = make_field(item, 0.5)
            warnings.append(f"coerced_field:{field}")
        value = clean_str(item.get("value", ""))
        evidence = clean_str(item.get("evidence_span", ""))
        try:
            conf = float(item.get("confidence", 0.0))
        except Exception:
            conf = 0.0
            warnings.append(f"bad_confidence:{field}")
        if conf < 0 or conf > 1:
            warnings.append(f"confidence_out_of_range:{field}")
            conf = max(0.0, min(1.0, conf))
        out[field] = {"value": value, "confidence": conf, "evidence_span": evidence}
    for field, low, high in [("decimalLatitude", -90, 90), ("decimalLongitude", -180, 180)]:
        value = out[field]["value"]
        if value:
            try:
                x = float(value)
                if x < low or x > high:
                    warnings.append(f"{field}_out_of_range")
            except Exception:
                warnings.append(f"{field}_not_numeric")
    out["warnings"] = warnings
    return out


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    row = {}
    for field in EXTRACTION_FIELDS:
        row[field] = record.get(field, {}).get("value", "") if isinstance(record.get(field), dict) else ""
        row[f"{field}_confidence"] = record.get(field, {}).get("confidence", 0.0) if isinstance(record.get(field), dict) else 0.0
    row["validation_warnings"] = ";".join(record.get("warnings", []))
    return row
