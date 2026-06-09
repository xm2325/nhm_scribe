from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .extract_rules import extract_rule_based
from .llm_backends import call_llm_with_metadata
from .metadata import clean_str
from .ocr import (
    decode_barcodes,
    ocr_catalog_number_ensemble,
    ocr_image_hespi_trocr,
    ocr_image_tesseract,
)
from .pipeline import _parse_llm_json
from .qwen_vision import image_data_url
from .schema import EXTRACTION_FIELDS

IDENTIFIER_SOURCE_PRIORITY = {
    "barcode_decoder": 0,
    "database_label": 1,
    "number": 2,
    "whole_sheet": 3,
    "primary_specimen_label": 4,
}
TYPE_STATUS_TERMS = (
    "holotype", "isotype", "lectotype", "isolectotype", "neotype",
    "isoneotype", "syntype", "isosyntype", "paratype", "isoparatype",
)


def _json_value(value: Any, fallback: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(clean_str(value))
    except (TypeError, json.JSONDecodeError):
        return fallback


def _identifier_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", clean_str(value)).upper()


def plausible_identifier(value: str) -> bool:
    key = _identifier_key(value)
    return (
        5 <= len(key) <= 24
        and any(char.isalpha() for char in key)
        and any(char.isdigit() for char in key)
    )


def _reading(
    *,
    component: pd.Series,
    engine: str,
    raw_text: str,
    ocr_confidence: float | str | None = None,
    ocr_status: str = "",
    decoder_status: str = "",
    model_reported_confidence: float | str | None = None,
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "occurrenceID": clean_str(component.get("occurrenceID")),
        "catalogNumber": clean_str(component.get("catalogNumber")),
        "region_id": clean_str(component.get("region_id", component.get("component_id"))),
        "component_type": clean_str(component.get("component_type")),
        "engine": engine,
        "raw_text": clean_str(raw_text),
        "ocr_confidence": "" if ocr_confidence is None else ocr_confidence,
        "ocr_status": ocr_status,
        "decoder_status": decoder_status,
        "model_reported_confidence": (
            "" if model_reported_confidence is None else model_reported_confidence
        ),
        "candidates_json": json.dumps(candidates or [], ensure_ascii=False),
        "crop_path": clean_str(component.get("crop_path")),
    }


def read_sheet_components(
    components: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    ocfg = cfg.get("ocr", {})
    lang = clean_str(ocfg.get("tesseract_lang", "eng")) or "eng"
    ensemble_cfg = ocfg.get("catalog_number_ensemble", {})
    rows: list[dict[str, Any]] = []
    for _, component in components.iterrows():
        component_type = clean_str(component.get("component_type")).lower()
        crop_path = clean_str(component.get("crop_path"))
        if component_type == "barcode":
            decoded, decoder_status, _ = decode_barcodes(crop_path)
            decoded_values = [item["value"] for item in decoded]
            rows.append(_reading(
                component=component,
                engine="zxingcpp",
                raw_text="\n".join(decoded_values),
                ocr_status="not_applicable",
                decoder_status=decoder_status,
                candidates=decoded,
            ))
            if decoded_values:
                continue

        tesseract_config = "--psm 6"
        if component_type in {"barcode", "database_label", "number"}:
            tesseract_config = "--psm 7"
        elif component_type == "stamp":
            tesseract_config = "--psm 11"
        text, confidence, status = ocr_image_tesseract(
            crop_path,
            lang=lang,
            config=tesseract_config,
        )
        if component_type in {"barcode", "database_label", "number"}:
            ensemble_text, candidates, ensemble_status, _, _ = ocr_catalog_number_ensemble(
                crop_path,
                standard_text=text,
                lang=lang,
                config=ensemble_cfg,
            )
            rows.append(_reading(
                component=component,
                engine="tesseract_catalog_number_ensemble",
                raw_text=ensemble_text,
                ocr_confidence=confidence,
                ocr_status=f"{status};ensemble_{ensemble_status}",
                decoder_status="fallback_after_no_barcode" if component_type == "barcode" else "not_applicable",
                candidates=candidates,
            ))
        else:
            candidates = []
            if component_type == "type_label":
                candidates = [
                    {"value": term, "classification": "controlled_vocabulary"}
                    for term in TYPE_STATUS_TERMS
                    if re.search(rf"\b{term}\b", text, flags=re.IGNORECASE)
                ]
            rows.append(_reading(
                component=component,
                engine="tesseract",
                raw_text=text,
                ocr_confidence=confidence,
                ocr_status=status,
                decoder_status="not_applicable",
                candidates=candidates,
            ))
        htr_cfg = cfg.get("component_aware", {}).get("handwriting_recognition", {})
        if (
            bool(htr_cfg.get("enabled", False))
            and component_type in {
                clean_str(value).lower()
                for value in htr_cfg.get(
                    "component_types",
                    ["primary_specimen_label", "annotation_label"],
                )
            }
        ):
            model_size = clean_str(htr_cfg.get("model_size", "base")) or "base"
            htr_text, htr_status, _ = ocr_image_hespi_trocr(
                crop_path,
                model_size=model_size,
            )
            rows.append(_reading(
                component=component,
                engine=f"hespi_trocr_{model_size}",
                raw_text=htr_text,
                ocr_status=htr_status,
                decoder_status="not_applicable",
            ))
    vision_cfg = cfg.get("component_aware", {}).get("multimodal_transcription", {})
    if bool(vision_cfg.get("enabled", False)):
        supported_types = {
            clean_str(value).lower()
            for value in vision_cfg.get(
                "component_types",
                ["primary_specimen_label", "annotation_label"],
            )
        }
        for _, component in components.iterrows():
            if clean_str(component.get("component_type")).lower() not in supported_types:
                continue
            crop_path = clean_str(component.get("crop_path"))
            try:
                image_url, _ = image_data_url(
                    crop_path,
                    max_dimension=int(vision_cfg.get("max_dimension", 2200)),
                )
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Transcribe every visible character in this herbarium component. "
                            "Preserve line breaks and uncertain text. Do not infer missing text. "
                            "Return plain transcription only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Component type: {component.get('component_type')}"},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ]
                meta = call_llm_with_metadata(messages, cfg)
                text = clean_str(meta.get("content"))
                rows.append(_reading(
                    component=component,
                    engine=f"{clean_str(meta.get('backend'))}_multimodal_transcription",
                    raw_text=text,
                    ocr_status="ok" if text else "empty_output",
                    decoder_status="not_applicable",
                    model_reported_confidence=None,
                ))
            except Exception as exc:
                rows.append(_reading(
                    component=component,
                    engine="multimodal_transcription",
                    raw_text="",
                    ocr_status=f"error:{type(exc).__name__}",
                    decoder_status="not_applicable",
                ))
    return pd.DataFrame(rows)


def build_evidence_packets(
    components: pd.DataFrame,
    readings: pd.DataFrame,
) -> list[dict[str, Any]]:
    packets = []
    for occurrence_id, specimen_components in components.groupby("occurrenceID", sort=False):
        packet_components = []
        for _, component in specimen_components.iterrows():
            region_id = clean_str(component.get("region_id", component.get("component_id")))
            component_readings = readings[readings["region_id"].astype(str) == region_id]
            packet_components.append({
                "region_id": region_id,
                "component_type": clean_str(component.get("component_type")),
                "bbox_xyxy": _json_value(component.get("bbox_xyxy", component.get("bbox")), []),
                "detector_confidence": float(component.get("detector_confidence", component.get("confidence", 0)) or 0),
                "readings": [
                    {
                        "engine": clean_str(item.get("engine")),
                        "raw_text": clean_str(item.get("raw_text")),
                        "ocr_confidence": (
                            None if clean_str(item.get("ocr_confidence")) == ""
                            else float(item.get("ocr_confidence"))
                        ),
                        "ocr_status": clean_str(item.get("ocr_status")),
                        "decoder_status": clean_str(item.get("decoder_status")),
                        "model_reported_confidence": (
                            None if clean_str(item.get("model_reported_confidence")) == ""
                            else float(item.get("model_reported_confidence"))
                        ),
                    }
                    for _, item in component_readings.iterrows()
                ],
            })
        packets.append({
            "occurrenceID": clean_str(occurrence_id),
            "catalogNumber_gold": "",
            "gold_withheld_until_evaluation": True,
            "components": packet_components,
        })
    return packets


def evidence_packet_text(packet: dict[str, Any]) -> str:
    blocks = []
    for component in packet.get("components", []):
        for reading in component.get("readings", []):
            text = clean_str(reading.get("raw_text"))
            if text:
                blocks.append(
                    f"[region_id={component.get('region_id')}; "
                    f"component_type={component.get('component_type')}; "
                    f"engine={reading.get('engine')}]\n{text}"
                )
    return "\n\n".join(blocks)


def direct_evidence_packet(packet: dict[str, Any]) -> dict[str, Any]:
    non_whole = [
        component
        for component in packet.get("components", [])
        if clean_str(component.get("component_type")).lower() != "whole_sheet"
        and any(clean_str(reading.get("raw_text")) for reading in component.get("readings", []))
    ]
    if non_whole:
        return {**packet, "components": non_whole, "whole_sheet_fallback_used": False}
    whole = [
        component
        for component in packet.get("components", [])
        if clean_str(component.get("component_type")).lower() == "whole_sheet"
    ]
    return {**packet, "components": whole, "whole_sheet_fallback_used": True}


def identifier_candidates(packet: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for component in packet.get("components", []):
        component_type = clean_str(component.get("component_type")).lower()
        for reading in component.get("readings", []):
            engine = clean_str(reading.get("engine"))
            source_type = "barcode_decoder" if engine == "zxingcpp" else component_type
            for line in clean_str(reading.get("raw_text")).splitlines():
                value = line.strip(" .,:;|")
                if not plausible_identifier(value):
                    continue
                key = _identifier_key(value)
                source = f"{component.get('region_id')}:{engine}"
                candidate = candidates.setdefault(key, {
                    "value": value,
                    "normalised": key,
                    "priority": IDENTIFIER_SOURCE_PRIORITY.get(source_type, 99),
                    "evidence_source": clean_str(component.get("region_id")),
                    "supporting_sources": [],
                })
                if source not in candidate["supporting_sources"]:
                    candidate["supporting_sources"].append(source)
                source_priority = IDENTIFIER_SOURCE_PRIORITY.get(source_type, 99)
                if source_priority < int(candidate["priority"]):
                    candidate["value"] = value
                    candidate["priority"] = source_priority
                    candidate["evidence_source"] = clean_str(component.get("region_id"))
    return sorted(
        candidates.values(),
        key=lambda item: (item["priority"], -len(item["supporting_sources"]), item["normalised"]),
    )


def resolve_catalog_number(packet: dict[str, Any]) -> dict[str, Any]:
    candidates = identifier_candidates(packet)
    if not candidates:
        return empty_reconciled_field()
    best = candidates[0]
    same_priority = [
        item for item in candidates
        if item["priority"] == best["priority"] and item["normalised"] != best["normalised"]
    ]
    alternatives = [
        item["value"] for item in candidates[1:]
        if item["normalised"] != best["normalised"]
    ]
    return {
        "value": best["value"],
        "model_reported_confidence": 1.0 if best["priority"] == 0 and not same_priority else 0.75,
        "evidence_span": best["value"],
        "evidence_source": best["evidence_source"],
        "supporting_sources": best["supporting_sources"],
        "alternative_candidates": alternatives,
        "review_required": bool(same_priority or alternatives),
    }


def empty_reconciled_field() -> dict[str, Any]:
    return {
        "value": "",
        "model_reported_confidence": 0.0,
        "evidence_span": "",
        "evidence_source": "",
        "supporting_sources": [],
        "alternative_candidates": [],
        "review_required": False,
    }


def validate_reconciled_record(
    record: dict[str, Any],
    packet: dict[str, Any],
) -> dict[str, Any]:
    valid_sources = {
        clean_str(component.get("region_id"))
        for component in packet.get("components", [])
    }
    evidence_text_by_source = {
        clean_str(component.get("region_id")): "\n".join(
            clean_str(reading.get("raw_text"))
            for reading in component.get("readings", [])
            if clean_str(reading.get("raw_text"))
        )
        for component in packet.get("components", [])
    }
    output = {}
    for field in EXTRACTION_FIELDS:
        item = record.get(field, {}) if isinstance(record, dict) else {}
        if not isinstance(item, dict):
            item = {"value": item}
        value = clean_str(item.get("value"))
        span = clean_str(item.get("evidence_span"))
        source = clean_str(item.get("evidence_source"))
        supporting = item.get("supporting_sources", [])
        alternatives = item.get("alternative_candidates", [])
        if isinstance(supporting, str):
            supporting = [supporting]
        if isinstance(alternatives, str):
            alternatives = [alternatives]
        try:
            confidence = max(0.0, min(1.0, float(item.get("model_reported_confidence", 0) or 0)))
        except (TypeError, ValueError):
            confidence = 0.0
        unsupported = bool(value) and (
            source not in valid_sources
            or not span
            or span.lower() not in evidence_text_by_source.get(source, "").lower()
        )
        if unsupported:
            value = ""
            confidence = 0.0
            span = ""
            source = ""
            supporting = []
            alternatives = []
        output[field] = {
            "value": value,
            "model_reported_confidence": confidence,
            "evidence_span": span,
            "evidence_source": source,
            "supporting_sources": [clean_str(value) for value in supporting if clean_str(value)],
            "alternative_candidates": [clean_str(value) for value in alternatives if clean_str(value)],
            "review_required": bool(item.get("review_required", False)) or unsupported or bool(alternatives),
        }
    resolved_catalog = resolve_catalog_number(packet)
    if resolved_catalog["value"]:
        llm_catalog = output["catalogNumber"]
        if (
            llm_catalog["value"]
            and _identifier_key(llm_catalog["value"]) != _identifier_key(resolved_catalog["value"])
            and llm_catalog["value"] not in resolved_catalog["alternative_candidates"]
        ):
            resolved_catalog["alternative_candidates"].append(llm_catalog["value"])
            resolved_catalog["review_required"] = True
        output["catalogNumber"] = resolved_catalog
    return output


def deterministic_reconciliation(packet: dict[str, Any]) -> dict[str, Any]:
    text = evidence_packet_text(packet)
    rule = extract_rule_based(text)
    record = {field: empty_reconciled_field() for field in EXTRACTION_FIELDS}
    for field in EXTRACTION_FIELDS:
        value = clean_str(rule.get(field, {}).get("value"))
        evidence = clean_str(rule.get(field, {}).get("evidence_span"))
        if not value:
            continue
        source = ""
        supporting = []
        for component in packet.get("components", []):
            component_text = "\n".join(
                clean_str(reading.get("raw_text"))
                for reading in component.get("readings", [])
            )
            if evidence and evidence.lower() in component_text.lower():
                source = clean_str(component.get("region_id"))
                supporting = [
                    f"{source}:{reading.get('engine')}"
                    for reading in component.get("readings", [])
                    if evidence.lower() in clean_str(reading.get("raw_text")).lower()
                ]
                break
        record[field] = {
            "value": value,
            "model_reported_confidence": float(rule.get(field, {}).get("confidence", 0) or 0),
            "evidence_span": evidence,
            "evidence_source": source,
            "supporting_sources": supporting,
            "alternative_candidates": [],
            "review_required": False,
        }
    return validate_reconciled_record(record, packet)


def component_reconciliation_prompt(
    packet: dict[str, Any],
    retrieval_context: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    references = retrieval_context or []
    reference_text = "\n".join(
        f"[REFERENCE rank={item.get('rank')} occurrenceID={item.get('reference_occurrenceID')} "
        f"similarity={item.get('combined_similarity', 0):.4f}] "
        f"scientificName={item.get('gold_scientificName', '')}; "
        f"recordedBy={item.get('gold_recordedBy', '')}; eventDate={item.get('gold_eventDate', '')}"
        for item in references
    )
    schema = {
        field: {
            "value": "",
            "model_reported_confidence": 0.0,
            "evidence_span": "",
            "evidence_source": "region_id",
            "supporting_sources": ["region_id:engine"],
            "alternative_candidates": [],
            "review_required": False,
        }
        for field in EXTRACTION_FIELDS
    }
    return [
        {
            "role": "system",
            "content": (
                "Reconcile component-level herbarium evidence into JSON. Every non-empty value must be "
                "directly supported by an exact evidence_span and a valid region_id. Retrieval references "
                "are contextual examples only: never copy their catalogNumber or overwrite visible evidence. "
                "For catalogNumber use decoded barcode, database label, number label, whole sheet, then "
                "primary label in that order. Preserve conflicting plausible identifiers as alternatives and "
                "set review_required true. Do not guess. Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"EVIDENCE PACKET\n{evidence_packet_text(packet)}\n\n"
                f"RETRIEVAL REFERENCES\n{reference_text or 'none'}\n\n"
                f"OUTPUT SCHEMA\n{json.dumps(schema, ensure_ascii=False)}"
            ),
        },
    ]


def reconcile_with_optional_llm(
    packet: dict[str, Any],
    cfg: dict[str, Any],
    retrieval_context: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deterministic = deterministic_reconciliation(packet)
    if not evidence_packet_text(packet).strip():
        return deterministic, {
            "llm_status": "not_evaluated",
            "not_evaluated_reason": "empty_component_evidence",
            "raw_output": "",
        }
    if clean_str(cfg.get("llm", {}).get("backend", "none")).lower() == "none":
        return deterministic, {
            "llm_status": "not_evaluated",
            "not_evaluated_reason": "api_credentials_or_backend_unavailable",
            "raw_output": "",
        }
    messages = component_reconciliation_prompt(packet, retrieval_context)
    meta = call_llm_with_metadata(messages, cfg)
    raw = clean_str(meta.get("content"))
    parsed = _parse_llm_json(raw)
    if not parsed:
        return deterministic, {
            **meta,
            "llm_status": "not_evaluated" if not raw else "parse_failure",
            "not_evaluated_reason": clean_str(meta.get("error_message")) or ("empty_raw_output" if not raw else ""),
            "raw_output": raw,
        }
    return validate_reconciled_record(parsed, packet), {
        **meta,
        "llm_status": "parsed",
        "not_evaluated_reason": "",
        "raw_output": raw,
    }


def flatten_reconciled_record(
    occurrence_id: str,
    branch: str,
    record: dict[str, Any],
    *,
    status: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "occurrenceID": occurrence_id,
        "branch": branch,
        "status": status,
    }
    for field in EXTRACTION_FIELDS:
        item = record.get(field, empty_reconciled_field())
        row[field] = clean_str(item.get("value"))
        row[f"{field}_evidence_source"] = clean_str(item.get("evidence_source"))
        row[f"{field}_evidence_span"] = clean_str(item.get("evidence_span"))
        row[f"{field}_supporting_sources"] = json.dumps(item.get("supporting_sources", []), ensure_ascii=False)
        row[f"{field}_alternative_candidates"] = json.dumps(item.get("alternative_candidates", []), ensure_ascii=False)
        row[f"{field}_model_reported_confidence"] = item.get("model_reported_confidence", 0)
        row[f"{field}_review_required"] = bool(item.get("review_required", False))
    return row
