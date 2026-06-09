import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from herbarium_scribe.component_aware import (
    build_evidence_packets,
    direct_evidence_packet,
    resolve_catalog_number,
    validate_reconciled_record,
)
from herbarium_scribe.hespi_layout import (
    non_max_suppression,
    normalise_bbox,
    tile_bbox_to_full,
)
from herbarium_scribe.rag import assert_no_rag_leakage
from herbarium_scribe.review_bundle import read_bundle_csv, read_bundle_jsonl


def packet_with_identifiers() -> dict:
    return {
        "occurrenceID": "eval:1",
        "components": [
            {
                "region_id": "eval:1::number",
                "component_type": "number",
                "readings": [{"engine": "tesseract", "raw_text": "L.2714152"}],
            },
            {
                "region_id": "eval:1::barcode",
                "component_type": "barcode",
                "readings": [{"engine": "zxingcpp", "raw_text": "L2714152"}],
            },
        ],
    }


def test_bounding_boxes_are_valid_and_normalized():
    normalized = normalise_bbox([-10, 20, 120, 140], width=100, height=100)
    assert normalized == [0.0, 0.2, 1.0, 1.0]
    assert all(0 <= value <= 1 for value in normalized)


def test_tile_local_bbox_maps_to_full_sheet():
    assert tile_bbox_to_full([5, 10, 50, 70], [100, 200, 300, 400]) == [105, 210, 150, 270]


def test_nms_merges_overlapping_same_class_only():
    rows = [
        {"label": "number", "confidence": 0.9, "bbox": [0, 0, 100, 100]},
        {"label": "number", "confidence": 0.8, "bbox": [5, 5, 95, 95]},
        {"label": "barcode", "confidence": 0.7, "bbox": [5, 5, 95, 95]},
    ]
    kept = non_max_suppression(rows, iou_threshold=0.5)
    assert [(row["label"], row["confidence"]) for row in kept] == [
        ("number", 0.9),
        ("barcode", 0.7),
    ]


def test_rag_leakage_prevention_fails_on_overlap():
    with pytest.raises(ValueError, match="contains EVAL"):
        assert_no_rag_leakage({"eval:1"}, {"ref:1", "eval:1"})


def test_catalog_number_prefers_decoded_barcode():
    resolved = resolve_catalog_number(packet_with_identifiers())
    assert resolved["value"] == "L2714152"
    assert resolved["evidence_source"] == "eval:1::barcode"


def test_alternative_identifiers_are_preserved_and_reviewed():
    packet = packet_with_identifiers()
    packet["components"][0]["readings"][0]["raw_text"] = "L.2714153"
    resolved = resolve_catalog_number(packet)
    assert resolved["value"] == "L2714152"
    assert "L.2714153" in resolved["alternative_candidates"]
    assert resolved["review_required"] is True


def test_unsupported_llm_field_is_removed_and_reviewed():
    packet = packet_with_identifiers()
    record = {
        "scientificName": {
            "value": "Inventus plantus",
            "model_reported_confidence": 0.99,
            "evidence_span": "Inventus plantus",
            "evidence_source": "eval:1::number",
            "supporting_sources": [],
            "alternative_candidates": [],
            "review_required": False,
        }
    }
    validated = validate_reconciled_record(record, packet)
    assert validated["scientificName"]["value"] == ""
    assert validated["scientificName"]["review_required"] is True


def test_evidence_packets_keep_confidences_separate():
    components = pd.DataFrame([{
        "occurrenceID": "eval:1",
        "region_id": "eval:1::number",
        "component_type": "number",
        "bbox_xyxy": "[1,2,3,4]",
        "detector_confidence": "0.8",
    }])
    readings = pd.DataFrame([{
        "region_id": "eval:1::number",
        "engine": "tesseract",
        "raw_text": "L.2714152",
        "ocr_confidence": "0.6",
        "ocr_status": "ok",
        "decoder_status": "not_applicable",
        "model_reported_confidence": "",
    }])
    packet = build_evidence_packets(components, readings)[0]
    assert packet["components"][0]["detector_confidence"] == 0.8
    assert packet["components"][0]["readings"][0]["ocr_confidence"] == 0.6
    assert packet["components"][0]["readings"][0]["model_reported_confidence"] is None


def test_whole_sheet_is_only_used_when_component_evidence_is_empty():
    packet = {
        "components": [
            {
                "region_id": "whole",
                "component_type": "whole_sheet",
                "readings": [{"engine": "tesseract", "raw_text": "full sheet"}],
            },
            {
                "region_id": "number",
                "component_type": "number",
                "readings": [{"engine": "tesseract", "raw_text": "L.2714152"}],
            },
        ]
    }
    direct = direct_evidence_packet(packet)
    assert [item["region_id"] for item in direct["components"]] == ["number"]
    assert direct["whole_sheet_fallback_used"] is False


def test_review_bundle_csv_and_jsonl_loading(tmp_path: Path):
    archive_path = tmp_path / "bundle.zip"
    csv_data = "occurrenceID,branch\neval:1,component_aware_no_rag\n"
    jsonl_data = json.dumps({"occurrenceID": "eval:1"}) + "\n"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("component_aware_eval10/processed/branch_comparison.csv", csv_data)
        archive.writestr("component_aware_eval10/processed/evidence_packets.jsonl", jsonl_data)
    frame = read_bundle_csv(archive_path, "branch_comparison.csv")
    rows = read_bundle_jsonl(archive_path, "evidence_packets.jsonl")
    assert frame.loc[0, "branch"] == "component_aware_no_rag"
    assert rows[0]["occurrenceID"] == "eval:1"
