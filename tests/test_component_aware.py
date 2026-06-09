import json
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np
import pytest
from PIL import Image

from herbarium_scribe.component_aware import (
    build_evidence_packets,
    direct_evidence_packet,
    identifier_shape_score,
    read_sheet_components,
    reconcile_with_optional_llm,
    resolve_catalog_number,
    validate_reconciled_record,
)
from herbarium_scribe.hespi_layout import (
    _normalise_label,
    non_max_suppression,
    normalise_bbox,
    tile_bbox_to_full,
)
from herbarium_scribe.rag import assert_no_rag_leakage
from herbarium_scribe.review_bundle import read_bundle_csv, read_bundle_jsonl
from scripts.run_component_aware_eval10 import (
    build_review_bundle,
    normalized_mean_embedding,
)


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


def test_hespi_database_label_alias_is_canonical():
    assert _normalise_label("full database label") == "database_label"


def test_non_text_component_skips_ocr():
    readings = read_sheet_components(
        pd.DataFrame([{
            "occurrenceID": "eval:1",
            "catalogNumber": "A1",
            "region_id": "eval:1::swatch",
            "component_type": "swatch",
            "crop_path": "/does/not/exist.jpg",
        }]),
        {"ocr": {}},
    )
    assert readings.loc[0, "engine"] == "not_applicable"
    assert readings.loc[0, "ocr_status"] == "non_text_component_skipped"


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


def test_identifier_shape_prefers_institution_pattern():
    assert identifier_shape_score("L.2714152", ("L",)) > identifier_shape_score(
        "a0 seen - SS SS oe . a v mu. 1S",
        ("L",),
    )


def test_catalog_number_uses_ensemble_score_and_institution_prefix():
    packet = {
        "occurrenceID": "http://data.biodiversitydata.nl/naturalis/specimen/example",
        "components": [
            {
                "region_id": "example::number_bad",
                "component_type": "number",
                "detector_confidence": 0.9,
                "readings": [{
                    "engine": "tesseract_catalog_number_ensemble",
                    "raw_text": "a0 seen - SS SS oe . a v mu. 1S",
                    "candidates": [{
                        "value": "a0 seen - SS SS oe . a v mu. 1S",
                        "score": 14,
                    }],
                }],
            },
            {
                "region_id": "example::number_good",
                "component_type": "number",
                "detector_confidence": 0.86,
                "readings": [{
                    "engine": "tesseract_catalog_number_ensemble",
                    "raw_text": "L.2714152",
                    "candidates": [{"value": "L.2714152", "score": 34}],
                }],
            },
        ],
    }
    resolved = resolve_catalog_number(packet)
    assert resolved["value"] == "L.2714152"
    assert resolved["evidence_source"] == "example::number_good"


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


def test_unreliable_identifier_component_adds_whole_sheet_fallback():
    packet = {
        "occurrenceID": "https://plutof.ut.ee/#/specimen/view/example",
        "components": [
            {
                "region_id": "whole",
                "component_type": "whole_sheet",
                "readings": [{"engine": "tesseract", "raw_text": "TU351001"}],
            },
            {
                "region_id": "number",
                "component_type": "number",
                "readings": [{"engine": "tesseract", "raw_text": "2 SRA Aer ane cece"}],
            },
        ],
    }
    direct = direct_evidence_packet(packet)
    assert [item["region_id"] for item in direct["components"]] == [
        "number",
        "whole",
    ]
    assert direct["whole_sheet_fallback_used"] is True


def test_empty_component_evidence_skips_llm_call():
    record, meta = reconcile_with_optional_llm(
        {
            "occurrenceID": "eval:1",
            "components": [{
                "region_id": "eval:1::number",
                "component_type": "number",
                "readings": [{"engine": "tesseract", "raw_text": ""}],
            }],
        },
        {"llm": {"backend": "qwen_api"}},
    )
    assert meta["llm_status"] == "not_evaluated"
    assert meta["not_evaluated_reason"] == "empty_component_evidence"
    assert record["catalogNumber"]["value"] == ""


def test_visual_embedding_average_is_normalized():
    result = normalized_mean_embedding([
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
    ])
    assert result is not None
    assert np.linalg.norm(result) == pytest.approx(1.0)
    assert result.tolist() == pytest.approx([2 ** -0.5, 2 ** -0.5])


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


def test_review_bundle_preserves_same_named_overviews(tmp_path: Path):
    processed = tmp_path / "processed"
    report_dir = tmp_path / "reports"
    processed.mkdir()
    report_dir.mkdir()
    report_path = report_dir / "component_aware_eval10_report.md"
    report_path.write_text("# Report\n", encoding="utf-8")
    (processed / "branch_comparison.csv").write_text(
        "branch,coverage\nbaseline_full_sheet,0\n",
        encoding="utf-8",
    )

    component_rows = []
    for index, catalog in enumerate(["A1", "B2"]):
        source_dir = tmp_path / f"record_{index}"
        source_dir.mkdir()
        annotation = source_dir / "sheet_components_annotated.jpg"
        crop = source_dir / "number.jpg"
        Image.new("RGB", (20, 20), color=(index * 50, 20, 20)).save(annotation)
        Image.new("RGB", (10, 10), color=(20, index * 50, 20)).save(crop)
        component_rows.append({
            "occurrenceID": f"eval:{index}",
            "catalogNumber": catalog,
            "region_id": f"eval:{index}::number",
            "component_type": "number",
            "annotation_path": str(annotation),
            "crop_path": str(crop),
        })

    archive = build_review_bundle(
        report_dir,
        processed,
        pd.DataFrame(component_rows),
        report_path,
    )
    frame = read_bundle_csv(archive, "sheet_components.csv")
    assert frame["review_overview_member"].nunique() == 2
    with zipfile.ZipFile(archive) as bundle:
        overview_names = [
            name for name in bundle.namelist()
            if "/assets/overviews/" in name
        ]
    assert len(overview_names) == 2
