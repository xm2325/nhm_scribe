from pathlib import Path

import pandas as pd
from PIL import Image

from scripts.run_yoloe26_llm_crop_eval10 import (
    BRANCHES,
    build_metrics,
    build_review_bundle,
    prepare_artifact_members,
)


def selected_candidate(branch: str, source: Path, crop: Path) -> dict:
    return {
        "branch": branch,
        "occurrenceID": "record:1",
        "catalogNumber": "E00289691",
        "region_id": f"record:1::{branch}",
        "component_type": "primary_specimen_label",
        "detector_family": "hespi" if branch == "hespi" else "yoloe26",
        "detector_confidence": 0.9,
        "bbox_xyxy": [5, 5, 55, 55],
        "crop_bbox_xyxy": [5, 5, 55, 55],
        "source_image_path": str(source),
        "crop_path": str(crop),
        "raw_text": "E00289691 Abelia China 29 May 2006",
        "decoded_barcode": "E00289691",
        "selection_status": "selected",
        "selection_reason": "diverse_role_utility",
        "input_order": 1,
    }


def test_metrics_and_visual_report_are_self_contained(tmp_path: Path):
    source = tmp_path / "source.jpg"
    crop = tmp_path / "crop.jpg"
    Image.new("RGB", (80, 80), "white").save(source)
    Image.new("RGB", (50, 50), "white").save(crop)
    decisions = [selected_candidate(branch, source, crop) for branch in BRANCHES]
    prepare_artifact_members(decisions)
    eval_df = pd.DataFrame([{
        "occurrenceID": "record:1",
        "catalogNumber": "E00289691",
        "scientificName": "Abelia",
        "recordedBy": "",
        "eventDate": "2006-05-29",
        "country": "China",
        "stateProvince": "",
        "decimalLatitude": "",
        "decimalLongitude": "",
        "typeStatus": "",
    }])
    records, branches, fields, field_summary = build_metrics(decisions, eval_df)
    assert len(records) == 4
    assert len(fields) == 36
    assert len(field_summary) == 36
    assert branches["field_evidence_hit_count"].gt(0).all()

    processed = tmp_path / "processed"
    reports = tmp_path / "reports"
    processed.mkdir()
    reports.mkdir()
    archive = build_review_bundle(
        reports,
        processed,
        decisions,
        eval_df,
        pd.DataFrame([{"occurrenceID": "record:1", "image_path": str(source)}]),
        branches,
        fields,
        field_summary,
    )
    visual = (reports / "visual_report.html").read_text(encoding="utf-8")
    assert archive.is_file()
    assert "data:image/jpeg;base64," in visual
    assert 'src="../' not in visual
    assert 'src="assets/' not in visual
