from herbarium_scribe.llm_crop_selector import (
    bbox_iou,
    containment_ratio,
    duplicate_reason,
    evidence_gate,
    hierarchical_deduplicate,
    select_diverse_crops,
    selected_duplicate_pair_count,
)


def candidate(
    region_id: str,
    component_type: str,
    bbox: list[int],
    *,
    confidence: float = 0.8,
    text: str = "visible text",
    utility: float = 50.0,
) -> dict:
    return {
        "region_id": region_id,
        "component_type": component_type,
        "bbox_xyxy": bbox,
        "detector_confidence": confidence,
        "raw_text": text,
        "decoded_barcode": "",
        "utility_score": utility,
    }


def test_containment_catches_nested_boxes_with_low_iou():
    outer = [0, 0, 100, 100]
    inner = [10, 10, 30, 30]
    assert bbox_iou(outer, inner) == 0.04
    assert containment_ratio(outer, inner) == 1.0


def test_same_role_nested_detection_is_removed():
    outer = candidate("outer", "annotation_label", [0, 0, 100, 100], utility=90)
    inner = candidate("inner", "handwritten_data", [10, 10, 30, 30], utility=40)
    kept, rejected = hierarchical_deduplicate([inner, outer])
    assert [item["region_id"] for item in kept] == ["outer"]
    assert rejected[0]["selection_reason"] == "same_role_geometry_duplicate"
    assert rejected[0]["duplicate_of_region_id"] == "outer"


def test_identifier_parent_and_child_are_preserved():
    parent = candidate("database", "database_label", [0, 0, 200, 100], utility=80)
    child = candidate("number", "number", [20, 20, 80, 50], text="L.2714152", utility=70)
    assert duplicate_reason(child, parent) == ""
    kept, rejected = hierarchical_deduplicate([parent, child])
    assert {item["region_id"] for item in kept} == {"database", "number"}
    assert rejected == []


def test_pairwise_mask_overlap_removes_same_role_detection():
    left = candidate("left", "annotation_label", [0, 0, 100, 100], utility=90)
    right = candidate("right", "handwritten_data", [100, 0, 200, 100], utility=80)
    right["mask_iou_by_region_id"] = {"left": 0.8}
    assert duplicate_reason(right, left) == "same_role_geometry_duplicate"


def test_empty_low_confidence_identifier_is_rejected():
    row = candidate("number", "number", [0, 0, 50, 20], confidence=0.09, text="")
    accepted, reason = evidence_gate(row)
    assert accepted is False
    assert reason == "empty_evidence"


def test_high_confidence_handwriting_can_be_vision_only():
    row = candidate(
        "handwriting",
        "handwritten_data",
        [0, 0, 100, 30],
        confidence=0.45,
        text="",
    )
    assert evidence_gate(row) == (True, "vision_only_high_confidence")


def test_crop_selector_preserves_role_diversity_and_limit():
    rows = [
        candidate("primary", "primary_specimen_label", [0, 0, 200, 100], utility=100),
        candidate("database", "database_label", [0, 100, 100, 150], utility=95),
        candidate("number", "number", [20, 110, 80, 140], utility=90),
        candidate("annotation", "annotation_label", [0, 160, 100, 220], utility=85),
        candidate("number2", "number", [120, 110, 180, 140], utility=80),
    ]
    selected, rejected = select_diverse_crops(rows, total_limit=4)
    assert len(selected) == 4
    assert {item["region_id"] for item in selected} == {
        "primary",
        "database",
        "number",
        "annotation",
    }
    assert rejected[0]["region_id"] == "number2"


def test_crop_selector_honours_zero_total_limit():
    rows = [candidate("primary", "primary_specimen_label", [0, 0, 200, 100])]
    selected, rejected = select_diverse_crops(rows, total_limit=0)
    assert selected == []
    assert [item["region_id"] for item in rejected] == ["primary"]


def test_duplicate_pair_count_excludes_identifier_parent_child():
    rows = [
        candidate("database", "database_label", [0, 0, 200, 100]),
        candidate("number", "number", [10, 10, 70, 40], text="L.2714152"),
        candidate("number_dup", "number", [11, 11, 71, 41], text="L.2714152"),
    ]
    assert selected_duplicate_pair_count(rows) == 1
