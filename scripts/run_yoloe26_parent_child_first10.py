from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_yoloe26_llm_crop_eval10 as base  # noqa: E402

BRANCHES = (
    "hespi",
    "yoloe_parent_child",
    "hespi_yoloe_parent_child_union",
)
base.BRANCHES = BRANCHES

PARENT_TYPES = {
    "primary_specimen_label",
    "printed_label",
    "annotation_label",
    "handwritten_data",
    "database_label",
    "type_label",
}
HIGH_VALUE_CHILD_TYPES = {
    "barcode",
    "barcode_label",
    "catalog_number",
    "number",
}


def bbox_area(bbox: Any) -> float:
    x0, y0, x1, y1 = base.parse_bbox(bbox)
    return float(max(0, x1 - x0) * max(0, y1 - y0))


def intersection_area(a: Any, b: Any) -> float:
    ax0, ay0, ax1, ay1 = base.parse_bbox(a)
    bx0, by0, bx1, by1 = base.parse_bbox(b)
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    return float(max(0, x1 - x0) * max(0, y1 - y0))


def is_high_value_child(row: dict[str, Any]) -> bool:
    component_type = base.clean_str(row.get("component_type"))
    if component_type in HIGH_VALUE_CHILD_TYPES:
        return True
    # Small database labels are often catalogue/barcode stickers. Keep them
    # as independent OCR evidence even if they sit inside a larger label.
    if component_type == "database_label" and float(row.get("area_fraction") or 0.0) <= 0.030:
        return True
    return False


def parent_child_prune(
    rows: list[dict[str, Any]],
    *,
    containment_threshold: float = 0.85,
    child_parent_area_ratio: float = 0.35,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep parent boxes, drop non-key child boxes, preserve high-value children."""
    retained: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    prepared: list[dict[str, Any]] = []
    for row in rows:
        bbox = base.parse_bbox(row.get("bbox_xyxy"))
        prepared.append({**row, "_bbox": bbox, "_area": bbox_area(bbox)})

    for child in prepared:
        best_parent: dict[str, Any] | None = None
        best_containment = 0.0
        best_ratio = 1.0
        child_area = max(float(child["_area"]), 1.0)
        for parent in prepared:
            if parent is child:
                continue
            parent_area = max(float(parent["_area"]), 1.0)
            if parent_area <= child_area:
                continue
            if base.clean_str(parent.get("component_type")) not in PARENT_TYPES:
                continue
            containment = intersection_area(child["_bbox"], parent["_bbox"]) / child_area
            area_ratio = child_area / parent_area
            if containment > best_containment:
                best_parent = parent
                best_containment = containment
                best_ratio = area_ratio

        annotated = {key: value for key, value in child.items() if not key.startswith("_")}
        annotated["parent_child_containment"] = round(best_containment, 6)
        annotated["parent_child_area_ratio"] = round(best_ratio, 6)
        annotated["parent_region_id"] = base.clean_str(best_parent.get("region_id")) if best_parent else ""
        should_fallback = (
            best_parent is not None
            and best_containment >= containment_threshold
            and best_ratio <= child_parent_area_ratio
            and not is_high_value_child(annotated)
        )
        if should_fallback:
            annotated["parent_child_status"] = "fallback_child_inside_parent"
            annotated["parent_child_reason"] = (
                f"contained={best_containment:.3f}; area_ratio={best_ratio:.3f}; "
                f"parent={annotated['parent_region_id']}"
            )
            fallback.append(annotated)
        else:
            if best_parent is not None and best_containment >= containment_threshold and best_ratio <= child_parent_area_ratio:
                annotated["parent_child_status"] = "retained_high_value_child"
                annotated["parent_child_reason"] = (
                    f"high_value_child; contained={best_containment:.3f}; area_ratio={best_ratio:.3f}; "
                    f"parent={annotated['parent_region_id']}"
                )
            else:
                annotated["parent_child_status"] = "retained_parent_or_independent"
                annotated["parent_child_reason"] = "no_drop_parent_child_rule"
            retained.append(annotated)
    return retained, fallback


def select_branch_parent_child(
    branch: str,
    candidates: list[dict[str, Any]],
    eval_df: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    apply_parent_child: bool,
) -> list[dict[str, Any]]:
    selector_cfg = cfg.get("crop_selector", {})
    decisions: list[dict[str, Any]] = []
    for occurrence_id in eval_df["occurrenceID"].astype(str):
        record_candidates: list[dict[str, Any]] = []
        for candidate in candidates:
            if base.clean_str(candidate.get("occurrenceID")) != occurrence_id:
                continue
            row = {**candidate, "branch": branch}
            row["utility_score"] = base.candidate_utility(row)
            accepted, reason = base.evidence_gate(
                row,
                class_thresholds=selector_cfg.get("class_thresholds", {}),
                vision_only_confidence=float(selector_cfg.get("vision_only_confidence", 0.20)),
            )
            if accepted:
                row["evidence_gate_reason"] = reason
                record_candidates.append(row)
            else:
                decisions.append({
                    **row,
                    "evidence_gate_reason": reason,
                    "selection_status": "rejected_evidence",
                    "selection_reason": reason,
                    "duplicate_of_region_id": "",
                    "input_order": "",
                    "parent_child_status": base.clean_str(row.get("parent_child_status")) or "not_applied",
                    "parent_child_reason": base.clean_str(row.get("parent_child_reason")) or "rejected_before_parent_child",
                    "parent_region_id": base.clean_str(row.get("parent_region_id")),
                })

        if apply_parent_child:
            record_candidates, fallback = parent_child_prune(record_candidates)
            decisions.extend({
                **row,
                "selection_status": "fallback_parent_child",
                "selection_reason": row.get("parent_child_reason", "fallback_child_inside_parent"),
                "duplicate_of_region_id": row.get("parent_region_id", ""),
                "input_order": "",
            } for row in fallback)
        else:
            for row in record_candidates:
                row.setdefault("parent_child_status", "not_applied")
                row.setdefault("parent_child_reason", "parent_child_not_applied")
                row.setdefault("parent_region_id", "")

        selected, quota_rejected = base.select_diverse_crops(
            record_candidates,
            total_limit=int(selector_cfg.get("total_limit", 8)),
            role_quotas=selector_cfg.get("role_quotas", {}),
        )
        decisions.extend({**row, "branch": branch, "input_order": ""} for row in quota_rejected)
        decisions.extend({**row, "branch": branch} for row in selected)
    return decisions


def serializable_decisions(decisions: list[dict[str, Any]]) -> pd.DataFrame:
    frame = base.serializable_decisions(decisions)
    for column in ("parent_child_containment", "parent_child_area_ratio"):
        if column in frame.columns:
            frame[column] = frame[column].fillna("")
    return frame


def write_parent_child_summary(path: Path, decisions: list[dict[str, Any]]) -> None:
    rows = []
    for row in decisions:
        if not base.clean_str(row.get("parent_child_status")):
            continue
        rows.append({
            "branch": row.get("branch"),
            "catalogNumber": row.get("catalogNumber"),
            "occurrenceID": row.get("occurrenceID"),
            "region_id": row.get("region_id"),
            "component_type": row.get("component_type"),
            "detector_family": row.get("detector_family"),
            "selection_status": row.get("selection_status"),
            "parent_child_status": row.get("parent_child_status"),
            "parent_child_reason": row.get("parent_child_reason"),
            "parent_region_id": row.get("parent_region_id"),
            "detector_confidence": row.get("detector_confidence"),
            "bbox_xyxy": json.dumps(base.parse_bbox(row.get("bbox_xyxy"))),
            "raw_text": base.clean_str(row.get("raw_text"))[:250],
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def write_packets(processed: Path, decisions: list[dict[str, Any]], eval_df: pd.DataFrame) -> None:
    packets = []
    for branch in BRANCHES:
        for _, record in eval_df.iterrows():
            occurrence_id = base.clean_str(record.get("occurrenceID"))
            selected = sorted(
                [
                    row for row in decisions
                    if row["branch"] == branch
                    and base.clean_str(row.get("occurrenceID")) == occurrence_id
                    and row.get("selection_status") == "selected"
                ],
                key=lambda row: int(row.get("input_order") or 0),
            )
            packets.append({
                "branch": branch,
                "occurrenceID": occurrence_id,
                "reference_data_excluded": True,
                "images": [
                    {
                        "input_order": row.get("input_order"),
                        "region_id": row.get("region_id"),
                        "component_type": row.get("component_type"),
                        "component_role": base.component_role(row.get("component_type")),
                        "detector_family": row.get("detector_family"),
                        "bbox_xyxy": row.get("bbox_xyxy"),
                        "crop_path": row.get("crop_path"),
                        "review_crop_member": row.get("review_crop_member"),
                        "ocr_text_for_audit_only": row.get("raw_text"),
                        "decoded_barcode_for_audit_only": row.get("decoded_barcode"),
                        "selection_reason": row.get("selection_reason"),
                        "parent_child_status": row.get("parent_child_status"),
                        "parent_child_reason": row.get("parent_child_reason"),
                    }
                    for row in selected
                ],
            })
    base.write_jsonl(processed / "llm_input_packets.jsonl", packets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/yoloe26_llm_crop_eval10.yaml")
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)

    cfg, paths = base.load_runtime(args.config)
    processed = paths["processed"]
    print("[parent-child] loading main_data rows 2-11", flush=True)
    _, eval_df, _ = base.stage_metadata(args.config)
    if "source_row" in eval_df.columns:
        eval_df = eval_df.sort_values("source_row", key=lambda values: pd.to_numeric(values, errors="coerce")).reset_index(drop=True)
        eval_df.to_csv(processed / "eval_set.csv", index=False)
    manifest = base.stage_download(args.config)
    print("[parent-child] running Hespi", flush=True)
    base.stage_layout(args.config)
    hespi_frame = pd.read_csv(processed / "hespi_sheet_components.csv", dtype=str).fillna("")
    hespi = base.hespi_candidates(hespi_frame, set(eval_df["occurrenceID"].astype(str)))
    standard_yoloe, hierarchical_yoloe = base.run_yoloe(eval_df, manifest, cfg)

    all_base = hespi + standard_yoloe + hierarchical_yoloe
    all_base = base.add_clean_crops(all_base, manifest, paths, cfg)
    print(f"[parent-child] OCR on {len(all_base)} candidate crops", flush=True)
    all_base = base.add_ocr_evidence(all_base, cfg)
    by_variant = {
        "hespi": [row for row in all_base if row["detector_family"] == "hespi"],
        "hierarchical": [row for row in all_base if row["detector_variant"] == "class_aware"],
    }

    decisions: list[dict[str, Any]] = []
    decisions.extend(select_branch_parent_child("hespi", by_variant["hespi"], eval_df, cfg, apply_parent_child=False))
    decisions.extend(select_branch_parent_child("yoloe_parent_child", by_variant["hierarchical"], eval_df, cfg, apply_parent_child=True))
    decisions.extend(select_branch_parent_child(
        "hespi_yoloe_parent_child_union",
        by_variant["hespi"] + by_variant["hierarchical"],
        eval_df,
        cfg,
        apply_parent_child=True,
    ))

    base.prepare_artifact_members(decisions)
    record_metrics, branch_metrics, field_metrics, field_summary = base.build_metrics(decisions, eval_df)
    serializable_decisions(decisions).to_csv(processed / "crop_decisions.csv", index=False)
    serializable_decisions([row for row in decisions if row.get("selection_status") == "selected"]).to_csv(
        processed / "selected_crops.csv", index=False,
    )
    write_parent_child_summary(processed / "parent_child_summary.csv", decisions)
    record_metrics.to_csv(processed / "record_metrics.csv", index=False)
    branch_metrics.to_csv(processed / "branch_metrics.csv", index=False)
    field_metrics.to_csv(processed / "field_evidence_metrics.csv", index=False)
    field_summary.to_csv(processed / "field_evidence_summary.csv", index=False)
    write_packets(processed, decisions, eval_df)
    run_manifest = {
        "data_source": "data/fixtures/techtest_main_data_first10.csv",
        "source_workbook": "techtest_herbariumdata.xlsx",
        "source_workbook_sha256": "dcef830b778b967eb2e000a5f31d2654a480546f149e8e271d2189b4f2e047af",
        "source_sheet": "main_data",
        "source_rows": "2-11",
        "record_count": len(eval_df),
        "branches": list(BRANCHES),
        "parent_child_strategy": {
            "containment_threshold": 0.85,
            "child_parent_area_ratio": 0.35,
            "high_value_children_retained": sorted(HIGH_VALUE_CHILD_TYPES),
            "small_database_label_area_fraction_retained_if_lte": 0.030,
            "fallback_child_policy": "not in primary LLM packet; use if parent OCR is insufficient",
        },
        "max_crops_per_packet": int(cfg.get("crop_selector", {}).get("total_limit", 8)),
        "external_llm_calls": 0,
        "llm_backend": base.clean_str(cfg.get("llm", {}).get("backend", "none")) or "none",
        "reference_data_in_llm_packets": False,
    }
    (processed / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    base.write_report(paths["reports"] / "comparison_report.md", branch_metrics, field_summary)
    archive = base.build_review_bundle(
        paths["reports"],
        processed,
        decisions,
        eval_df,
        manifest,
        branch_metrics,
        field_metrics,
        field_summary,
    )
    print(branch_metrics.to_string(index=False), flush=True)
    print(f"[parent-child] review bundle: {archive}", flush=True)


if __name__ == "__main__":
    main()
