from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_yoloe26_llm_crop_eval10 as base  # noqa: E402
import run_yoloe26_parent_child_first10 as old_pc  # noqa: E402

BRANCHES = ("yoloe_parent_child", "yoloe_region_panel_v1", "hespi_yoloe_region_panel_union_v1")
base.BRANCHES = BRANCHES
PARENTS = {"primary_specimen_label", "printed_label", "annotation_label", "handwritten_data", "database_label", "type_label"}
DETAIL_ROLES = {"identifier_child", "annotation", "identifier_parent"}
SOURCE_BRANCH = {
    "yoloe_region_panel_v1": "yoloe_parent_child",
    "hespi_yoloe_region_panel_union_v1": "hespi_yoloe_parent_child_union",
}


def ctext(value: Any) -> str:
    return base.clean_str(value)


def bbox(row: dict[str, Any]) -> list[int]:
    return base.parse_bbox(row.get("bbox_xyxy"))


def area(box: Any) -> float:
    x0, y0, x1, y1 = base.parse_bbox(box)
    return float(max(0, x1 - x0) * max(0, y1 - y0))


def inter(a: Any, b: Any) -> float:
    ax0, ay0, ax1, ay1 = base.parse_bbox(a)
    bx0, by0, bx1, by1 = base.parse_bbox(b)
    return float(max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0)))


def iou(a: Any, b: Any) -> float:
    v = inter(a, b)
    u = area(a) + area(b) - v
    return v / u if u else 0.0


def contained(child: Any, parent: Any) -> float:
    return inter(child, parent) / max(area(child), 1.0)


def role(row: dict[str, Any]) -> str:
    return base.component_role(row.get("component_type"))


def comp(row: dict[str, Any]) -> str:
    return base.canonical_component_type(row.get("component_type"))


def score(row: dict[str, Any]) -> float:
    role_bonus = {"primary_label": 34, "annotation": 32, "identifier_parent": 24, "identifier_child": 22}.get(role(row), -80)
    conf = float(row.get("detector_confidence") or 0.0)
    sharp = min(float(row.get("sharpness_score") or 0.0), 5000.0) / 280.0
    frac = float(row.get("area_fraction") or 0.0)
    area_bonus = 12 if 0.002 <= frac <= 0.28 else (-20 if frac > 0.55 or frac < 0.0007 else 0)
    return round(conf * 100 + role_bonus + min(sharp, 14) + area_bonus, 6)


def visual_ok(row: dict[str, Any]) -> bool:
    if role(row) == "other":
        return False
    if float(row.get("detector_confidence") or 0.0) < 0.05:
        return False
    frac = float(row.get("area_fraction") or 0.0)
    return 0.0 < frac <= 0.82


def duplicate_reason(row: dict[str, Any], kept: dict[str, Any]) -> str:
    if role(row) != role(kept) and comp(row) != comp(kept):
        return ""
    ov = iou(row.get("bbox_xyxy"), kept.get("bbox_xyxy"))
    c1 = contained(row.get("bbox_xyxy"), kept.get("bbox_xyxy"))
    c2 = contained(kept.get("bbox_xyxy"), row.get("bbox_xyxy"))
    ratio = min(area(row.get("bbox_xyxy")), area(kept.get("bbox_xyxy"))) / max(area(row.get("bbox_xyxy")), area(kept.get("bbox_xyxy")), 1.0)
    if ov >= 0.72:
        return f"not_sent_duplicate_visual_crop_iou={ov:.3f}; kept={ctext(kept.get('region_id'))}"
    if max(c1, c2) >= 0.92 and ratio >= 0.62:
        return f"not_sent_duplicate_visual_crop_containment={max(c1, c2):.3f}; kept={ctext(kept.get('region_id'))}"
    return ""


def dedup(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda x: float(x.get("visual_score") or 0.0), reverse=True):
        why = ""
        dup = None
        for old in kept:
            why = duplicate_reason(row, old)
            if why:
                dup = old
                break
        if dup:
            rejected.append({**row, "selection_status": "not_sent_duplicate_visual_crop", "selection_reason": why, "duplicate_of_region_id": ctext(dup.get("region_id")), "input_order": ""})
        else:
            kept.append(row)
    return kept, rejected


def find_parent(row: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    child_area = max(area(row.get("bbox_xyxy")), 1.0)
    best = None
    best_key = (0.0, 0.0)
    for parent in rows:
        if parent is row or comp(parent) not in PARENTS:
            continue
        parent_area = max(area(parent.get("bbox_xyxy")), 1.0)
        if parent_area <= child_area * 1.05:
            continue
        ratio = child_area / parent_area
        cover = contained(row.get("bbox_xyxy"), parent.get("bbox_xyxy"))
        if cover >= 0.80 and ratio <= 0.72 and (cover, parent_area) > best_key:
            best = parent
            best_key = (cover, parent_area)
    return best


def union(rows: list[dict[str, Any]]) -> list[int]:
    boxes = [bbox(r) for r in rows]
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def make_panel(members: list[dict[str, Any]], out: Path) -> None:
    thumbs = []
    for idx, row in enumerate(members, 1):
        src = Path(ctext(row.get("crop_path")))
        if src.is_file():
            with Image.open(src) as im:
                im = im.convert("RGB")
                im.thumbnail((440, 320), Image.Resampling.LANCZOS)
                thumbs.append((im.copy(), f"{idx}. {ctext(row.get('component_type'))} | {ctext(row.get('detector_family'))}"))
    if not thumbs:
        return
    cols = 1 if len(thumbs) == 1 else 2
    cell_w, cell_h = 480, 380
    canvas = Image.new("RGB", (cols * cell_w, math.ceil(len(thumbs) / cols) * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (im, label) in enumerate(thumbs):
        col, row = idx % cols, idx // cols
        draw.text((col * cell_w + 12, row * cell_h + 8), label, fill="black")
        canvas.paste(im, (col * cell_w + (cell_w - im.width) // 2, row * cell_h + 30))
        draw.rectangle((col * cell_w + 6, row * cell_h + 4, (col + 1) * cell_w - 6, (row + 1) * cell_h - 6), outline="black", width=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, format="JPEG", quality=92)


def concat(rows: list[dict[str, Any]], key: str) -> str:
    vals = [f"[{i}] {ctext(r.get(key))}" for i, r in enumerate(rows, 1) if ctext(r.get(key))]
    return "\nPANEL_MEMBER_BOUNDARY\n".join(vals)


def choose_members(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parents = [r for r in rows if comp(r) in PARENTS]
    context = max(parents, key=lambda r: (area(r.get("bbox_xyxy")), float(r.get("visual_score") or 0))) if parents else max(rows, key=lambda r: float(r.get("visual_score") or 0))
    details = [r for r in rows if r is not context and role(r) in DETAIL_ROLES]
    details = sorted(details, key=lambda r: (role(r) == "identifier_child", float(r.get("visual_score") or 0)), reverse=True)[:5]
    members = [context] + details
    extra = [r for r in rows if r not in members]
    return members, extra


def selected_panel(branch: str, occ: str, cluster_id: str, members: list[dict[str, Any]], paths: dict[str, Path], order: int) -> dict[str, Any]:
    panel_id = f"{occ}::region_panel_{order:02d}_{base.safe_filename(cluster_id)}"
    panel_path = paths["crops"] / "llm_crop_eval" / "region_panel" / base.safe_filename(occ) / f"{base.safe_filename(panel_id)}.jpg"
    if len(members) > 1:
        make_panel(members, panel_path)
    first = members[0]
    return {**first, "branch": branch, "region_id": panel_id, "component_type": "region_panel" if len(members) > 1 else first.get("component_type"), "detector_family": "region_panel" if len(members) > 1 else first.get("detector_family"), "detector_variant": "region_panel_v1", "detector_model": "yoloe26_region_panel_v1", "detector_confidence": max(float(r.get("detector_confidence") or 0) for r in members), "bbox_xyxy": union(members), "crop_bbox_xyxy": union(members), "crop_path": str(panel_path) if len(members) > 1 and panel_path.is_file() else first.get("crop_path"), "raw_text": concat(members, "raw_text"), "decoded_barcode": concat(members, "decoded_barcode"), "ocr_engine": "panel_member_audit_concat", "ocr_status": "panel_member_audit_concat", "decoder_status": "panel_member_audit_concat", "selection_status": "selected", "selection_reason": "selected_region_panel_context_detail" if len(members) > 1 else "selected_standalone_region", "duplicate_of_region_id": "", "input_order": order, "physical_region_id": cluster_id, "parent_child_status": "selected_region_panel" if len(members) > 1 else "selected_standalone_region", "parent_child_reason": "context_parent_plus_detail_children", "panel_member_count": len(members), "panel_member_region_ids": [ctext(r.get("region_id")) for r in members]}


def region_branch(source_rows: list[dict[str, Any]], branch: str, eval_df: pd.DataFrame, paths: dict[str, Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for occ in eval_df["occurrenceID"].astype(str):
        candidates = []
        for row in source_rows:
            if ctext(row.get("occurrenceID")) == occ:
                row = {**row, "branch": branch, "visual_score": score(row)}
                if visual_ok(row):
                    candidates.append(row)
                else:
                    out.append({**row, "selection_status": "not_sent_visual_gate", "selection_reason": "not_label_like_visual_candidate", "input_order": ""})
        kept, rejects = dedup(candidates)
        out.extend({**r, "branch": branch} for r in rejects)
        clusters: dict[str, list[dict[str, Any]]] = {}
        for row in kept:
            parent = find_parent(row, kept)
            cid = ctext(parent.get("region_id")) if parent else ctext(row.get("region_id"))
            clusters.setdefault(cid, []).append({**row, "physical_region_id": cid, "parent_region_id": ctext(parent.get("region_id")) if parent else ""})
        ranked = sorted(clusters.items(), key=lambda kv: max(float(r.get("visual_score") or 0) for r in kv[1]) + 12 * len(kv[1]), reverse=True)
        for order, (cid, rows) in enumerate(ranked[:12], 1):
            members, extra = choose_members(rows)
            panel = selected_panel(branch, occ, cid, members, paths, order)
            out.append(panel)
            for i, member in enumerate(members, 1):
                out.append({**member, "branch": branch, "selection_status": "panel_member_in_selected_llm_image", "selection_reason": "included_inside_selected_region_panel", "duplicate_of_region_id": panel["region_id"], "input_order": "", "selected_panel_region_id": panel["region_id"], "panel_member_order": i})
            for member in extra:
                out.append({**member, "branch": branch, "selection_status": "not_sent_panel_member_quota", "selection_reason": "same_physical_region_lower_priority", "duplicate_of_region_id": panel["region_id"], "input_order": "", "selected_panel_region_id": panel["region_id"]})
        for cid, rows in ranked[12:]:
            out.extend({**r, "branch": branch, "selection_status": "not_sent_region_quota", "selection_reason": "physical_region_rank_below_adaptive_panel_limit", "input_order": ""} for r in rows)
    return out


def serializable(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = base.serializable_decisions(rows)
    for col in ["panel_member_region_ids"]:
        if col in frame.columns:
            frame[col] = frame[col].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, tuple, dict)) else v).fillna("")
    return frame


def write_region_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame([{
        "branch": r.get("branch"), "occurrenceID": r.get("occurrenceID"), "catalogNumber": r.get("catalogNumber"),
        "panel_region_id": r.get("selected_panel_region_id", r.get("region_id")), "member_region_id": r.get("region_id"),
        "member_status": r.get("selection_status"), "component_type": r.get("component_type"),
        "selection_reason": r.get("selection_reason"), "bbox_xyxy": json.dumps(base.parse_bbox(r.get("bbox_xyxy")))
    } for r in rows if ctext(r.get("branch")) in {"yoloe_region_panel_v1", "hespi_yoloe_region_panel_union_v1"}]).to_csv(path, index=False)


def write_packets(path: Path, rows: list[dict[str, Any]], eval_df: pd.DataFrame) -> None:
    packets = []
    for br in BRANCHES:
        for occ in eval_df["occurrenceID"].astype(str):
            selected = sorted([r for r in rows if r.get("branch") == br and ctext(r.get("occurrenceID")) == occ and r.get("selection_status") == "selected"], key=lambda r: int(r.get("input_order") or 0))
            packets.append({"branch": br, "occurrenceID": occ, "reference_data_excluded": True, "llm_image_unit": "region_panel_or_standalone_crop", "images": [{"input_order": r.get("input_order"), "region_id": r.get("region_id"), "component_type": r.get("component_type"), "crop_path": r.get("crop_path"), "review_crop_member": r.get("review_crop_member"), "panel_member_count": r.get("panel_member_count", ""), "panel_member_region_ids": r.get("panel_member_region_ids", [])} for r in selected]})
    base.write_jsonl(path / "llm_input_packets.jsonl", packets)


def main() -> None:
    base.BRANCHES = old_pc.BRANCHES
    old_pc.main()
    base.BRANCHES = BRANCHES
    cfg, paths = base.load_runtime("configs/yoloe26_llm_crop_eval10.yaml")
    processed = paths["processed"]
    eval_df = pd.read_csv(processed / "eval_set.csv", dtype=str).fillna("")
    old_rows = pd.read_csv(processed / "crop_decisions.csv", dtype=str).fillna("").to_dict("records")
    decisions = [{**r, "branch": "yoloe_parent_child"} for r in old_rows if r.get("branch") == "yoloe_parent_child"]
    for out_branch, source_branch in SOURCE_BRANCH.items():
        source = [r for r in old_rows if r.get("branch") == source_branch and r.get("detector_family") in {"yoloe26", "hespi"}]
        decisions.extend(region_branch(source, out_branch, eval_df, paths))
    base.prepare_artifact_members(decisions)
    record_metrics, branch_metrics, field_metrics, field_summary = base.build_metrics(decisions, eval_df)
    serializable(decisions).to_csv(processed / "crop_decisions.csv", index=False)
    serializable([r for r in decisions if r.get("selection_status") == "selected"]).to_csv(processed / "selected_crops.csv", index=False)
    old_pc.write_parent_child_summary(processed / "parent_child_summary.csv", decisions)
    write_region_summary(processed / "region_panel_summary.csv", decisions)
    record_metrics.to_csv(processed / "record_metrics.csv", index=False)
    branch_metrics.to_csv(processed / "branch_metrics.csv", index=False)
    field_metrics.to_csv(processed / "field_evidence_metrics.csv", index=False)
    field_summary.to_csv(processed / "field_evidence_summary.csv", index=False)
    write_packets(processed, decisions, eval_df)
    (processed / "run_manifest.json").write_text(json.dumps({"branches": list(BRANCHES), "record_count": len(eval_df), "region_panel_strategy": {"selection_unit": "physical_label_region_panel_or_standalone_crop", "uses_ocr_for_selection": False, "uses_audit_ocr_after_selection": True, "max_panel_images_per_record": 12, "different_physical_identifier_regions_are_preserved": True}}, indent=2), encoding="utf-8")
    (paths["reports"] / "region_panel_report.md").write_text("# YOLOE-26 region-panel LLM OCR queue eval10\n\n" + branch_metrics.to_markdown(index=False) + "\n", encoding="utf-8")
    archive = base.build_review_bundle(paths["reports"], processed, decisions, eval_df, pd.read_csv(processed / "image_manifest.csv", dtype=str).fillna(""), branch_metrics, field_metrics, field_summary)
    print(branch_metrics.to_string(index=False), flush=True)
    print(f"[region-panel] review bundle: {archive}", flush=True)


if __name__ == "__main__":
    main()
