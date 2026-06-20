from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import random
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageStat

from herbarium_scribe.component_aware import read_sheet_components
from herbarium_scribe.download import safe_filename
from herbarium_scribe.llm_crop_selector import (
    candidate_utility,
    canonical_component_type,
    component_role,
    evidence_gate,
    hierarchical_deduplicate,
    normalized_text,
    parse_bbox,
    REFERENCE_EVIDENCE_FIELDS,
    reference_evidence_hit,
    select_diverse_crops,
    selected_duplicate_pair_count,
)
from herbarium_scribe.metadata import clean_str
from herbarium_scribe.pipeline import load_runtime, stage_download, stage_layout, stage_metadata


BRANCHES = (
    "hespi",
    "yoloe_standard_nms",
    "yoloe_hierarchical",
    "hespi_yoloe_union",
)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clamp_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(int(x0), width - 1))
    y0 = max(0, min(int(y0), height - 1))
    x1 = max(x0 + 1, min(int(x1), width))
    y1 = max(y0 + 1, min(int(y1), height))
    return [x0, y0, x1, y1]


def build_clean_crop(
    image: Image.Image,
    bbox: list[int],
    out_path: Path,
    *,
    padding_fraction: float,
    min_short_side: int,
    max_long_side: int,
) -> tuple[list[int], float]:
    x0, y0, x1, y1 = clamp_bbox(bbox, image.width, image.height)
    pad_x = int(round((x1 - x0) * max(0.0, padding_fraction)))
    pad_y = int(round((y1 - y0) * max(0.0, padding_fraction)))
    crop_bbox = clamp_bbox(
        [x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y],
        image.width,
        image.height,
    )
    crop = image.crop(tuple(crop_bbox)).convert("RGB")
    short_side = max(1, min(crop.size))
    long_side = max(crop.size)
    scale = min(
        max_long_side / long_side if max_long_side > 0 else 1.0,
        max(1.0, min_short_side / short_side if min_short_side > 0 else 1.0),
    )
    if abs(scale - 1.0) > 0.01:
        crop = crop.resize(
            (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
            Image.Resampling.LANCZOS,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path, format="JPEG", quality=94)
    edges = crop.convert("L").filter(ImageFilter.FIND_EDGES)
    sharpness = float(ImageStat.Stat(edges).var[0])
    return crop_bbox, sharpness


def mask_iou_maps(mask_data: Any, region_ids: list[str]) -> list[dict[str, float]]:
    maps: list[dict[str, float]] = [{} for _ in region_ids]
    if mask_data is None:
        return maps
    masks = np.asarray(mask_data).astype(bool)
    if masks.ndim != 3 or len(masks) != len(region_ids):
        return maps
    for left in range(len(region_ids)):
        for right in range(left):
            intersection = int(np.logical_and(masks[left], masks[right]).sum())
            union = int(np.logical_or(masks[left], masks[right]).sum())
            value = intersection / union if union else 0.0
            if value:
                maps[left][region_ids[right]] = round(value, 6)
                maps[right][region_ids[left]] = round(value, 6)
    return maps


def yoloe_candidates_for_image(
    model: Any,
    image_path: str,
    occurrence_id: str,
    *,
    detector_variant: str,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    ycfg = cfg["yoloe"]
    result = model.predict(
        source=image_path,
        imgsz=int(ycfg.get("image_size", 1024)),
        conf=float(ycfg.get("confidence", 0.05)),
        iou=float(ycfg.get("iou", 0.55)),
        max_det=int(ycfg.get("max_detections", 60)),
        agnostic_nms=(
            bool(ycfg.get("agnostic_nms", True))
            if detector_variant == "standard_nms"
            else False
        ),
        device=clean_str(ycfg.get("device", "cpu")) or "cpu",
        verbose=False,
    )[0]
    if result.boxes is None:
        return []
    boxes = result.boxes.xyxy.detach().cpu().tolist()
    confidences = result.boxes.conf.detach().cpu().tolist()
    class_ids = result.boxes.cls.detach().cpu().tolist()
    names = result.names
    rows: list[dict[str, Any]] = []
    region_ids = []
    for index, (bbox, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
        prompt = names.get(int(class_id), str(int(class_id))) if isinstance(names, dict) else names[int(class_id)]
        component_type = canonical_component_type(prompt)
        region_id = f"{occurrence_id}::yoloe_{detector_variant}_{index:03d}_{component_type}"
        region_ids.append(region_id)
        rows.append({
            "occurrenceID": occurrence_id,
            "region_id": region_id,
            "component_type": component_type,
            "detector_family": "yoloe26",
            "detector_variant": detector_variant,
            "detector_model": clean_str(ycfg.get("model_name", "yoloe-26s-seg.pt")),
            "detector_confidence": round(float(confidence), 6),
            "bbox_xyxy": [int(round(value)) for value in bbox],
            "prompt_class": clean_str(prompt),
        })
    masks = None
    if result.masks is not None and getattr(result.masks, "data", None) is not None:
        masks = result.masks.data.detach().cpu().numpy()
    overlap_maps = mask_iou_maps(masks, region_ids)
    for index, row in enumerate(rows):
        row["mask_iou_by_region_id"] = overlap_maps[index]
        if masks is not None and index < len(masks):
            row["mask_area_fraction"] = round(float(np.asarray(masks[index]).mean()), 6)
        else:
            row["mask_area_fraction"] = ""
    return rows


def run_yoloe(
    eval_df: pd.DataFrame,
    manifest: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        from ultralytics import YOLOE
    except Exception as exc:
        raise RuntimeError("Install the yoloe extra before running this experiment") from exc

    ycfg = cfg["yoloe"]
    weights = clean_str(os.environ.get(clean_str(ycfg.get("weights_env", "YOLOE_WEIGHTS")), ""))
    weights = weights or clean_str(ycfg.get("model_name", "yoloe-26s-seg.pt"))
    model = YOLOE(weights)
    prompts = [clean_str(value) for value in ycfg.get("prompts", []) if clean_str(value)]
    model.set_classes(prompts)
    paths = dict(zip(manifest["occurrenceID"].astype(str), manifest["image_path"].astype(str)))
    standard: list[dict[str, Any]] = []
    hierarchical_source: list[dict[str, Any]] = []
    for _, record in eval_df.iterrows():
        occurrence_id = clean_str(record.get("occurrenceID"))
        image_path = paths.get(occurrence_id, "")
        if not image_path or not Path(image_path).is_file():
            continue
        print(f"[yoloe-crop-eval] YOLOE {occurrence_id}", flush=True)
        standard.extend(yoloe_candidates_for_image(
            model, image_path, occurrence_id, detector_variant="standard_nms", cfg=cfg,
        ))
        hierarchical_source.extend(yoloe_candidates_for_image(
            model, image_path, occurrence_id, detector_variant="class_aware", cfg=cfg,
        ))
    return standard, hierarchical_source


def hespi_candidates(components: pd.DataFrame, eval_ids: set[str]) -> list[dict[str, Any]]:
    rows = []
    for _, item in components[components["occurrenceID"].astype(str).isin(eval_ids)].iterrows():
        rows.append({
            "occurrenceID": clean_str(item.get("occurrenceID")),
            "catalogNumber": clean_str(item.get("catalogNumber")),
            "region_id": clean_str(item.get("region_id")),
            "component_type": canonical_component_type(item.get("component_type")),
            "detector_family": "hespi",
            "detector_variant": "standard_nms",
            "detector_model": clean_str(item.get("detector_model")),
            "detector_confidence": float(item.get("detector_confidence") or 0.0),
            "bbox_xyxy": parse_bbox(item.get("bbox_xyxy")),
            "prompt_class": clean_str(item.get("component_type")),
            "mask_iou_by_region_id": {},
            "mask_area_fraction": "",
        })
    return rows


def add_clean_crops(
    candidates: list[dict[str, Any]],
    manifest: pd.DataFrame,
    paths: dict[str, Path],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    selector_cfg = cfg.get("crop_selector", {})
    source_paths = dict(zip(manifest["occurrenceID"].astype(str), manifest["image_path"].astype(str)))
    image_cache: dict[str, Image.Image] = {}
    rows = []
    for candidate in candidates:
        occurrence_id = clean_str(candidate.get("occurrenceID"))
        source_path = source_paths.get(occurrence_id, "")
        if not source_path or not Path(source_path).is_file():
            continue
        if source_path not in image_cache:
            image_cache[source_path] = Image.open(source_path).convert("RGB")
        image = image_cache[source_path]
        bbox = clamp_bbox(parse_bbox(candidate.get("bbox_xyxy")), image.width, image.height)
        family = clean_str(candidate.get("detector_family"))
        crop_path = (
            paths["crops"] / "llm_crop_eval" / family / safe_filename(occurrence_id)
            / f"{safe_filename(clean_str(candidate.get('region_id')))}.jpg"
        )
        crop_bbox, sharpness = build_clean_crop(
            image,
            bbox,
            crop_path,
            padding_fraction=float(selector_cfg.get("padding_fraction", 0.08)),
            min_short_side=int(selector_cfg.get("min_short_side", 256)),
            max_long_side=int(selector_cfg.get("max_long_side", 1800)),
        )
        area_fraction = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (image.width * image.height)
        rows.append({
            **candidate,
            "bbox_xyxy": bbox,
            "crop_bbox_xyxy": crop_bbox,
            "source_image_path": source_path,
            "crop_path": str(crop_path),
            "source_width": image.width,
            "source_height": image.height,
            "area_fraction": round(area_fraction, 6),
            "sharpness_score": round(sharpness, 6),
        })
    for image in image_cache.values():
        image.close()
    return rows


def add_ocr_evidence(candidates: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    readings = read_sheet_components(pd.DataFrame(candidates), cfg)
    by_region = {clean_str(row.get("region_id")): row for _, row in readings.iterrows()}
    rows = []
    for candidate in candidates:
        reading = by_region.get(clean_str(candidate.get("region_id")))
        raw_text = clean_str(reading.get("raw_text")) if reading is not None else ""
        engine = clean_str(reading.get("engine")) if reading is not None else ""
        rows.append({
            **candidate,
            "raw_text": raw_text,
            "decoded_barcode": raw_text if engine == "zxingcpp" else "",
            "ocr_engine": engine,
            "ocr_status": clean_str(reading.get("ocr_status")) if reading is not None else "missing_reading",
            "decoder_status": clean_str(reading.get("decoder_status")) if reading is not None else "",
            "ocr_candidates_json": clean_str(reading.get("candidates_json")) if reading is not None else "[]",
        })
    return rows


def select_branch(
    branch: str,
    candidates: list[dict[str, Any]],
    eval_df: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    deduplicate: bool,
) -> list[dict[str, Any]]:
    selector_cfg = cfg.get("crop_selector", {})
    decisions: list[dict[str, Any]] = []
    for occurrence_id in eval_df["occurrenceID"].astype(str):
        record_candidates = []
        for candidate in candidates:
            if clean_str(candidate.get("occurrenceID")) != occurrence_id:
                continue
            row = {**candidate, "branch": branch}
            row["utility_score"] = candidate_utility(row)
            accepted, reason = evidence_gate(
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
                })
        dedup_rejected: list[dict[str, Any]] = []
        if deduplicate:
            record_candidates, dedup_rejected = hierarchical_deduplicate(record_candidates)
        selected, quota_rejected = select_diverse_crops(
            record_candidates,
            total_limit=int(selector_cfg.get("total_limit", 8)),
            role_quotas=selector_cfg.get("role_quotas", {}),
        )
        decisions.extend({**row, "branch": branch, "input_order": ""} for row in dedup_rejected)
        decisions.extend({**row, "branch": branch, "input_order": ""} for row in quota_rejected)
        decisions.extend({**row, "branch": branch} for row in selected)
    return decisions


def selected_evidence_text(selected: list[dict[str, Any]]) -> str:
    return "\nEVIDENCE_BOUNDARY\n".join(
        clean_str(row.get("decoded_barcode")) or clean_str(row.get("raw_text"))
        for row in selected
        if clean_str(row.get("decoded_barcode")) or clean_str(row.get("raw_text"))
    )


def build_metrics(
    decisions: list[dict[str, Any]],
    eval_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    record_rows = []
    field_rows = []
    records_by_id = {
        clean_str(record.get("occurrenceID")): record
        for _, record in eval_df.iterrows()
    }
    for branch in BRANCHES:
        for occurrence_id in eval_df["occurrenceID"].astype(str):
            reference = records_by_id[occurrence_id]
            rows = [
                row for row in decisions
                if row["branch"] == branch and clean_str(row.get("occurrenceID")) == occurrence_id
            ]
            selected = sorted(
                [row for row in rows if row.get("selection_status") == "selected"],
                key=lambda row: int(row.get("input_order") or 0),
            )
            usable = [row for row in selected if clean_str(row.get("raw_text")) or clean_str(row.get("decoded_barcode"))]
            unique_texts = {
                normalized_text(row.get("decoded_barcode") or row.get("raw_text"))
                for row in usable
                if normalized_text(row.get("decoded_barcode") or row.get("raw_text"))
            }
            evidence_text = selected_evidence_text(selected)
            reference_count = 0
            hit_count = 0
            for field in REFERENCE_EVIDENCE_FIELDS:
                value = clean_str(reference.get(field))
                available = bool(value)
                hit = available and reference_evidence_hit(field, value, evidence_text)
                reference_count += int(available)
                hit_count += int(hit)
                field_rows.append({
                    "branch": branch,
                    "occurrenceID": occurrence_id,
                    "catalogNumber": clean_str(reference.get("catalogNumber")),
                    "field": field,
                    "reference_value": value,
                    "reference_available": available,
                    "evidence_hit": hit,
                    "matching_scope": "selected_packet_ocr_and_barcode",
                })
            record_rows.append({
                "branch": branch,
                "occurrenceID": occurrence_id,
                "catalogNumber": clean_str(reference.get("catalogNumber")),
                "candidate_count": len(rows),
                "evidence_accepted_count": sum(row.get("selection_status") != "rejected_evidence" for row in rows),
                "input_image_count": len(selected),
                "ocr_usable_crop_count": len(usable),
                "ocr_usable_rate": round(len(usable) / len(selected), 6) if selected else 0.0,
                "unique_text_count": len(unique_texts),
                "text_uniqueness_rate": round(len(unique_texts) / len(usable), 6) if usable else 0.0,
                "selected_duplicate_pair_count": selected_duplicate_pair_count(selected),
                "reference_field_count": reference_count,
                "field_evidence_hit_count": hit_count,
                "field_evidence_hit_rate": round(hit_count / reference_count, 6) if reference_count else 0.0,
                "catalog_number_evidence_hit": reference_evidence_hit(
                    "catalogNumber",
                    reference.get("catalogNumber"),
                    evidence_text,
                ),
            })
    records = pd.DataFrame(record_rows)
    fields = pd.DataFrame(field_rows)
    field_summary_rows = []
    for branch in BRANCHES:
        for field in REFERENCE_EVIDENCE_FIELDS:
            group = fields[fields["branch"].eq(branch) & fields["field"].eq(field)]
            available = group[group["reference_available"]]
            hits = int(available["evidence_hit"].sum())
            field_summary_rows.append({
                "branch": branch,
                "field": field,
                "reference_available_count": len(available),
                "evidence_hit_count": hits,
                "evidence_hit_rate": round(hits / len(available), 4) if len(available) else 0.0,
            })
    field_summary = pd.DataFrame(field_summary_rows)
    branch_rows = []
    for branch in BRANCHES:
        group = records[records["branch"].eq(branch)]
        branch_fields = fields[fields["branch"].eq(branch) & fields["reference_available"]]
        branch_decisions = [row for row in decisions if row["branch"] == branch]
        selected = [row for row in branch_decisions if row.get("selection_status") == "selected"]
        usable_count = sum(bool(clean_str(row.get("raw_text")) or clean_str(row.get("decoded_barcode"))) for row in selected)
        branch_rows.append({
            "branch": branch,
            "record_count": len(group),
            "candidate_count": int(group["candidate_count"].sum()),
            "selected_crop_count": int(group["input_image_count"].sum()),
            "mean_input_images_per_record": round(float(group["input_image_count"].mean()), 4),
            "records_without_input": int(group["input_image_count"].eq(0).sum()),
            "catalog_number_evidence_hit_rate": round(float(group["catalog_number_evidence_hit"].mean()), 4),
            "field_reference_count": len(branch_fields),
            "field_evidence_hit_count": int(branch_fields["evidence_hit"].sum()),
            "field_evidence_hit_rate": round(float(branch_fields["evidence_hit"].mean()), 4) if len(branch_fields) else 0.0,
            "selected_crop_ocr_usable_rate": round(usable_count / len(selected), 4) if selected else 0.0,
            "mean_text_uniqueness_rate": round(float(group["text_uniqueness_rate"].mean()), 4),
            "selected_duplicate_pair_count": int(group["selected_duplicate_pair_count"].sum()),
            "evidence_rejection_count": sum(row.get("selection_status") == "rejected_evidence" for row in branch_decisions),
            "dedup_rejection_count": sum(row.get("selection_status") == "rejected_duplicate" for row in branch_decisions),
            "quota_rejection_count": sum(row.get("selection_status") == "rejected_quota" for row in branch_decisions),
        })
    return records, pd.DataFrame(branch_rows), fields, field_summary


def draw_overview(source_path: str, rows: list[dict[str, Any]], destination: Path, *, selected_only: bool) -> None:
    image = Image.open(source_path).convert("RGB")
    scale = min(1.0, 1600 / max(image.size))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    for row in rows:
        selected = row.get("selection_status") == "selected"
        if selected_only and not selected:
            continue
        bbox = [round(value * scale) for value in parse_bbox(row.get("bbox_xyxy"))]
        color = "#00d26a" if selected else "#ef4444"
        width = 5 if selected else 2
        draw.rectangle(bbox, outline=color, width=width)
        label = (
            f"{row.get('input_order')}: " if selected else ""
        ) + f"{row.get('detector_family')} {row.get('component_type')}"
        text_bbox = draw.textbbox((bbox[0], bbox[1]), label)
        draw.rectangle(text_bbox, fill="black")
        draw.text((bbox[0], bbox[1]), label, fill="white")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="JPEG", quality=90)


def image_data_uri(path: Path, *, max_side: int, quality: int = 76) -> str:
    if not path.is_file():
        return ""
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def prepare_artifact_members(decisions: list[dict[str, Any]]) -> None:
    for row in decisions:
        row["review_crop_member"] = ""
        if row.get("selection_status") != "selected":
            continue
        branch = safe_filename(row["branch"])
        record = safe_filename(clean_str(row.get("occurrenceID")))
        order = int(row.get("input_order") or 0)
        region = safe_filename(clean_str(row.get("region_id")))
        row["review_crop_member"] = f"assets/crops/{branch}/{record}/{order:02d}_{region}.jpg"


def build_review_bundle(
    report_dir: Path,
    processed: Path,
    decisions: list[dict[str, Any]],
    eval_df: pd.DataFrame,
    manifest: pd.DataFrame,
    branch_metrics: pd.DataFrame,
    field_metrics: pd.DataFrame,
    field_summary: pd.DataFrame,
) -> Path:
    bundle = report_dir / "review_bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / "assets" / "crops").mkdir(parents=True)
    (bundle / "assets" / "overviews").mkdir(parents=True)
    (bundle / "records").mkdir(parents=True)
    for source in processed.glob("*"):
        if source.is_file():
            destination = bundle / "processed" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    source_by_id = dict(zip(manifest["occurrenceID"].astype(str), manifest["image_path"].astype(str)))
    record_links = []
    embedded_records = []
    for _, record in eval_df.iterrows():
        occurrence_id = clean_str(record.get("occurrenceID"))
        slug = safe_filename(occurrence_id)
        source_path = source_by_id.get(occurrence_id, "")
        sections = []
        embedded_sections = []
        for branch in BRANCHES:
            rows = [
                row for row in decisions
                if row["branch"] == branch and clean_str(row.get("occurrenceID")) == occurrence_id
            ]
            selected = sorted(
                [row for row in rows if row.get("selection_status") == "selected"],
                key=lambda row: int(row.get("input_order") or 0),
            )
            branch_slug = safe_filename(branch)
            candidate_member = f"assets/overviews/{slug}_{branch_slug}_candidates.jpg"
            selected_member = f"assets/overviews/{slug}_{branch_slug}_selected.jpg"
            if source_path and Path(source_path).is_file():
                draw_overview(source_path, rows, bundle / candidate_member, selected_only=False)
                draw_overview(source_path, rows, bundle / selected_member, selected_only=True)
            crop_cards = []
            embedded_crop_cards = []
            for row in selected:
                member = clean_str(row.get("review_crop_member"))
                source = Path(clean_str(row.get("crop_path")))
                if member and source.is_file():
                    destination = bundle / member
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                crop_cards.append(
                    f'<figure><img src="../{html.escape(member)}"><figcaption>'
                    f'{row.get("input_order")}. {html.escape(clean_str(row.get("component_type")))} | '
                    f'{html.escape(clean_str(row.get("detector_family")))}<br>'
                    f'{html.escape(clean_str(row.get("raw_text"))[:160])}</figcaption></figure>'
                )
                crop_uri = image_data_uri(source, max_side=300)
                embedded_crop_cards.append(
                    f'<figure><img src="{crop_uri}"><figcaption>'
                    f'{row.get("input_order")}. {html.escape(clean_str(row.get("component_type")))} | '
                    f'{html.escape(clean_str(row.get("detector_family")))}<br>'
                    f'{html.escape(clean_str(row.get("raw_text"))[:220])}</figcaption></figure>'
                )
            sections.append(
                f'<section><h2>{html.escape(branch)}</h2><div class="overviews">'
                f'<img src="../{candidate_member}"><img src="../{selected_member}"></div>'
                f'<div class="crops">{"".join(crop_cards) or "No selected crops"}</div></section>'
            )
            record_fields = field_metrics[
                field_metrics["branch"].eq(branch)
                & field_metrics["occurrenceID"].astype(str).eq(occurrence_id)
            ]
            field_rows_html = "".join(
                "<tr>"
                f'<td>{html.escape(clean_str(item.get("field")))}</td>'
                f'<td>{html.escape(clean_str(item.get("reference_value"))) or "not available"}</td>'
                f'<td class="{("hit" if bool(item.get("evidence_hit")) else "miss")}">'
                f'{("hit" if bool(item.get("evidence_hit")) else ("miss" if bool(item.get("reference_available")) else "n/a"))}</td>'
                "</tr>"
                for _, item in record_fields.iterrows()
            )
            candidate_uri = image_data_uri(bundle / candidate_member, max_side=900)
            selected_uri = image_data_uri(bundle / selected_member, max_side=900)
            embedded_sections.append(
                f'<article class="branch"><h3>{html.escape(branch)}</h3>'
                f'<p>{len(rows)} candidates; {len(selected)} crops in the simulated LLM packet.</p>'
                f'<div class="overviews"><figure><img src="{candidate_uri}"><figcaption>All candidates: green selected, red rejected</figcaption></figure>'
                f'<figure><img src="{selected_uri}"><figcaption>Final packet order</figcaption></figure></div>'
                f'<h4>Reference field evidence</h4><table><thead><tr><th>Field</th><th>main_data reference</th><th>Evidence</th></tr></thead>'
                f'<tbody>{field_rows_html}</tbody></table><h4>Selected crops</h4>'
                f'<div class="crops">{"".join(embedded_crop_cards) or "No selected crops"}</div></article>'
            )
        page = f"""<!doctype html><meta charset="utf-8"><title>{html.escape(occurrence_id)}</title>
<style>body{{font-family:system-ui;margin:24px;background:#f6f7f9}}section{{background:white;padding:16px;margin:18px 0;border-radius:8px}}.overviews,.crops{{display:flex;gap:12px;flex-wrap:wrap}}.overviews img{{max-width:48%;height:auto}}figure{{width:260px;margin:0}}figure img{{width:100%;max-height:220px;object-fit:contain;background:#eee}}figcaption{{font-size:12px;white-space:pre-wrap}}a{{color:#155eef}}</style>
<a href="../index.html">Back to summary</a><h1>{html.escape(clean_str(record.get("catalogNumber")))}</h1><p>{html.escape(occurrence_id)}</p>{''.join(sections)}"""
        (bundle / "records" / f"{slug}.html").write_text(page, encoding="utf-8")
        record_links.append(
            f'<li><a href="records/{slug}.html">{html.escape(clean_str(record.get("catalogNumber")))} | {html.escape(occurrence_id)}</a></li>'
        )
        embedded_records.append(
            f'<details class="record" id="record-{slug}"><summary>'
            f'{html.escape(clean_str(record.get("catalogNumber")))} | {html.escape(occurrence_id)}</summary>'
            f'{"".join(embedded_sections)}</details>'
        )

    summary_html = branch_metrics.to_html(index=False, border=0)
    index = f"""<!doctype html><meta charset="utf-8"><title>YOLOE-26 LLM crop eval10</title>
<style>body{{font-family:system-ui;margin:28px;max-width:1400px}}table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #ddd;padding:7px}}th{{background:#f2f4f7}}li{{margin:8px 0}}</style>
<h1>YOLOE-26 LLM crop selector eval10</h1><p>This compares the final clean image packet sent to an LLM. No LLM was called.</p>{summary_html}<h2>Per-record review</h2><ul>{''.join(record_links)}</ul>"""
    (bundle / "index.html").write_text(index, encoding="utf-8")
    visual_report = report_dir / "visual_report.html"
    visual_html = f"""<!doctype html><html><head><meta charset="utf-8"><title>main_data first10 crop comparison</title>
<style>:root{{--ink:#16202a;--muted:#667085;--line:#d0d5dd;--paper:#fff;--bg:#f2f4f7;--accent:#175cd3}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:28px}}header,.summary,.record{{background:var(--paper);border:1px solid var(--line);border-radius:12px;margin-bottom:18px;padding:20px}}h1{{margin:0 0 8px}}h2,h3,h4{{margin-top:18px}}.note{{color:var(--muted)}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border-bottom:1px solid #eaecf0;padding:6px;text-align:left;vertical-align:top}}th{{background:#f9fafb}}.hit{{color:#067647;font-weight:700}}.miss{{color:#b42318}}summary{{font-size:18px;font-weight:700;cursor:pointer}}.branch{{border-top:3px solid var(--accent);margin-top:22px;padding-top:4px}}.overviews,.crops{{display:flex;gap:12px;flex-wrap:wrap}}.overviews figure{{width:min(48%,700px)}}figure{{margin:0;width:250px}}figure img{{display:block;width:100%;max-height:620px;object-fit:contain;background:#eef0f3;border-radius:6px}}figcaption{{font-size:11px;color:var(--muted);white-space:pre-wrap;margin-top:4px}}@media(max-width:800px){{.overviews figure{{width:100%}}main{{padding:10px}}}}</style></head>
<body><main><header><h1>Hespi vs YOLOE-26: main_data first 10</h1><p>Deterministic rows 2-11 from <code>main_data</code>. This report evaluates the final crop packet before multimodal inference; no external LLM was called.</p><p class="note">Field evidence hit means a normalized reference value appears in selected OCR/barcode evidence. It is evidence coverage, not extraction accuracy, and the supplied reference data may contain errors.</p></header>
<section class="summary"><h2>Branch packet metrics</h2>{branch_metrics.to_html(index=False, border=0)}<h2>Field evidence metrics</h2>{field_summary.to_html(index=False, border=0)}</section>{''.join(embedded_records)}</main></body></html>"""
    visual_report.write_text(visual_html, encoding="utf-8")
    shutil.copy2(visual_report, bundle / "visual_report.html")
    archive = report_dir / "review_bundle.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for source in bundle.rglob("*"):
            if source.is_file():
                handle.write(source, Path("yoloe26_llm_crop_eval10") / source.relative_to(bundle))
    return archive


def write_report(path: Path, branch_metrics: pd.DataFrame, field_summary: pd.DataFrame) -> None:
    lines = [
        "# YOLOE-26 LLM crop selector eval10",
        "",
        "This experiment stops immediately before multimodal LLM inference. It uses rows 2-11 from the supplied `main_data` worksheet. All branches share the same records, OCR settings, evidence gate, role quotas, and eight-image cap.",
        "",
        "## Branches",
        "",
        "- `hespi`: Hespi component boxes with the common gate and quota selector.",
        "- `yoloe_standard_nms`: YOLOE-26 with ordinary class-agnostic NMS.",
        "- `yoloe_hierarchical`: class-aware YOLOE output followed by evidence-aware hierarchical deduplication.",
        "- `hespi_yoloe_union`: Hespi plus class-aware YOLOE followed by hierarchical deduplication.",
        "",
        "## Final packet metrics",
        "",
        branch_metrics.to_markdown(index=False),
        "",
        "## Reference field evidence metrics",
        "",
        field_summary.to_markdown(index=False),
        "",
        "## Interpretation guardrails",
        "",
        "- A field evidence hit checks whether the normalized `main_data` reference value appears in selected OCR or barcode evidence; it is not detector mAP or LLM extraction accuracy.",
        "- Reference fields are used only after packet selection for evaluation and are never included in the simulated LLM input packet.",
        "- OCR usability and text uniqueness are packet-quality diagnostics, not CER/WER because verified full-label transcriptions are unavailable.",
        "- Ten records are sufficient for visual diagnosis, not a production accuracy claim.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def serializable_decisions(decisions: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(decisions)
    for column in ("bbox_xyxy", "crop_bbox_xyxy", "mask_iou_by_region_id"):
        if column in frame.columns:
            frame[column] = frame[column].map(
                lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value
            )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/yoloe26_llm_crop_eval10.yaml")
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)

    cfg, paths = load_runtime(args.config)
    processed = paths["processed"]
    print("[yoloe-crop-eval] loading main_data rows 2-11", flush=True)
    _, eval_df, _ = stage_metadata(args.config)
    if "source_row" in eval_df.columns:
        eval_df = eval_df.sort_values("source_row", key=lambda values: pd.to_numeric(values, errors="coerce")).reset_index(drop=True)
        eval_df.to_csv(processed / "eval_set.csv", index=False)
    manifest = stage_download(args.config)
    print("[yoloe-crop-eval] running Hespi", flush=True)
    stage_layout(args.config)
    hespi_frame = pd.read_csv(processed / "hespi_sheet_components.csv", dtype=str).fillna("")
    hespi = hespi_candidates(hespi_frame, set(eval_df["occurrenceID"].astype(str)))
    standard_yoloe, hierarchical_yoloe = run_yoloe(eval_df, manifest, cfg)

    all_base = hespi + standard_yoloe + hierarchical_yoloe
    all_base = add_clean_crops(all_base, manifest, paths, cfg)
    print(f"[yoloe-crop-eval] OCR on {len(all_base)} candidate crops", flush=True)
    all_base = add_ocr_evidence(all_base, cfg)
    by_variant = {
        "hespi": [row for row in all_base if row["detector_family"] == "hespi"],
        "standard": [row for row in all_base if row["detector_variant"] == "standard_nms" and row["detector_family"] == "yoloe26"],
        "hierarchical": [row for row in all_base if row["detector_variant"] == "class_aware"],
    }
    decisions = []
    decisions.extend(select_branch("hespi", by_variant["hespi"], eval_df, cfg, deduplicate=False))
    decisions.extend(select_branch("yoloe_standard_nms", by_variant["standard"], eval_df, cfg, deduplicate=False))
    decisions.extend(select_branch("yoloe_hierarchical", by_variant["hierarchical"], eval_df, cfg, deduplicate=True))
    decisions.extend(select_branch(
        "hespi_yoloe_union",
        by_variant["hespi"] + by_variant["hierarchical"],
        eval_df,
        cfg,
        deduplicate=True,
    ))
    prepare_artifact_members(decisions)
    record_metrics, branch_metrics, field_metrics, field_summary = build_metrics(decisions, eval_df)

    serializable_decisions(decisions).to_csv(processed / "crop_decisions.csv", index=False)
    serializable_decisions([
        row for row in decisions if row.get("selection_status") == "selected"
    ]).to_csv(processed / "selected_crops.csv", index=False)
    record_metrics.to_csv(processed / "record_metrics.csv", index=False)
    branch_metrics.to_csv(processed / "branch_metrics.csv", index=False)
    field_metrics.to_csv(processed / "field_evidence_metrics.csv", index=False)
    field_summary.to_csv(processed / "field_evidence_summary.csv", index=False)
    packets = []
    for branch in BRANCHES:
        for _, record in eval_df.iterrows():
            occurrence_id = clean_str(record.get("occurrenceID"))
            selected = sorted(
                [
                    row for row in decisions
                    if row["branch"] == branch
                    and clean_str(row.get("occurrenceID")) == occurrence_id
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
                        "component_role": component_role(row.get("component_type")),
                        "detector_family": row.get("detector_family"),
                        "bbox_xyxy": row.get("bbox_xyxy"),
                        "crop_path": row.get("crop_path"),
                        "review_crop_member": row.get("review_crop_member"),
                        "ocr_text_for_audit_only": row.get("raw_text"),
                        "decoded_barcode_for_audit_only": row.get("decoded_barcode"),
                        "selection_reason": row.get("selection_reason"),
                    }
                    for row in selected
                ],
            })
    write_jsonl(processed / "llm_input_packets.jsonl", packets)
    run_manifest = {
        "data_source": "data/fixtures/techtest_main_data_first10.csv",
        "source_workbook": "techtest_herbariumdata.xlsx",
        "source_workbook_sha256": "dcef830b778b967eb2e000a5f31d2654a480546f149e8e271d2189b4f2e047af",
        "source_sheet": "main_data",
        "source_rows": "2-11",
        "record_count": len(eval_df),
        "branches": list(BRANCHES),
        "max_crops_per_packet": int(cfg.get("crop_selector", {}).get("total_limit", 8)),
        "external_llm_calls": 0,
        "llm_backend": clean_str(cfg.get("llm", {}).get("backend", "none")) or "none",
        "reference_data_in_llm_packets": False,
    }
    (processed / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_path = paths["reports"] / "comparison_report.md"
    write_report(report_path, branch_metrics, field_summary)
    archive = build_review_bundle(
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
    print(f"[yoloe-crop-eval] review bundle: {archive}", flush=True)


if __name__ == "__main__":
    main()
