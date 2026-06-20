from __future__ import annotations

import base64
import html
import io
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from PIL import Image, ImageDraw

OUT = Path("reports/yoloe26_hespi_compare")
OUT.mkdir(parents=True, exist_ok=True)
IMAGE_DIR = OUT / "raw_images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
HTML_PATH = OUT / "yoloe26_vs_hespi_eval10.html"
CSV_PATH = OUT / "detections.csv"
SUMMARY_PATH = OUT / "per_image_summary.csv"
BUNDLE = Path("app_data/component_aware_eval10_review_bundle.zip")
EVAL_MEMBER = "component_aware_eval10/processed/eval_set.csv"
PROMPTS = [
    "primary specimen label", "specimen label", "annotation label",
    "database label", "barcode label", "barcode", "catalog number",
    "handwritten note", "printed label", "stamp", "scale bar", "color chart",
]
HESPI_OCR_CLASSES = {
    "primary_specimen_label", "primary_label", "institutional_label",
    "handwritten_data", "annotation_label", "stamp", "swing_tag", "number",
    "small_database_label", "database_label", "full_database_label", "barcode",
}
YOLOE_EXCLUDE = {"scale bar", "color chart"}
HIGH_VALUE_CHILD = {"barcode", "barcode label", "catalog number", "number", "small database label"}
PARENT_LABELS = {"primary specimen label", "specimen label", "printed label", "annotation label", "database label", "handwritten note", "stamp"}


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", clean(value))[:100] or "record"


def load_eval10() -> pd.DataFrame:
    with zipfile.ZipFile(BUNDLE) as archive:
        with archive.open(EVAL_MEMBER) as handle:
            frame = pd.read_csv(handle, dtype=str).fillna("")
    url_candidates = [column for column in frame.columns if "url" in column.lower()]
    preferred = [column for column in url_candidates if column.lower() == "image_url"]
    if not preferred:
        preferred = [column for column in url_candidates if "image" in column.lower()]
    if not preferred:
        raise RuntimeError(f"No image URL column found in eval_set.csv. Columns: {list(frame.columns)}")
    url_column = preferred[0]
    catalog_column = "catalogNumber" if "catalogNumber" in frame.columns else "occurrenceID"
    selected = frame[[catalog_column, url_column]].copy()
    selected.columns = ["catalog", "image_url"]
    selected = selected[selected["image_url"].astype(str).str.startswith("http")].drop_duplicates("catalog")
    if len(selected) < 10:
        raise RuntimeError(f"Expected at least 10 raw image URLs; found {len(selected)}")
    return selected.iloc[:10].reset_index(drop=True)


def download_raw_images(frame: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "nhm-scribe-yoloe-pruning-comparison/1.0"})
    for index, row in frame.iterrows():
        catalog = clean(row["catalog"])
        url = clean(row["image_url"])
        destination = IMAGE_DIR / f"{index + 1:02d}_{safe_name(catalog)}.jpg"
        error = ""
        for attempt in range(4):
            try:
                response = session.get(url, timeout=120)
                response.raise_for_status()
                destination.write_bytes(response.content)
                with Image.open(destination) as image:
                    image.verify()
                rows.append({"catalog": catalog, "image_url": url, "image_path": str(destination)})
                error = ""
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                time.sleep(2 ** attempt)
        if error:
            raise RuntimeError(f"Could not download raw image for {catalog}: {error}")
    return rows


def box_rows(result: Any) -> list[dict[str, Any]]:
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    rows: list[dict[str, Any]] = []
    for box in boxes:
        class_id = int(box.cls.cpu().item())
        confidence = float(box.conf.cpu().item())
        coords = box.xyxy.cpu().tolist()[0]
        label = clean(names.get(class_id, str(class_id))).lower().replace(" ", "_")
        rows.append({
            "label": label,
            "confidence": round(confidence, 5),
            "bbox": [int(round(float(value))) for value in coords[:4]],
        })
    return sorted(rows, key=lambda row: row["confidence"], reverse=True)


def area(box: dict[str, Any]) -> int:
    x0, y0, x1, y1 = box["bbox"]
    return max(0, x1 - x0) * max(0, y1 - y0)


def intersection_area(left: dict[str, Any], right: dict[str, Any]) -> int:
    lx0, ly0, lx1, ly1 = left["bbox"]
    rx0, ry0, rx1, ry1 = right["bbox"]
    x0 = max(lx0, rx0); y0 = max(ly0, ry0)
    x1 = min(lx1, rx1); y1 = min(ly1, ry1)
    return max(0, x1 - x0) * max(0, y1 - y0)


def iou(left_bbox: list[int], right_bbox: list[int]) -> float:
    left = {"bbox": left_bbox}; right = {"bbox": right_bbox}
    inter = intersection_area(left, right)
    union = area(left) + area(right) - inter
    return inter / union if union else 0.0


def containment(child: dict[str, Any], parent: dict[str, Any]) -> float:
    child_area = area(child)
    return intersection_area(child, parent) / child_area if child_area else 0.0


def class_agnostic_nms(rows: list[dict[str, Any]], threshold: float = 0.55) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in sorted(rows, key=lambda row: float(row["confidence"]), reverse=True):
        if any(iou(candidate["bbox"], existing["bbox"]) >= threshold for existing in kept):
            continue
        candidate = dict(candidate)
        candidate["nms_status"] = "kept_after_class_agnostic_nms"
        kept.append(candidate)
    return kept


def prune_yoloe_candidates(rows: list[dict[str, Any]], image_size: tuple[int, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    image_w, image_h = image_size
    image_area = max(1, image_w * image_h)
    retained: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for candidate in sorted(rows, key=lambda row: area(row), reverse=True):
        candidate = dict(candidate)
        candidate_area = area(candidate)
        candidate_label = clean(candidate["label"])
        is_high_value = candidate_label in HIGH_VALUE_CHILD
        drop_reason = ""
        keep_reason = "parent_or_independent_region"
        for parent in retained:
            parent_area = area(parent)
            if parent_area <= candidate_area:
                continue
            if parent_area / image_area > 0.55:
                continue
            cont = containment(candidate, parent)
            ratio = candidate_area / parent_area if parent_area else 0.0
            if cont >= 0.85 and ratio <= 0.35:
                if is_high_value:
                    keep_reason = f"high_value_child_inside:{parent['label']};containment={cont:.2f};area_ratio={ratio:.2f}"
                    drop_reason = ""
                    break
                if parent.get("label") in PARENT_LABELS:
                    drop_reason = f"contained_in_parent:{parent['label']};containment={cont:.2f};area_ratio={ratio:.2f}"
                    break
        if drop_reason:
            candidate["prune_status"] = "dropped_child_inside_parent"
            candidate["prune_reason"] = drop_reason
            dropped.append(candidate)
        else:
            candidate["prune_status"] = "retained_for_ocr_queue"
            candidate["prune_reason"] = keep_reason
            retained.append(candidate)
    return sorted(retained, key=lambda row: float(row["confidence"]), reverse=True), sorted(dropped, key=lambda row: float(row["confidence"]), reverse=True)


def greedy_match(hespi_rows: list[dict[str, Any]], yoloe_rows: list[dict[str, Any]], threshold: float = 0.30):
    candidates: list[tuple[float, int, int]] = []
    for hi, hespi_row in enumerate(hespi_rows):
        for yi, yoloe_row in enumerate(yoloe_rows):
            score = iou(hespi_row["bbox"], yoloe_row["bbox"])
            if score >= threshold:
                candidates.append((score, hi, yi))
    pairs: list[dict[str, Any]] = []
    used_h: set[int] = set(); used_y: set[int] = set()
    for score, hi, yi in sorted(candidates, reverse=True):
        if hi in used_h or yi in used_y:
            continue
        used_h.add(hi); used_y.add(yi)
        pairs.append({"iou": score, "hespi": hespi_rows[hi], "yoloe": yoloe_rows[yi]})
    hespi_only = [row for index, row in enumerate(hespi_rows) if index not in used_h]
    yoloe_only = [row for index, row in enumerate(yoloe_rows) if index not in used_y]
    return pairs, hespi_only, yoloe_only


def add_header(canvas: Image.Image, title: str) -> ImageDraw.ImageDraw:
    drawing = ImageDraw.Draw(canvas)
    drawing.rectangle((0, 0, canvas.width, 42), fill=(255, 255, 255))
    drawing.text((10, 12), title, fill=(0, 0, 0))
    return drawing


def draw_rows(image: Image.Image, rows: list[dict[str, Any]], title: str, colour: tuple[int, int, int]) -> Image.Image:
    canvas = image.copy().convert("RGB")
    drawing = add_header(canvas, title)
    width = max(3, canvas.width // 850)
    for index, row in enumerate(rows, start=1):
        x0, y0, x1, y1 = row["bbox"]
        drawing.rectangle((x0, y0, x1, y1), outline=colour, width=width)
        label = f"{index}: {row['label']} {float(row['confidence']):.2f}"
        text_y = max(43, y0 - 20)
        drawing.rectangle((x0, text_y, min(canvas.width, x0 + 12 + 7 * len(label)), text_y + 19), fill=(255, 255, 255))
        drawing.text((x0 + 3, text_y + 3), label, fill=colour)
    return canvas


def draw_pruned(image: Image.Image, retained: list[dict[str, Any]], dropped: list[dict[str, Any]]) -> Image.Image:
    canvas = image.copy().convert("RGB")
    drawing = add_header(canvas, "YOLOE-26 pruning: blue=retained OCR queue, grey=dropped child")
    width = max(3, canvas.width // 850)
    for row in dropped:
        drawing.rectangle(tuple(row["bbox"]), outline=(130, 130, 130), width=max(2, width - 1))
    for index, row in enumerate(retained, start=1):
        x0, y0, x1, y1 = row["bbox"]
        drawing.rectangle((x0, y0, x1, y1), outline=(0, 80, 220), width=width)
        label = f"{index}: {row['label']} {float(row['confidence']):.2f}"
        text_y = max(43, y0 - 20)
        drawing.rectangle((x0, text_y, min(canvas.width, x0 + 12 + 7 * len(label)), text_y + 19), fill=(255, 255, 255))
        drawing.text((x0 + 3, text_y + 3), label, fill=(0, 80, 220))
    return canvas


def draw_difference(image: Image.Image, pairs, hespi_only, yoloe_only) -> Image.Image:
    canvas = image.copy().convert("RGB")
    drawing = add_header(canvas, "Retained queue vs Hespi: green=matched, orange=Hespi-only, blue=YOLOE-retained-only")
    width = max(3, canvas.width // 850)
    for pair in pairs:
        drawing.rectangle(tuple(pair["yoloe"]["bbox"]), outline=(0, 140, 0), width=width)
    for row in hespi_only:
        drawing.rectangle(tuple(row["bbox"]), outline=(230, 120, 0), width=width)
    for row in yoloe_only:
        drawing.rectangle(tuple(row["bbox"]), outline=(0, 80, 220), width=width)
    return canvas


def data_uri(image: Image.Image) -> str:
    preview = image.copy().convert("RGB")
    preview.thumbnail((1000, 1400))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def rows_table(rows: list[dict[str, Any]], include_reason: bool = False) -> str:
    if not rows:
        return "<p><em>None</em></p>"
    header = "<tr><th>#</th><th>region</th><th>confidence</th><th>bbox xyxy</th>"
    if include_reason:
        header += "<th>reason</th>"
    header += "</tr>"
    body_parts = []
    for index, row in enumerate(rows, start=1):
        body = f"<tr><td>{index}</td><td>{html.escape(clean(row['label']))}</td><td>{float(row['confidence']):.3f}</td><td>{html.escape(str(row['bbox']))}</td>"
        if include_reason:
            body += f"<td>{html.escape(clean(row.get('prune_reason', '')))}</td>"
        body += "</tr>"
        body_parts.append(body)
    return "<table><thead>" + header + "</thead><tbody>" + "".join(body_parts) + "</tbody></table>"


def pairs_table(pairs: list[dict[str, Any]]) -> str:
    if not pairs:
        return "<p><em>No matched regions at IoU >= 0.30.</em></p>"
    body = "".join(
        f"<tr><td>{html.escape(clean(pair['hespi']['label']))}</td><td>{html.escape(clean(pair['yoloe']['label']))}</td><td>{float(pair['iou']):.3f}</td></tr>"
        for pair in pairs
    )
    return "<table><thead><tr><th>Hespi region</th><th>Retained YOLOE region</th><th>IoU</th></tr></thead><tbody>" + body + "</tbody></table>"


def main() -> None:
    raw_images = download_raw_images(load_eval10())
    from hespi.hespi import Hespi
    from ultralytics import YOLO
    hespi_model = Hespi(gpu=False, htr=False, fuzzy=False, llm_model="none", force_download=False, sheet_component_res=2048, label_field_res=1280).sheet_component_model
    yoloe_model = YOLO("yoloe-26s-seg.pt")
    yoloe_model.set_classes(PROMPTS)
    cards: list[str] = []
    detection_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for item in raw_images:
        catalog = item["catalog"]
        image = Image.open(item["image_path"]).convert("RGB")
        hespi_result = hespi_model.predict(source=[item["image_path"]], show=False, save=False, batch=1, imgsz=2048, conf=0.15, verbose=False)[0]
        yoloe_result = yoloe_model.predict(source=[item["image_path"]], show=False, save=False, batch=1, imgsz=1024, conf=0.05, verbose=False)[0]
        hespi_regions = [row for row in box_rows(hespi_result) if row["label"] in HESPI_OCR_CLASSES]
        yoloe_raw = []
        for row in box_rows(yoloe_result):
            row["label"] = row["label"].replace("_", " ")
            if row["label"] not in YOLOE_EXCLUDE:
                yoloe_raw.append(row)
        yoloe_after_nms = class_agnostic_nms(yoloe_raw)
        yoloe_retained, yoloe_dropped = prune_yoloe_candidates(yoloe_after_nms, image.size)
        pairs, hespi_only, yoloe_retained_only = greedy_match(hespi_regions, yoloe_retained)
        summary_rows.append({
            "catalog": catalog,
            "hespi_regions": len(hespi_regions),
            "yoloe_raw": len(yoloe_raw),
            "yoloe_after_nms": len(yoloe_after_nms),
            "yoloe_retained_queue": len(yoloe_retained),
            "yoloe_dropped_children": len(yoloe_dropped),
            "matched_with_hespi": len(pairs),
            "hespi_only_vs_retained": len(hespi_only),
            "yoloe_retained_only": len(yoloe_retained_only),
        })
        for status, detector, rows in [
            ("hespi_retained", "hespi_yolov8_finetuned", hespi_regions),
            ("yoloe_raw", "yoloe_26s_open_vocab", yoloe_raw),
            ("yoloe_after_nms", "yoloe_26s_open_vocab", yoloe_after_nms),
            ("yoloe_retained_for_ocr_queue", "yoloe_26s_open_vocab", yoloe_retained),
            ("yoloe_dropped_child_inside_parent", "yoloe_26s_open_vocab", yoloe_dropped),
        ]:
            for row in rows:
                detection_rows.append({
                    "catalog": catalog,
                    "detector": detector,
                    "status": status,
                    "region": row["label"],
                    "confidence": row["confidence"],
                    "bbox_xyxy": json.dumps(row["bbox"]),
                    "prune_reason": row.get("prune_reason", ""),
                    "source_image_url": item["image_url"],
                })
        cards.append(
            f"<section><h2>{html.escape(catalog)}</h2><p><a href='{html.escape(item['image_url'])}'>Open raw source image</a></p>"
            "<div class='grid5'>"
            f"<figure><img src='{data_uri(image)}'><figcaption>1. Raw image</figcaption></figure>"
            f"<figure><img src='{data_uri(draw_rows(image, hespi_regions, 'Hespi fine-tuned YOLOv8', (210, 30, 30)))}'><figcaption>2. Hespi baseline</figcaption></figure>"
            f"<figure><img src='{data_uri(draw_rows(image, yoloe_after_nms, 'YOLOE-26 after NMS', (90, 90, 90)))}'><figcaption>3. YOLOE after NMS</figcaption></figure>"
            f"<figure><img src='{data_uri(draw_pruned(image, yoloe_retained, yoloe_dropped))}'><figcaption>4. Pruned OCR queue</figcaption></figure>"
            f"<figure><img src='{data_uri(draw_difference(image, pairs, hespi_only, yoloe_retained_only))}'><figcaption>5. Retained vs Hespi</figcaption></figure>"
            "</div>"
            f"<p><strong>Counts:</strong> Hespi {len(hespi_regions)}; YOLOE raw {len(yoloe_raw)}; after NMS {len(yoloe_after_nms)}; retained OCR queue {len(yoloe_retained)}; dropped children {len(yoloe_dropped)}; matched {len(pairs)}; Hespi-only {len(hespi_only)}; YOLOE-retained-only {len(yoloe_retained_only)}.</p>"
            "<div class='threecol'><div><h3>Matched retained YOLOE ↔ Hespi</h3>" + pairs_table(pairs) + "</div>"
            "<div><h3>YOLOE retained-only regions</h3>" + rows_table(yoloe_retained_only, include_reason=True) + "</div>"
            "<div><h3>Dropped YOLOE child regions</h3>" + rows_table(yoloe_dropped, include_reason=True) + "</div></div>"
            "<h3>Hespi-only regions after YOLOE pruning</h3>" + rows_table(hespi_only) + "</section>"
        )
    pd.DataFrame(detection_rows).to_csv(CSV_PATH, index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_PATH, index=False)
    totals = {key: int(summary[key].sum()) for key in ["hespi_regions", "yoloe_raw", "yoloe_after_nms", "yoloe_retained_queue", "yoloe_dropped_children", "matched_with_hespi", "hespi_only_vs_retained", "yoloe_retained_only"]}
    totals["images"] = int(len(summary))
    total_table = "<table><tbody>" + "".join(f"<tr><th>{html.escape(key)}</th><td>{value}</td></tr>" for key, value in totals.items()) + "</tbody></table>"
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>Containment-aware YOLOE-26 pruning vs Hespi</title><style>body{{font-family:Arial,sans-serif;max-width:2200px;margin:20px auto;padding:0 16px;line-height:1.42;color:#222}}section{{border-top:2px solid #ddd;margin-top:30px;padding-top:12px}}.grid5{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}}.threecol{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}img{{width:100%;height:auto;border:1px solid #bbb}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #ccc;padding:5px;text-align:left;vertical-align:top}}.note{{background:#eef6ff;padding:12px;border-left:5px solid #2374c6}}.warn{{background:#fff7dc;padding:12px;border-left:5px solid #d09a00}}@media(max-width:1400px){{.grid5{{grid-template-columns:repeat(2,1fr)}}.threecol{{grid-template-columns:1fr}}}}@media(max-width:700px){{.grid5{{grid-template-columns:1fr}}}}</style></head><body><h1>Containment-aware YOLOE-26 OCR candidate pruning vs Hespi</h1><p><strong>Executable GitHub Actions comparison.</strong> Both detectors receive the same raw, unannotated source image. YOLOE-26 is first used as a high-recall detector, then pruned before downstream OCR.</p><div class='note'><strong>Pruning rule:</strong> class-agnostic NMS first; then remove a smaller child box if at least 85% of it is inside a retained parent and its area is at most 35% of the parent, unless the child label is high-value: barcode, barcode label, catalog number, number, or small database label.</div><div class='warn'><strong>Interpretation limit:</strong> this is still a proxy comparison without human bounding-box ground truth. It measures what would be sent to OCR, not true precision or recall.</div><h2>Total counts</h2>{total_table}<h2>Per-image summary</h2>{summary.to_html(index=False, escape=True)}<h2>Per-image visual comparison</h2>{''.join(cards)}</body></html>"""
    HTML_PATH.write_text(document, encoding="utf-8")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
