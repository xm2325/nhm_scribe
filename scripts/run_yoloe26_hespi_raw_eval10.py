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

OUT = Path("reports/yoloe26_hespi_raw_compare")
OUT.mkdir(parents=True, exist_ok=True)
IMAGE_DIR = OUT / "raw_images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
HTML_PATH = OUT / "yoloe26_vs_hespi_raw_eval10.html"
CSV_PATH = OUT / "detections_raw_and_filtered.csv"
PAIR_PATH = OUT / "matched_regions.csv"
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
    session.headers.update({"User-Agent": "nhm-scribe-raw-yoloe-hespi-comparison/1.0"})
    for index, row in frame.iterrows():
        catalog = clean(row["catalog"])
        url = clean(row["image_url"])
        destination = IMAGE_DIR / f"{index + 1:02d}_{safe_name(catalog)}.jpg"
        last_error = ""
        for attempt in range(4):
            try:
                response = session.get(url, timeout=120)
                response.raise_for_status()
                destination.write_bytes(response.content)
                with Image.open(destination) as image:
                    image.verify()
                rows.append({"catalog": catalog, "image_url": url, "image_path": str(destination)})
                last_error = ""
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(2 ** attempt)
        if last_error:
            raise RuntimeError(f"Could not download raw image for {catalog}: {last_error}")
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


def iou(left: list[int], right: list[int]) -> float:
    x0 = max(left[0], right[0]); y0 = max(left[1], right[1])
    x1 = min(left[2], right[2]); y1 = min(left[3], right[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def class_agnostic_nms(rows: list[dict[str, Any]], threshold: float = 0.55) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in sorted(rows, key=lambda row: float(row["confidence"]), reverse=True):
        if any(iou(candidate["bbox"], existing["bbox"]) >= threshold for existing in kept):
            continue
        kept.append(candidate)
    return kept


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


def draw_difference(image: Image.Image, pairs, hespi_only, yoloe_only) -> Image.Image:
    canvas = image.copy().convert("RGB")
    drawing = add_header(canvas, "Difference view: green=matched, orange=Hespi-only, blue=YOLOE-only")
    width = max(3, canvas.width // 850)
    for pair in pairs:
        drawing.rectangle(tuple(pair["hespi"]["bbox"]), outline=(0, 140, 0), width=width)
    for row in hespi_only:
        drawing.rectangle(tuple(row["bbox"]), outline=(230, 120, 0), width=width)
    for row in yoloe_only:
        drawing.rectangle(tuple(row["bbox"]), outline=(0, 80, 220), width=width)
    return canvas


def data_uri(image: Image.Image) -> str:
    preview = image.copy().convert("RGB")
    preview.thumbnail((950, 1350))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def rows_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p><em>None</em></p>"
    body = "".join(
        "<tr>" f"<td>{index}</td><td>{html.escape(clean(row['label']))}</td>"
        f"<td>{float(row['confidence']):.3f}</td><td>{html.escape(str(row['bbox']))}</td>" "</tr>"
        for index, row in enumerate(rows, start=1)
    )
    return "<table><thead><tr><th>#</th><th>region</th><th>confidence</th><th>bbox xyxy</th></tr></thead><tbody>" + body + "</tbody></table>"


def pairs_table(pairs: list[dict[str, Any]]) -> str:
    if not pairs:
        return "<p><em>No matched regions at IoU >= 0.30.</em></p>"
    body = "".join(
        "<tr>" f"<td>{html.escape(clean(pair['hespi']['label']))}</td>"
        f"<td>{html.escape(clean(pair['yoloe']['label']))}</td><td>{float(pair['iou']):.3f}</td>" "</tr>"
        for pair in pairs
    )
    return "<table><thead><tr><th>Hespi region</th><th>YOLOE-26 region</th><th>IoU</th></tr></thead><tbody>" + body + "</tbody></table>"


def main() -> None:
    eval10 = load_eval10()
    raw_images = download_raw_images(eval10)
    from hespi.hespi import Hespi
    from ultralytics import YOLO

    hespi = Hespi(gpu=False, htr=False, fuzzy=False, llm_model="none", force_download=False, sheet_component_res=2048, label_field_res=1280)
    hespi_model = hespi.sheet_component_model
    yoloe_model = YOLO("yoloe-26s-seg.pt")
    yoloe_model.set_classes(PROMPTS)

    cards: list[str] = []
    detection_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []

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
        yoloe_regions = class_agnostic_nms(yoloe_raw, threshold=0.55)
        pairs, hespi_only, yoloe_only = greedy_match(hespi_regions, yoloe_regions, threshold=0.30)
        summary_records.append({
            "catalog": catalog, "hespi_regions": len(hespi_regions), "yoloe_raw_regions": len(yoloe_raw),
            "yoloe_after_nms": len(yoloe_regions), "matched_regions": len(pairs),
            "hespi_only": len(hespi_only), "yoloe_only": len(yoloe_only),
        })
        for status, detector, rows in [
            ("retained", "hespi_yolov8_finetuned", hespi_regions),
            ("raw", "yoloe_26s_open_vocab", yoloe_raw),
            ("after_class_agnostic_nms", "yoloe_26s_open_vocab", yoloe_regions),
        ]:
            for row in rows:
                detection_rows.append({
                    "catalog": catalog, "detector": detector, "status": status,
                    "region": row["label"], "confidence": row["confidence"],
                    "bbox_xyxy": json.dumps(row["bbox"]), "source_image_url": item["image_url"],
                })
        for pair in pairs:
            pair_rows.append({
                "catalog": catalog, "hespi_region": pair["hespi"]["label"],
                "yoloe_region": pair["yoloe"]["label"], "iou": round(float(pair["iou"]), 5),
                "hespi_bbox": json.dumps(pair["hespi"]["bbox"]), "yoloe_bbox": json.dumps(pair["yoloe"]["bbox"]),
            })
        cards.append(
            f"<section><h2>{html.escape(catalog)}</h2><p><a href='{html.escape(item['image_url'])}'>Open raw source image</a></p>"
            "<div class='grid4'>"
            f"<figure><img src='{data_uri(image)}'><figcaption>1. Raw source image: no detector boxes</figcaption></figure>"
            f"<figure><img src='{data_uri(draw_rows(image, hespi_regions, 'Hespi fine-tuned YOLOv8 only', (210, 30, 30)))}'><figcaption>2. Hespi-only boxes</figcaption></figure>"
            f"<figure><img src='{data_uri(draw_rows(image, yoloe_regions, 'YOLOE-26s only after class-agnostic NMS', (0, 80, 220)))}'><figcaption>3. YOLOE-26-only boxes after NMS</figcaption></figure>"
            f"<figure><img src='{data_uri(draw_difference(image, pairs, hespi_only, yoloe_only))}'><figcaption>4. Difference view</figcaption></figure>"
            "</div>"
            f"<p><strong>Counts:</strong> Hespi {len(hespi_regions)}; YOLOE raw {len(yoloe_raw)}; YOLOE after NMS {len(yoloe_regions)}; matched {len(pairs)}; Hespi-only {len(hespi_only)}; YOLOE-only {len(yoloe_only)}.</p>"
            "<div class='threecol'><div><h3>Matched pairs</h3>" + pairs_table(pairs) + "</div>"
            "<div><h3>Hespi-only regions</h3>" + rows_table(hespi_only) + "</div>"
            "<div><h3>YOLOE-26-only regions after NMS</h3>" + rows_table(yoloe_only) + "</div></div></section>"
        )

    detections = pd.DataFrame(detection_rows)
    detections.to_csv(CSV_PATH, index=False)
    pd.DataFrame(pair_rows).to_csv(PAIR_PATH, index=False)
    summary = pd.DataFrame(summary_records)
    summary.to_csv(OUT / "per_image_summary.csv", index=False)
    totals = {
        "images": len(summary), "hespi_regions": int(summary["hespi_regions"].sum()),
        "yoloe_raw_regions": int(summary["yoloe_raw_regions"].sum()),
        "yoloe_after_nms": int(summary["yoloe_after_nms"].sum()),
        "matched_regions": int(summary["matched_regions"].sum()),
        "hespi_only": int(summary["hespi_only"].sum()), "yoloe_only": int(summary["yoloe_only"].sum()),
    }
    total_table = "<table><tbody>" + "".join(f"<tr><th>{html.escape(key)}</th><td>{value}</td></tr>" for key, value in totals.items()) + "</tbody></table>"
    per_image_table = summary.to_html(index=False, escape=True)
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>Raw-image YOLOE-26 vs Hespi comparison</title>
<style>body{{font-family:Arial,sans-serif;max-width:1900px;margin:20px auto;padding:0 16px;line-height:1.42;color:#222}}section{{border-top:2px solid #ddd;margin-top:30px;padding-top:12px}}.grid4{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}.threecol{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}img{{width:100%;height:auto;border:1px solid #bbb}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #ccc;padding:5px;text-align:left;vertical-align:top}}.note{{background:#eef6ff;padding:12px;border-left:5px solid #2374c6}}.warn{{background:#fff7dc;padding:12px;border-left:5px solid #d09a00}}@media(max-width:1100px){{.grid4,.threecol{{grid-template-columns:1fr 1fr}}}}@media(max-width:700px){{.grid4,.threecol{{grid-template-columns:1fr}}}}</style></head><body>
<h1>Raw-image herbarium layout detection: YOLOE-26 vs Hespi fine-tuned YOLOv8</h1>
<p><strong>Corrected executable comparison.</strong> Each detector receives the same unannotated raw source image. The first panel is not taken from a previous overview bundle and contains no detector boxes.</p>
<div class='note'><strong>Four-panel reading guide:</strong> panel 1 is the raw input; panel 2 contains only Hespi boxes in red; panel 3 contains only YOLOE-26 boxes in blue after class-agnostic NMS; panel 4 shows matched regions in green, Hespi-only regions in orange, and YOLOE-only regions in blue.</div>
<div class='warn'><strong>Interpretation limit:</strong> this remains a visual and proxy comparison because there are no human-reviewed bounding-box ground-truth annotations for these ten images. YOLOE-26 uses open-vocabulary prompts without herbarium-specific fine-tuning. Hespi uses herbarium-specific fine-tuned weights.</div>
<h2>Methods</h2><p><strong>Raw input:</strong> image URLs stored in component_aware_eval10/processed/eval_set.csv. <strong>Hespi:</strong> sheet-component checkpoint, imgsz 2048, conf 0.15. <strong>YOLOE-26:</strong> yoloe-26s-seg.pt, imgsz 1024, conf 0.05, followed by class-agnostic NMS at IoU 0.55. <strong>Pair matching:</strong> greedy one-to-one matching at IoU >= 0.30.</p>
<h2>Total proxy counts</h2>{total_table}<h2>Per-image summary</h2>{per_image_table}<h2>Per-image visual comparison</h2>{''.join(cards)}
<h2>Source notes</h2><ul><li>Ultralytics YOLO26 documentation: https://docs.ultralytics.com/models/yolo26/</li><li>Ultralytics YOLOE documentation: https://docs.ultralytics.com/models/yoloe/</li><li>Hespi repository: https://github.com/rbturnbull/hespi</li></ul></body></html>"""
    HTML_PATH.write_text(document, encoding="utf-8")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
