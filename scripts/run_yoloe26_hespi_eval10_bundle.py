from __future__ import annotations

import base64
import html
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw

OUT = Path("reports/yoloe26_hespi_compare")
OUT.mkdir(parents=True, exist_ok=True)
IMAGE_DIR = OUT / "frozen_eval10_images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
HTML_PATH = OUT / "yoloe26_vs_hespi_eval10.html"
CSV_PATH = OUT / "detections.csv"
BUNDLES = [
    Path("app_data/hespi_v10_ocr_visual_report.zip"),
    Path("app_data/component_aware_eval10_review_bundle.zip"),
]
REVIEW_JSON = Path("app_data/gpt_primary_label_reviews_eval10.json")
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


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", clean(value).lower())


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", clean(value))[:100] or "record"


def review_catalogs() -> list[str]:
    payload = json.loads(REVIEW_JSON.read_text(encoding="utf-8"))
    return [
        clean(record.get("display_identifier") or record.get("catalog_reference"))
        for record in payload.get("records", [])
        if clean(record.get("display_identifier") or record.get("catalog_reference"))
    ]


def member_key(member: str) -> str:
    stem = Path(member).stem
    return norm(stem[:-3] if stem.endswith("_00") else stem)


def extract_eval10() -> tuple[list[dict[str, str]], Path]:
    catalogs = review_catalogs()
    for bundle in BUNDLES:
        if not bundle.exists():
            continue
        with zipfile.ZipFile(bundle) as archive:
            members = sorted(
                name for name in archive.namelist()
                if "/assets/overviews/" in name and name.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            if len(members) < 10:
                continue
            by_key = {member_key(member): member for member in members}
            chosen: list[tuple[str, str]] = []
            used: set[str] = set()
            for catalog in catalogs:
                target = norm(catalog)
                member = by_key.get(target, "")
                if not member:
                    for candidate_key, candidate_member in by_key.items():
                        if target and candidate_key and (target in candidate_key or candidate_key in target):
                            member = candidate_member
                            break
                if member and member not in used:
                    chosen.append((catalog, member))
                    used.add(member)
            for member in members:
                if len(chosen) >= 10:
                    break
                if member not in used:
                    chosen.append((Path(member).stem, member))
                    used.add(member)
            if len(chosen) < 10:
                continue
            rows: list[dict[str, str]] = []
            for index, (catalog, member) in enumerate(chosen[:10], start=1):
                suffix = Path(member).suffix.lower() or ".jpg"
                image_path = IMAGE_DIR / f"{index:02d}_{safe_name(catalog)}{suffix}"
                image_path.write_bytes(archive.read(member))
                rows.append({"catalog": catalog, "image_path": str(image_path), "bundle_member": member})
            return rows, bundle
    raise RuntimeError("The committed review bundles do not contain ten overview images.")


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


def draw_boxes(image: Image.Image, boxes: list[dict[str, Any]], title: str) -> Image.Image:
    canvas = image.copy().convert("RGB")
    drawing = ImageDraw.Draw(canvas)
    drawing.rectangle((0, 0, canvas.width, 34), fill=(255, 255, 255))
    drawing.text((8, 9), title, fill=(0, 0, 0))
    for index, item in enumerate(boxes, start=1):
        bbox = tuple(item["bbox"])
        drawing.rectangle(bbox, outline=(220, 40, 40), width=max(3, canvas.width // 850))
        label = f"{index}: {item['label']} {float(item['confidence']):.2f}"
        y = max(35, bbox[1] - 18)
        drawing.rectangle((bbox[0], y, min(canvas.width, bbox[0] + 10 + 7 * len(label)), y + 18), fill=(255, 255, 255))
        drawing.text((bbox[0] + 3, y + 2), label, fill=(180, 20, 20))
    return canvas


def data_uri(image: Image.Image) -> str:
    preview = image.copy().convert("RGB")
    preview.thumbnail((900, 1200))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=80, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p><em>No OCR-candidate region detected.</em></p>"
    body = "".join(
        "<tr>"
        f"<td>{index}</td><td>{html.escape(clean(row['label']))}</td>"
        f"<td>{float(row['confidence']):.3f}</td><td>{html.escape(str(row['bbox']))}</td>"
        "</tr>"
        for index, row in enumerate(rows, start=1)
    )
    return "<table><thead><tr><th>#</th><th>region</th><th>confidence</th><th>bbox xyxy</th></tr></thead><tbody>" + body + "</tbody></table>"


def main() -> None:
    images, bundle = extract_eval10()
    from hespi.hespi import Hespi
    from ultralytics import YOLO

    hespi = Hespi(gpu=False, htr=False, fuzzy=False, llm_model="none", force_download=False, sheet_component_res=2048, label_field_res=1280)
    hespi_model = hespi.sheet_component_model
    yoloe_model = YOLO("yoloe-26s-seg.pt")
    yoloe_model.set_classes(PROMPTS)

    csv_rows: list[dict[str, Any]] = []
    cards: list[str] = []
    overlap_scores: list[float] = []
    hespi_total = 0
    yoloe_total = 0

    for item in images:
        catalog = item["catalog"]
        image = Image.open(item["image_path"]).convert("RGB")
        hespi_result = hespi_model.predict(source=[item["image_path"]], show=False, save=False, batch=1, imgsz=2048, conf=0.15, verbose=False)[0]
        yoloe_result = yoloe_model.predict(source=[item["image_path"]], show=False, save=False, batch=1, imgsz=1024, conf=0.05, verbose=False)[0]
        hespi_regions = [row for row in box_rows(hespi_result) if row["label"] in HESPI_OCR_CLASSES]
        yoloe_regions = []
        for row in box_rows(yoloe_result):
            row["label"] = row["label"].replace("_", " ")
            if row["label"] not in YOLOE_EXCLUDE:
                yoloe_regions.append(row)
        hespi_total += len(hespi_regions)
        yoloe_total += len(yoloe_regions)
        scores = [max([iou(h["bbox"], y["bbox"]) for y in yoloe_regions] or [0.0]) for h in hespi_regions]
        overlap_scores.extend(scores)
        for detector, rows in (("hespi_yolov8_finetuned", hespi_regions), ("yoloe_26s_open_vocab", yoloe_regions)):
            for row in rows:
                csv_rows.append({
                    "catalog": catalog, "detector": detector, "region": row["label"],
                    "confidence": row["confidence"], "bbox_xyxy": json.dumps(row["bbox"]),
                    "bundle_member": item["bundle_member"],
                })
        note = "No Hespi OCR region for overlap proxy." if not scores else f"Hespi regions with YOLOE IoU >= 0.30: {sum(score >= 0.30 for score in scores)}/{len(scores)}; mean best IoU: {sum(scores) / len(scores):.3f}."
        cards.append(
            f"<section><h2>{html.escape(catalog)}</h2><p>{html.escape(item['bundle_member'])}</p>"
            "<div class='grid'>"
            f"<figure><img src='{data_uri(image)}'><figcaption>Original</figcaption></figure>"
            f"<figure><img src='{data_uri(draw_boxes(image, hespi_regions, 'Hespi fine-tuned YOLOv8: OCR candidate regions'))}'><figcaption>Hespi fine-tuned YOLOv8</figcaption></figure>"
            f"<figure><img src='{data_uri(draw_boxes(image, yoloe_regions, 'YOLOE-26s open-vocabulary: OCR candidate regions'))}'><figcaption>YOLOE-26s open-vocabulary prompts</figcaption></figure>"
            "</div>"
            f"<p>{html.escape(note)}</p>"
            "<div class='twocol'><div><h3>Hespi regions for OCR</h3>" + table(hespi_regions) + "</div>"
            "<div><h3>YOLOE-26 regions for OCR</h3>" + table(yoloe_regions) + "</div></div></section>"
        )

    pd.DataFrame(csv_rows).to_csv(CSV_PATH, index=False)
    mean_iou = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0
    summary = [
        ("Images selected", len(images)), ("Images processed", len(images)),
        ("Hespi OCR-candidate boxes", hespi_total), ("YOLOE-26 OCR-candidate boxes", yoloe_total),
        ("Hespi boxes with YOLOE IoU >= 0.30", f"{sum(score >= 0.30 for score in overlap_scores)}/{len(overlap_scores)}"),
        ("Hespi boxes with YOLOE IoU >= 0.50", f"{sum(score >= 0.50 for score in overlap_scores)}/{len(overlap_scores)}"),
        ("Mean best YOLOE IoU for each Hespi OCR box", f"{mean_iou:.3f}"),
    ]
    summary_html = "<table><tbody>" + "".join(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>" for key, value in summary) + "</tbody></table>"
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>YOLOE-26 vs Hespi herbarium layout comparison</title>
<style>body{{font-family:Arial,sans-serif;max-width:1500px;margin:24px auto;padding:0 18px;line-height:1.45;color:#222}}section{{border-top:2px solid #ddd;margin-top:30px;padding-top:12px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.twocol{{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}}img{{width:100%;height:auto;border:1px solid #bbb}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top}}.note{{background:#fff7dc;padding:12px;border-left:5px solid #d09a00}}@media(max-width:900px){{.grid,.twocol{{grid-template-columns:1fr}}}}</style></head><body>
<h1>Herbarium layout detection: YOLOE-26 vs Hespi fine-tuned YOLOv8</h1>
<p><strong>Generated by executable GitHub Actions comparison.</strong> The same ten frozen Streamlit/Hespi v10 overview images are passed to both detectors. Red boxes are regions retained as candidates for the next OCR stage.</p>
<div class='note'><strong>Interpretation limit:</strong> this is a visual and proxy comparison, not a formal accuracy benchmark. The frozen image set does not contain human-reviewed bounding-box ground truth. The report does not claim mAP, recall, or precision. Hespi is a herbarium-specific fine-tuned detector. YOLOE-26s is tested in open-vocabulary prompt mode without herbarium fine-tuning.</div>
<h2>Methods</h2><p><strong>Hespi branch:</strong> Hespi sheet-component checkpoint, inference resolution 2048, confidence threshold 0.15.</p><p><strong>Latest-YOLO branch:</strong> yoloe-26s-seg.pt, open-vocabulary prompts, inference resolution 1024, confidence threshold 0.05.</p><p><strong>Frozen input bundle:</strong> {html.escape(str(bundle))}</p>
<h2>Summary proxy statistics</h2>{summary_html}<h2>Per-image visual comparison</h2>{''.join(cards)}
<h2>Source notes</h2><ul><li>Ultralytics YOLO26 documentation: https://docs.ultralytics.com/models/yolo26/</li><li>Ultralytics YOLOE documentation: https://docs.ultralytics.com/models/yoloe/</li><li>Hespi repository: https://github.com/rbturnbull/hespi</li></ul></body></html>"""
    HTML_PATH.write_text(document, encoding="utf-8")
    print({"html": str(HTML_PATH), "images_selected": len(images), "images_processed": len(images), "hespi_regions": hespi_total, "yoloe_regions": yoloe_total})


if __name__ == "__main__":
    main()
