from __future__ import annotations

import base64
import html
import io
import json
import re
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from PIL import Image, ImageDraw

OUT = Path("reports/yoloe26_hespi_compare")
OUT.mkdir(parents=True, exist_ok=True)
IMG_DIR = OUT / "downloaded_images"
IMG_DIR.mkdir(parents=True, exist_ok=True)
HTML_PATH = OUT / "yoloe26_vs_hespi_eval10.html"
CSV_URL = "https://zenodo.org/records/6372393/files/Data%20and%20Links%20excl%20extensions.csv?download=1"
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


def download_bytes(url: str, timeout: int = 120) -> bytes:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "nhm-scribe-yoloe26-comparison/1.0"})
    response.raise_for_status()
    return response.content


def read_metadata() -> pd.DataFrame:
    raw = download_bytes(CSV_URL)
    for encoding in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding, low_memory=False).fillna("")
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw), low_memory=False).fillna("")


def find_image_url_column(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    ranked = []
    for col in cols:
        key = norm(col)
        score = 0
        if "jpegurl" in key or "jpgurl" in key:
            score += 100
        if "jpeg" in key or "jpg" in key:
            score += 20
        if "image" in key:
            score += 10
        if "url" in key or "link" in key:
            score += 5
        ranked.append((score, col))
    for _, col in sorted(ranked, reverse=True):
        values = frame[col].astype(str)
        if values.str.startswith("http").any():
            return col
    raise RuntimeError(f"Could not find an image URL column in: {cols}")


def find_catalog_column(frame: pd.DataFrame) -> str:
    priorities = ["catalognumber", "cataloguenumber", "barcode", "occurrenceid", "identifier", "id"]
    by_norm = {norm(col): col for col in frame.columns}
    for key in priorities:
        if key in by_norm:
            return by_norm[key]
    for col in frame.columns:
        key = norm(col)
        if "catalog" in key or "barcode" in key or "occurrence" in key:
            return col
    return str(frame.columns[0])


def pick_eval_rows(frame: pd.DataFrame, n: int = 10) -> tuple[pd.DataFrame, str, str, str]:
    url_col = find_image_url_column(frame)
    catalog_col = find_catalog_column(frame)
    candidates = frame[frame[url_col].astype(str).str.startswith("http")].copy()
    candidates["__catalog"] = candidates[catalog_col].astype(str)
    candidates["__catalog_norm"] = candidates["__catalog"].map(norm)
    refs: list[str] = []
    review_path = Path("app_data/gpt_primary_label_reviews_eval10.json")
    if review_path.exists():
        data = json.loads(review_path.read_text(encoding="utf-8"))
        refs = [clean(row.get("display_identifier") or row.get("catalog_reference")) for row in data.get("records", [])]
    selected_indices: list[int] = []
    matched_refs = 0
    for ref in refs:
        target = norm(ref)
        if not target:
            continue
        exact = candidates[candidates["__catalog_norm"] == target]
        if exact.empty:
            exact = candidates[candidates["__catalog_norm"].map(lambda value: bool(value) and (target in value or value in target))]
        if not exact.empty:
            idx = int(exact.index[0])
            if idx not in selected_indices:
                selected_indices.append(idx)
                matched_refs += 1
    remaining = candidates.drop(index=selected_indices, errors="ignore")
    if len(selected_indices) < n:
        fill = remaining.sample(n=min(n - len(selected_indices), len(remaining)), random_state=42)
        selected_indices.extend(int(index) for index in fill.index)
    selected = candidates.loc[selected_indices[:n]].reset_index(drop=True)
    selection_note = f"Matched {matched_refs} Streamlit eval10 catalogue references; filled {max(0, len(selected) - matched_refs)} records by deterministic sampling."
    return selected, url_col, catalog_col, selection_note


def box_rows(result: Any, name_map: dict[int, str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = name_map or getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return rows
    for box in boxes:
        cls_value = box.cls.cpu().item() if hasattr(box.cls, "cpu") else box.cls
        conf_value = box.conf.cpu().item() if hasattr(box.conf, "cpu") else box.conf
        coords = box.xyxy.cpu().tolist()[0] if hasattr(box.xyxy, "cpu") else box.xyxy.tolist()[0]
        class_id = int(cls_value)
        label = clean(names.get(class_id, str(class_id))).lower().replace(" ", "_")
        rows.append({
            "label": label,
            "confidence": round(float(conf_value), 5),
            "bbox": [int(round(float(value))) for value in coords[:4]],
        })
    return sorted(rows, key=lambda row: row["confidence"], reverse=True)


def yoloe_box_rows(result: Any) -> list[dict[str, Any]]:
    rows = box_rows(result)
    for row in rows:
        row["label"] = row["label"].replace("_", " ")
    return rows


def iou(left: list[int], right: list[int]) -> float:
    x0 = max(left[0], right[0]); y0 = max(left[1], right[1])
    x1 = min(left[2], right[2]); y1 = min(left[3], right[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    la = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    ra = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = la + ra - inter
    return inter / union if union else 0.0


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


def as_data_uri(image: Image.Image) -> str:
    preview = image.copy().convert("RGB")
    preview.thumbnail((900, 1200))
    buf = io.BytesIO()
    preview.save(buf, format="JPEG", quality=80, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def rows_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p><em>No OCR-candidate region detected.</em></p>"
    body = []
    for index, row in enumerate(rows, start=1):
        body.append(
            "<tr>"
            f"<td>{index}</td><td>{html.escape(clean(row.get('label')))}</td>"
            f"<td>{float(row.get('confidence', 0)):.3f}</td>"
            f"<td>{html.escape(str(row.get('bbox', [])))}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>#</th><th>region</th><th>confidence</th><th>bbox xyxy</th></tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def write_failure_report(message: str) -> None:
    HTML_PATH.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>YOLOE-26 vs Hespi</title></head>"
        "<body><h1>YOLOE-26 vs Hespi comparison</h1>"
        "<p>The executable comparison could not complete. The captured error is shown below.</p>"
        f"<pre>{html.escape(message)}</pre></body></html>", encoding="utf-8"
    )


def main() -> None:
    records: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    frame = read_metadata()
    selected, url_col, catalog_col, selection_note = pick_eval_rows(frame, n=10)

    hespi_error = ""
    yoloe_error = ""
    hespi_model = None
    yoloe_model = None
    try:
        from hespi.hespi import Hespi
        hespi = Hespi(gpu=False, htr=False, fuzzy=False, llm_model="none", force_download=False, sheet_component_res=2048, label_field_res=1280)
        hespi_model = hespi.sheet_component_model
    except Exception:
        hespi_error = traceback.format_exc()
    try:
        from ultralytics import YOLO
        yoloe_model = YOLO("yoloe-26s-seg.pt")
        yoloe_model.set_classes(PROMPTS)
    except Exception:
        yoloe_error = traceback.format_exc()

    for index, row in selected.iterrows():
        catalog = clean(row.get(catalog_col)) or f"record_{index + 1:02d}"
        image_url = clean(row.get(url_col))
        target = IMG_DIR / f"{index + 1:02d}_{safe_name(catalog)}.jpg"
        item: dict[str, Any] = {"catalog": catalog, "url": image_url, "error": ""}
        try:
            target.write_bytes(download_bytes(image_url))
            image = Image.open(target).convert("RGB")
            hespi_boxes: list[dict[str, Any]] = []
            yoloe_boxes: list[dict[str, Any]] = []
            if hespi_model is not None:
                result = hespi_model.predict(source=[str(target)], show=False, save=False, batch=1, imgsz=2048, conf=0.15, verbose=False)[0]
                hespi_boxes = box_rows(result)
            if yoloe_model is not None:
                result = yoloe_model.predict(source=[str(target)], show=False, save=False, batch=1, imgsz=1024, conf=0.05, verbose=False)[0]
                yoloe_boxes = yoloe_box_rows(result)
            hespi_ocr = [box for box in hespi_boxes if box["label"] in HESPI_OCR_CLASSES]
            yoloe_ocr = [box for box in yoloe_boxes if box["label"] not in YOLOE_EXCLUDE]
            overlaps = [max([iou(h["bbox"], y["bbox"]) for y in yoloe_ocr] or [0.0]) for h in hespi_ocr]
            item.update({
                "image": image,
                "hespi_boxes": hespi_boxes,
                "hespi_ocr": hespi_ocr,
                "yoloe_boxes": yoloe_boxes,
                "yoloe_ocr": yoloe_ocr,
                "overlaps": overlaps,
            })
            for detector, boxes in (("hespi_yolov8_finetuned", hespi_ocr), ("yoloe_26s_open_vocab", yoloe_ocr)):
                for box in boxes:
                    detection_rows.append({
                        "catalog": catalog, "detector": detector, "region": box["label"],
                        "confidence": box["confidence"], "bbox_xyxy": json.dumps(box["bbox"]), "image_url": image_url,
                    })
        except Exception:
            item["error"] = traceback.format_exc()
        records.append(item)

    pd.DataFrame(detection_rows).to_csv(OUT / "detections.csv", index=False)
    processed = [record for record in records if "image" in record]
    hespi_regions = sum(len(record.get("hespi_ocr", [])) for record in processed)
    yoloe_regions = sum(len(record.get("yoloe_ocr", [])) for record in processed)
    all_overlaps = [score for record in processed for score in record.get("overlaps", [])]
    matched_03 = sum(score >= 0.30 for score in all_overlaps)
    matched_05 = sum(score >= 0.50 for score in all_overlaps)
    cards: list[str] = []
    for record in records:
        if "image" not in record:
            cards.append(f"<section><h2>{html.escape(record['catalog'])}</h2><pre>{html.escape(record.get('error', ''))}</pre></section>")
            continue
        image = record["image"]
        original_uri = as_data_uri(image)
        hespi_uri = as_data_uri(draw_boxes(image, record["hespi_ocr"], "Hespi fine-tuned YOLOv8: OCR candidate regions"))
        yoloe_uri = as_data_uri(draw_boxes(image, record["yoloe_ocr"], "YOLOE-26s open-vocabulary: OCR candidate regions"))
        overlaps = record.get("overlaps", [])
        overlap_note = "No Hespi OCR region for overlap proxy." if not overlaps else f"Hespi regions with YOLOE IoU >= 0.30: {sum(score >= 0.30 for score in overlaps)}/{len(overlaps)}; mean best IoU: {sum(overlaps) / len(overlaps):.3f}."
        cards.append(
            f"<section><h2>{html.escape(record['catalog'])}</h2>"
            f"<p><a href='{html.escape(record['url'])}'>Open source image</a></p>"
            "<div class='grid'>"
            f"<figure><img src='{original_uri}'><figcaption>Original</figcaption></figure>"
            f"<figure><img src='{hespi_uri}'><figcaption>Hespi fine-tuned YOLOv8</figcaption></figure>"
            f"<figure><img src='{yoloe_uri}'><figcaption>YOLOE-26s open-vocabulary prompts</figcaption></figure>"
            "</div>"
            f"<p>{html.escape(overlap_note)}</p>"
            "<div class='twocol'><div><h3>Hespi regions for OCR</h3>" + rows_table(record["hespi_ocr"]) + "</div>"
            "<div><h3>YOLOE-26 regions for OCR</h3>" + rows_table(record["yoloe_ocr"]) + "</div></div></section>"
        )
    summary_rows = [
        ("Images selected", len(selected)), ("Images processed", len(processed)),
        ("Hespi OCR-candidate boxes", hespi_regions), ("YOLOE-26 OCR-candidate boxes", yoloe_regions),
        ("Hespi boxes with YOLOE IoU >= 0.30", f"{matched_03}/{len(all_overlaps)}"),
        ("Hespi boxes with YOLOE IoU >= 0.50", f"{matched_05}/{len(all_overlaps)}"),
        ("Mean best YOLOE IoU for each Hespi OCR box", f"{(sum(all_overlaps) / len(all_overlaps)) if all_overlaps else 0.0:.3f}"),
    ]
    summary_table = "<table><tbody>" + "".join(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>" for key, value in summary_rows) + "</tbody></table>"
    model_errors = ""
    if hespi_error:
        model_errors += "<h3>Hespi load error</h3><pre>" + html.escape(hespi_error) + "</pre>"
    if yoloe_error:
        model_errors += "<h3>YOLOE-26 load error</h3><pre>" + html.escape(yoloe_error) + "</pre>"
    document = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>YOLOE-26 vs Hespi herbarium layout comparison</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1500px;margin:24px auto;padding:0 18px;line-height:1.45;color:#222}}
section{{border-top:2px solid #ddd;margin-top:30px;padding-top:12px}} .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
.twocol{{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}} img{{width:100%;height:auto;border:1px solid #bbb}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top}} pre{{white-space:pre-wrap;background:#f6f6f6;padding:10px}}
.note{{background:#fff7dc;padding:12px;border-left:5px solid #d09a00}} code{{background:#f1f1f1;padding:1px 3px}}
@media(max-width:900px){{.grid,.twocol{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Herbarium layout detection: YOLOE-26 vs Hespi fine-tuned YOLOv8</h1>
<p><strong>Generated by executable GitHub Actions comparison.</strong> The same ten selected public herbarium images are passed to both detectors. Red boxes are regions retained as candidates for the next OCR stage.</p>
<div class='note'><strong>Interpretation limit:</strong> this is a visual and proxy comparison, not a formal accuracy benchmark. The selected image set does not include human-reviewed bounding-box ground truth. Therefore, the report does not claim mAP, recall, or precision. Hespi is a herbarium-specific fine-tuned detector. YOLOE-26s is tested in open-vocabulary prompt mode without herbarium fine-tuning.</div>
<h2>Methods</h2>
<p><strong>Hespi branch:</strong> Hespi sheet-component checkpoint, loaded by Hespi, inference resolution 2048, confidence threshold 0.15. OCR candidates retain text-bearing component classes such as primary specimen label, annotation label, handwritten data, database label, number, barcode, and stamp.</p>
<p><strong>Latest-YOLO branch:</strong> <code>yoloe-26s-seg.pt</code>, open-vocabulary prompts, inference resolution 1024, confidence threshold 0.05. Prompts include primary specimen label, specimen label, annotation label, database label, barcode, catalog number, handwritten note, printed label, stamp, scale bar, and color chart. Scale bar and color chart are excluded from OCR candidates.</p>
<p><strong>Image source:</strong> Zenodo record 6372393 CSV with direct JPEG links. {html.escape(selection_note)}</p>
<h2>Summary proxy statistics</h2>{summary_table}
{model_errors}
<h2>Per-image visual comparison</h2>{''.join(cards)}
<h2>Source notes</h2>
<ul><li>Ultralytics YOLO26 documentation: https://docs.ultralytics.com/models/yolo26/</li><li>Ultralytics YOLOE documentation: https://docs.ultralytics.com/models/yoloe/</li><li>Hespi repository: https://github.com/rbturnbull/hespi</li><li>Public herbarium link dataset: https://zenodo.org/records/6372393</li></ul>
</body></html>"""
    HTML_PATH.write_text(document, encoding="utf-8")
    print({"html": str(HTML_PATH), "images_selected": len(selected), "images_processed": len(processed), "hespi_regions": hespi_regions, "yoloe_regions": yoloe_regions})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        write_failure_report(error)
        print(error)
        raise
