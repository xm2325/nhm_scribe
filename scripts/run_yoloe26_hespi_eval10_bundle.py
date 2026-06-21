from __future__ import annotations

import html
import json
import re
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests

OUT = Path("reports/yoloe26_hespi_compare")
OUT.mkdir(parents=True, exist_ok=True)
IMAGE_DIR = OUT / "raw_images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
HTML_PATH = OUT / "yoloe26_vs_hespi_eval10.html"
CSV_PATH = OUT / "detections.csv"
SUMMARY_PATH = OUT / "per_image_summary.csv"
BUNDLE = Path("app_data/component_aware_eval10_review_bundle.zip")
EVAL_MEMBER = "component_aware_eval10/processed/eval_set.csv"
PROMPTS = ["primary specimen label", "specimen label", "annotation label", "database label", "barcode label", "barcode", "catalog number", "handwritten note", "printed label", "stamp", "scale bar", "color chart"]
HESPI_OCR_CLASSES = {"primary_specimen_label", "primary_label", "institutional_label", "handwritten_data", "annotation_label", "stamp", "swing_tag", "number", "small_database_label", "database_label", "full_database_label", "barcode"}
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
    url_column = "image_url" if "image_url" in frame.columns else next((c for c in url_candidates if "image" in c.lower()), None)
    if not url_column:
        raise RuntimeError(f"No image URL column found in eval_set.csv. Columns: {list(frame.columns)}")
    catalog_column = "catalogNumber" if "catalogNumber" in frame.columns else "occurrenceID"
    selected = frame[[catalog_column, url_column]].copy()
    selected.columns = ["catalog", "image_url"]
    selected = selected[selected["image_url"].astype(str).str.startswith("http")].drop_duplicates("catalog")
    if len(selected) < 10:
        raise RuntimeError(f"Expected at least 10 raw image URLs; found {len(selected)}")
    return selected.iloc[:10].reset_index(drop=True)


def download_images(frame: pd.DataFrame) -> list[dict[str, str]]:
    rows = []
    session = requests.Session()
    session.headers.update({"User-Agent": "nhm-scribe-yoloe-pruning-comparison/1.0"})
    for index, row in frame.iterrows():
        catalog = clean(row["catalog"])
        url = clean(row["image_url"])
        path = IMAGE_DIR / f"{index + 1:02d}_{safe_name(catalog)}.jpg"
        last_error = ""
        for attempt in range(4):
            try:
                response = session.get(url, timeout=120)
                response.raise_for_status()
                path.write_bytes(response.content)
                rows.append({"catalog": catalog, "image_url": url, "image_path": str(path)})
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
    rows = []
    for box in boxes:
        class_id = int(box.cls.cpu().item())
        confidence = float(box.conf.cpu().item())
        coords = box.xyxy.cpu().tolist()[0]
        label = clean(names.get(class_id, str(class_id))).lower().replace(" ", "_")
        rows.append({"label": label, "confidence": round(confidence, 5), "bbox": [int(round(float(v))) for v in coords[:4]]})
    return sorted(rows, key=lambda row: row["confidence"], reverse=True)


def area(row: dict[str, Any]) -> int:
    x0, y0, x1, y1 = row["bbox"]
    return max(0, x1 - x0) * max(0, y1 - y0)


def inter(a: dict[str, Any], b: dict[str, Any]) -> int:
    ax0, ay0, ax1, ay1 = a["bbox"]
    bx0, by0, bx1, by1 = b["bbox"]
    x0 = max(ax0, bx0); y0 = max(ay0, by0)
    x1 = min(ax1, bx1); y1 = min(ay1, by1)
    return max(0, x1 - x0) * max(0, y1 - y0)


def iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    intersection = inter(a, b)
    union = area(a) + area(b) - intersection
    return intersection / union if union else 0.0


def containment(child: dict[str, Any], parent: dict[str, Any]) -> float:
    child_area = area(child)
    return inter(child, parent) / child_area if child_area else 0.0


def nms(rows: list[dict[str, Any]], threshold: float = 0.55) -> list[dict[str, Any]]:
    kept = []
    for candidate in sorted(rows, key=lambda r: float(r["confidence"]), reverse=True):
        if any(iou(candidate, existing) >= threshold for existing in kept):
            continue
        kept.append(dict(candidate))
    return kept


def prune(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retained = []
    dropped = []
    for candidate in sorted(rows, key=area, reverse=True):
        candidate = dict(candidate)
        candidate_area = area(candidate)
        label = clean(candidate["label"])
        drop_reason = ""
        keep_reason = "parent_or_independent_region"
        for parent in retained:
            parent_area = area(parent)
            if parent_area <= candidate_area:
                continue
            cont = containment(candidate, parent)
            ratio = candidate_area / parent_area if parent_area else 0.0
            if cont >= 0.85 and ratio <= 0.35:
                if label in HIGH_VALUE_CHILD:
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
    return sorted(retained, key=lambda r: float(r["confidence"]), reverse=True), sorted(dropped, key=lambda r: float(r["confidence"]), reverse=True)


def match(hespi: list[dict[str, Any]], yoloe: list[dict[str, Any]], threshold: float = 0.30):
    candidates = []
    for hi, h in enumerate(hespi):
        for yi, y in enumerate(yoloe):
            score = iou(h, y)
            if score >= threshold:
                candidates.append((score, hi, yi))
    pairs = []
    used_h = set(); used_y = set()
    for score, hi, yi in sorted(candidates, reverse=True):
        if hi in used_h or yi in used_y:
            continue
        used_h.add(hi); used_y.add(yi)
        pairs.append({"iou": score, "hespi": hespi[hi], "yoloe": yoloe[yi]})
    return pairs, [h for i, h in enumerate(hespi) if i not in used_h], [y for i, y in enumerate(yoloe) if i not in used_y]


def small_table(rows: list[dict[str, Any]], reason: bool = False) -> str:
    if not rows:
        return "<p><em>None</em></p>"
    header = "<tr><th>#</th><th>region</th><th>conf</th><th>bbox</th>" + ("<th>reason</th>" if reason else "") + "</tr>"
    body = ""
    for idx, row in enumerate(rows, start=1):
        body += f"<tr><td>{idx}</td><td>{html.escape(str(row['label']))}</td><td>{float(row['confidence']):.3f}</td><td>{html.escape(str(row['bbox']))}</td>"
        if reason:
            body += f"<td>{html.escape(str(row.get('prune_reason','')))}</td>"
        body += "</tr>"
    return "<table><thead>" + header + "</thead><tbody>" + body + "</tbody></table>"


def write_report(summary: pd.DataFrame, detections: pd.DataFrame, cards: list[str]) -> None:
    summary.to_csv(SUMMARY_PATH, index=False)
    detections.to_csv(CSV_PATH, index=False)
    totals = {key: int(summary[key].sum()) for key in summary.columns if key != "catalog"}
    total_html = pd.DataFrame([totals]).to_html(index=False, escape=True)
    retained_only = detections[detections["analysis_status"].eq("yoloe_retained_only")]["region"].value_counts().rename_axis("region").reset_index(name="count")
    dropped = detections[detections["analysis_status"].eq("yoloe_dropped_child_inside_parent")]["region"].value_counts().rename_axis("region").reset_index(name="count")
    html_text = f"""<!doctype html><html><head><meta charset='utf-8'><title>YOLOE-26 pruning vs Hespi</title><style>body{{font-family:Arial,sans-serif;max-width:1700px;margin:20px auto;line-height:1.42}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #ccc;padding:5px;text-align:left;vertical-align:top}}section{{border-top:2px solid #ddd;margin-top:28px;padding-top:10px}}.note{{background:#eef6ff;border-left:5px solid #2374c6;padding:12px}}.cols{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}}@media(max-width:1000px){{.cols{{grid-template-columns:1fr}}}}</style></head><body><h1>YOLOE-26 containment-aware OCR candidate pruning vs Hespi</h1><div class='note'>YOLOE-26 is used as a high-recall candidate detector. After NMS, a smaller child box is dropped if containment is at least 0.85 and area ratio is at most 0.35, unless it is high-value: barcode, barcode label, catalog number, number, or small database label.</div><h2>Total counts</h2>{total_html}<h2>Per-image summary</h2>{summary.to_html(index=False, escape=True)}<h2>YOLOE retained-only region types</h2>{retained_only.to_html(index=False, escape=True) if len(retained_only) else '<p><em>None</em></p>'}<h2>Dropped child region types</h2>{dropped.to_html(index=False, escape=True) if len(dropped) else '<p><em>None</em></p>'}<h2>Per-image comparison</h2>{''.join(cards)}</body></html>"""
    HTML_PATH.write_text(html_text, encoding="utf-8")


def main() -> None:
    try:
        raw_images = download_images(load_eval10())
        from hespi.hespi import Hespi
        from ultralytics import YOLO
        hespi_model = Hespi(gpu=False, htr=False, fuzzy=False, llm_model="none", force_download=False, sheet_component_res=2048, label_field_res=1280).sheet_component_model
        yoloe_model = YOLO("yoloe-26s-seg.pt")
        yoloe_model.set_classes(PROMPTS)
        all_rows = []
        summaries = []
        cards = []
        for item in raw_images:
            catalog = item["catalog"]
            hespi_result = hespi_model.predict(source=[item["image_path"]], show=False, save=False, batch=1, imgsz=2048, conf=0.15, verbose=False)[0]
            yoloe_result = yoloe_model.predict(source=[item["image_path"]], show=False, save=False, batch=1, imgsz=1024, conf=0.05, verbose=False)[0]
            hespi = [r for r in box_rows(hespi_result) if r["label"] in HESPI_OCR_CLASSES]
            for r in hespi:
                r["label"] = r["label"].replace("_", " ")
            raw = []
            for r in box_rows(yoloe_result):
                r["label"] = r["label"].replace("_", " ")
                if r["label"] not in YOLOE_EXCLUDE:
                    raw.append(r)
            after_nms = nms(raw)
            retained, dropped = prune(after_nms)
            pairs, hespi_only, retained_only = match(hespi, retained)
            summaries.append({"catalog": catalog, "hespi_regions": len(hespi), "yoloe_raw": len(raw), "yoloe_after_nms": len(after_nms), "yoloe_retained_queue": len(retained), "yoloe_dropped_children": len(dropped), "matched_with_hespi": len(pairs), "hespi_only_vs_retained": len(hespi_only), "yoloe_retained_only": len(retained_only)})
            for status, rows in [("hespi_baseline", hespi), ("yoloe_after_nms", after_nms), ("yoloe_retained_for_ocr_queue", retained), ("yoloe_dropped_child_inside_parent", dropped), ("yoloe_retained_only", retained_only), ("hespi_only_vs_retained", hespi_only)]:
                for row in rows:
                    all_rows.append({"catalog": catalog, "analysis_status": status, "region": row["label"], "confidence": row["confidence"], "bbox_xyxy": json.dumps(row["bbox"]), "prune_reason": row.get("prune_reason", ""), "source_image_url": item["image_url"]})
            cards.append(f"<section><h2>{html.escape(catalog)}</h2><p><a href='{html.escape(item['image_url'])}'>raw image</a></p><p><b>Counts:</b> Hespi {len(hespi)}; YOLOE after NMS {len(after_nms)}; retained {len(retained)}; dropped {len(dropped)}; matched {len(pairs)}; YOLOE retained-only {len(retained_only)}; Hespi-only {len(hespi_only)}.</p><div class='cols'><div><h3>YOLOE retained-only vs Hespi</h3>{small_table(retained_only, True)}</div><div><h3>Dropped child boxes</h3>{small_table(dropped, True)}</div><div><h3>Hespi-only after pruning</h3>{small_table(hespi_only)}</div></div></section>")
        write_report(pd.DataFrame(summaries), pd.DataFrame(all_rows), cards)
        print(pd.DataFrame(summaries).sum(numeric_only=True).to_json())
    except Exception:
        error = traceback.format_exc()
        pd.DataFrame().to_csv(CSV_PATH, index=False)
        pd.DataFrame().to_csv(SUMMARY_PATH, index=False)
        HTML_PATH.write_text("<html><body><h1>Pruning report failed</h1><pre>" + html.escape(error) + "</pre></body></html>", encoding="utf-8")
        print(error)
        raise


if __name__ == "__main__":
    main()
