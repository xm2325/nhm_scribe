from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from herbarium_scribe.extract_rules import extract_rule_based
from herbarium_scribe.llm_backends import call_llm_with_metadata
from herbarium_scribe.ocr import ocr_image_tesseract
from herbarium_scribe.pipeline import _parse_llm_json
from herbarium_scribe.rag import build_rag_corpus, format_context_for_prompt, retrieve_context
from herbarium_scribe.schema import EXTRACTION_FIELDS, flatten_record, validate_record

APP_BUNDLE = ROOT / "app_data" / "real_eval_100_streamlit_bundle.zip"
HESPI_V10_REPORT = ROOT / "app_data" / "hespi_v10_ocr_visual_report.zip"
HESPI_REPORT_DIR = "hespi_v10_ocr_visual_report"
QWEN_VISION_REPORT = ROOT / "app_data" / "hespi_v11_qwen_streamlit_bundle.zip"
QWEN_VISION_DIR = "hespi_v11_qwen_streamlit_bundle"
LOCAL_PROCESSED = ROOT / "data" / "processed"
LOCAL_LLM = ROOT / "data" / "interim" / "llm"
THUMB_PREFIX = "app_data/thumbnails/real_eval_100"


st.set_page_config(page_title="Herbarium SCRIBE", layout="wide")


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


# -----------------------------------------------------------------------------
# Hespi v10 report home page
# -----------------------------------------------------------------------------

def hespi_normalise_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


@st.cache_data(show_spinner=False)
def hespi_zip_names(path: str) -> list[str]:
    if not Path(path).exists():
        return []
    with zipfile.ZipFile(path) as zf:
        return zf.namelist()


@st.cache_data(show_spinner=False)
def hespi_read_csv(path: str, filename: str) -> pd.DataFrame:
    member = f"{HESPI_REPORT_DIR}/{filename}"
    with zipfile.ZipFile(path) as zf:
        with zf.open(member) as fh:
            return pd.read_csv(fh, dtype=str).fillna("")


@st.cache_data(show_spinner=False)
def hespi_read_bytes(path: str, member: str) -> bytes:
    with zipfile.ZipFile(path) as zf:
        return zf.read(member)


def hespi_find_overview_member(path: str, catalog_number: str) -> str:
    target = hespi_normalise_identifier(catalog_number)
    for member in hespi_zip_names(path):
        if "/assets/overviews/" not in member or not member.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = Path(member).stem
        if stem.endswith("_00"):
            stem = stem[:-3]
        if hespi_normalise_identifier(stem) == target:
            return member
    return ""


def hespi_crop_members(path: str, catalog_number: str) -> list[str]:
    target = hespi_normalise_identifier(catalog_number)
    members: list[str] = []
    for member in hespi_zip_names(path):
        if "/assets/crops/" not in member or not member.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = Path(member).stem
        if hespi_normalise_identifier(stem).startswith(target):
            members.append(member)
    return sorted(members)


def hespi_render_image(
    path: str,
    member: str,
    caption: str | None = None,
    max_width: int | None = None,
) -> tuple[bytes, tuple[int, int]]:
    """Render a report image without enlarging it beyond its native resolution.

    Stretching a small OCR crop or overview across a wide Streamlit column makes
    the image look blurred.  This helper caps the displayed width at the native
    width while still allowing large overview images to be reduced to a useful
    browser size.
    """
    image_bytes = hespi_read_bytes(path, member)
    image = Image.open(io.BytesIO(image_bytes))
    native_width, native_height = image.size
    display_width = native_width if max_width is None else min(native_width, max_width)
    st.image(image, caption=caption, width=display_width)
    return image_bytes, (native_width, native_height)


def hespi_record_order(detail: pd.DataFrame, path: str) -> list[str]:
    if detail.empty or "catalogNumber" not in detail.columns:
        return []
    records = [clean(value) for value in detail["catalogNumber"].drop_duplicates().tolist() if clean(value)]
    return sorted(records, key=lambda catalog: (not bool(hespi_find_overview_member(path, catalog)), catalog.lower()))


def hespi_display_summary_tables(path: str) -> None:
    field_metrics = hespi_read_csv(path, "field_metrics.csv")
    htr_summary = hespi_read_csv(path, "htr_engine_summary.csv")

    st.subheader("Evaluation summary")
    st.caption("Field-level extraction metrics for the ten-record Hespi v10 evaluation.")
    st.dataframe(field_metrics, use_container_width=True, hide_index=True)

    st.subheader("OCR / HTR engine comparison")
    st.caption("Component-level match rate and non-empty output rate for collector and date regions.")
    st.dataframe(htr_summary, use_container_width=True, hide_index=True)


def hespi_display_record(path: str, catalog_number: str, detail: pd.DataFrame) -> None:
    rows = detail[detail["catalogNumber"].astype(str) == catalog_number].copy()
    overview = hespi_find_overview_member(path, catalog_number)
    crops = hespi_crop_members(path, catalog_number)

    with st.container(border=True):
        st.subheader(catalog_number)
        occurrence_ids = [clean(value) for value in rows["occurrenceID"].drop_duplicates().tolist() if clean(value)]
        if occurrence_ids:
            st.caption(occurrence_ids[0])

        if overview:
            overview_bytes, (native_width, native_height) = hespi_render_image(
                path, overview, "Annotated overview", max_width=1200
            )
            st.caption(
                f"Displayed at no more than native resolution. "
                f"Annotated overview size: {native_width} × {native_height} pixels."
            )
            st.download_button(
                "Download full-resolution annotated overview",
                data=overview_bytes,
                file_name=Path(overview).name,
                mime="image/jpeg",
                key=f"download-overview-{hespi_normalise_identifier(catalog_number)}",
            )
        else:
            st.info("No annotated overview thumbnail is available for this record.")

        st.markdown("**OCR / HTR and extraction detail**")
        columns = [
            "region_type",
            "ocr_engine",
            "ocr_text",
            "ocr_status",
            "htr_prompt_accepted",
            "htr_prompt_reason",
            "gold_recordedBy",
            "gold_eventDate",
            "final_recordedBy",
            "final_eventDate",
            "final_catalogNumber",
        ]
        visible = [column for column in columns if column in rows.columns]
        st.dataframe(rows[visible], use_container_width=True, hide_index=True)

        if crops:
            with st.expander(f"OCR-focused crop images ({len(crops)})", expanded=False):
                st.caption("Crop images are shown without enlargement to preserve readable edges and handwriting strokes.")
                for start in range(0, len(crops), 3):
                    cols = st.columns(3)
                    for col, member in zip(cols, crops[start : start + 3]):
                        with col:
                            hespi_render_image(path, member, Path(member).name, max_width=360)


def show_hespi_v10_home(report_zip: Path) -> None:
    st.title("Herbarium SCRIBE")
    st.caption(
        "Hespi v10 ten-record OCR visual evaluation: annotated regions, OCR-focused crops, "
        "OCR / HTR outputs, and final extraction review."
    )

    if not report_zip.exists():
        st.error(f"Missing report bundle: {report_zip}")
        st.info("Upload app_data/hespi_v10_ocr_visual_report.zip to display the Hespi v10 results.")
        return

    detail = hespi_read_csv(str(report_zip), "ocr_focus_detail.csv")
    records = hespi_record_order(detail, str(report_zip))
    with_thumbnail = sum(bool(hespi_find_overview_member(str(report_zip), record)) for record in records)

    st.header("Record-level visual review")
    st.caption(
        "Records with annotated overview images are shown first. "
        "The record without an overview image is placed at the end. "
        "Images are not enlarged beyond their native resolution."
    )
    for record in records:
        hespi_display_record(str(report_zip), record, detail)

    st.divider()
    st.header("Herbarium SCRIBE overall results")
    cols = st.columns(4)
    cols[0].metric("EVAL records", len(records))
    cols[1].metric("Annotated overviews", with_thumbnail)
    cols[2].metric("Records without overview", len(records) - with_thumbnail)
    cols[3].metric("OCR detail rows", len(detail))

    contact_sheet = f"{HESPI_REPORT_DIR}/contact_sheet.jpg"
    if contact_sheet in hespi_zip_names(str(report_zip)):
        with st.expander("Contact sheet", expanded=True):
            hespi_render_image(str(report_zip), contact_sheet, "Hespi v10 OCR visual report", max_width=780)

    hespi_display_summary_tables(str(report_zip))


# -----------------------------------------------------------------------------
# Qwen primary-label vision comparison
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def qwen_read_csv(path: str, filename: str) -> pd.DataFrame:
    member = f"{QWEN_VISION_DIR}/{filename}"
    with zipfile.ZipFile(path) as zf:
        with zf.open(member) as handle:
            return pd.read_csv(handle, dtype=str).fillna("")


@st.cache_data(show_spinner=False)
def qwen_read_jsonl(path: str, filename: str) -> list[dict[str, Any]]:
    member = f"{QWEN_VISION_DIR}/{filename}"
    with zipfile.ZipFile(path) as zf:
        text = zf.read(member).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def qwen_output_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {clean(row.get("occurrenceID")): row for row in rows}


def qwen_normalized_contains(needle: str, haystack: str) -> bool:
    normalized_needle = re.sub(r"[^a-z0-9]+", "", clean(needle).lower())
    normalized_haystack = re.sub(r"[^a-z0-9]+", "", clean(haystack).lower())
    return bool(normalized_needle) and normalized_needle in normalized_haystack


def qwen_field_table(output: dict[str, Any], detail: pd.DataFrame) -> pd.DataFrame:
    parsed = output.get("parsed_json") or {}
    fields = parsed.get("fields", {}) if isinstance(parsed, dict) else {}
    transcription = clean(parsed.get("full_transcription")) if isinstance(parsed, dict) else ""
    rows = []
    for field in EXTRACTION_FIELDS:
        item = fields.get(field, {}) if isinstance(fields, dict) else {}
        if not isinstance(item, dict):
            item = {"value": item}
        match = detail[detail["field"] == field] if not detail.empty else pd.DataFrame()
        value = clean(item.get("value"))
        evidence = clean(item.get("evidence_span"))
        confidence = clean(item.get("confidence"))
        evidence_found = not value or qwen_normalized_contains(evidence or value, transcription)
        rows.append({
            "field": field,
            "gold": clean(match.iloc[0].get("gold")) if not match.empty else "",
            "Qwen value": value,
            "confidence": confidence,
            "evidence": evidence,
            "evidence in transcription": evidence_found,
            "exact match": clean(match.iloc[0].get("exact_match")) if not match.empty else "",
            "token F1": clean(match.iloc[0].get("token_f1")) if not match.empty else "",
        })
    return pd.DataFrame(rows)


def show_qwen_vision_comparison(report_zip: Path) -> None:
    st.title("Qwen Primary-Label Vision")
    st.caption(
        "Automatically cropped primary labels sent directly to Qwen-VL. "
        "The complete visual transcription is preserved before field extraction."
    )
    if not report_zip.exists():
        st.info(
            "The Qwen eval10 workflow is ready, but its result bundle has not been added yet. "
            "Run “Hespi v11 eval10 Qwen primary-label vision” and add the generated "
            "hespi_v11_qwen_streamlit_bundle.zip to app_data."
        )
        return

    manifest = qwen_read_csv(str(report_zip), "primary_label_manifest.csv")
    detail = qwen_read_csv(str(report_zip), "qwen_primary_label_evaluation_detail.csv")
    summary = qwen_read_csv(str(report_zip), "qwen_primary_label_evaluation_summary.csv")
    outputs = qwen_read_jsonl(str(report_zip), "qwen_vision_outputs.jsonl")
    output_by_id = qwen_output_index(outputs)
    record_ids = [clean(value) for value in manifest["occurrenceID"].drop_duplicates() if clean(value)]
    parsed_count = sum(row.get("status") == "parsed" for row in outputs)
    evidence_risks = 0
    for occurrence_id in record_ids:
        output = output_by_id.get(occurrence_id, {})
        record_detail = detail[detail["occurrenceID"] == occurrence_id]
        table = qwen_field_table(output, record_detail)
        evidence_risks += int((table["Qwen value"].ne("") & ~table["evidence in transcription"]).sum())

    metrics = st.columns(4)
    metrics[0].metric("Records with primary label", len(record_ids))
    metrics[1].metric("Parsed Qwen outputs", parsed_count)
    metrics[2].metric("Evidence mismatches", evidence_risks)
    actual_models = sorted({
        clean(row.get("actual_model"))
        for row in outputs
        if clean(row.get("actual_model"))
    })
    metrics[3].metric("Actual model", ", ".join(actual_models) or "unknown")

    st.subheader("Field-level comparison")
    st.caption(
        "The two evidence-proxy columns measure whether gold values appear in a transcription; "
        "they are not OCR CER/WER."
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    selected = st.selectbox(
        "Record",
        record_ids,
        format_func=lambda occurrence_id: (
            clean(manifest[manifest["occurrenceID"] == occurrence_id].iloc[0].get("catalogNumber"))
            or occurrence_id
        ),
    )
    crop_rows = manifest[manifest["occurrenceID"] == selected].sort_values("image_index")
    output = output_by_id.get(selected, {})
    parsed = output.get("parsed_json") or {}
    record_detail = detail[detail["occurrenceID"] == selected]

    st.divider()
    image_col, transcript_col = st.columns([0.9, 1.1])
    with image_col:
        st.subheader("1. Primary-label crop")
        with zipfile.ZipFile(report_zip) as zf:
            for _, crop in crop_rows.iterrows():
                member = f"{QWEN_VISION_DIR}/{clean(crop.get('bundle_crop_path'))}"
                image_bytes = zf.read(member)
                image = Image.open(io.BytesIO(image_bytes))
                st.image(
                    image,
                    caption=f"Crop {clean(crop.get('image_index'))} · {image.width} × {image.height}",
                    use_container_width=True,
                )
                st.caption(
                    f"Detection confidence: {clean(crop.get('layout_confidence')) or 'unknown'}"
                )
    with transcript_col:
        st.subheader("2. Complete visual transcription")
        if output.get("error_message"):
            st.warning(clean(output.get("error_message")))
        transcription = clean(parsed.get("full_transcription")) if isinstance(parsed, dict) else ""
        st.text_area(
            "Qwen transcription",
            transcription,
            height=310,
            label_visibility="collapsed",
        )
        transcriptions = parsed.get("transcriptions", []) if isinstance(parsed, dict) else []
        uncertain = [
            clean(span)
            for item in transcriptions
            if isinstance(item, dict)
            for span in item.get("uncertain_spans", [])
            if clean(span)
        ]
        if uncertain:
            st.warning("Uncertain visual readings: " + " · ".join(uncertain))
        with st.expander("Tesseract on the same crop", expanded=True):
            for _, crop in crop_rows.iterrows():
                st.markdown(f"**Crop {clean(crop.get('image_index'))}**")
                st.code(clean(crop.get("tesseract_text")) or "(empty)", language="text")

    st.subheader("3. Direct multimodal field extraction")
    field_table = qwen_field_table(output, record_detail)
    risky = field_table[
        field_table["Qwen value"].ne("")
        & ~field_table["evidence in transcription"]
    ]
    if len(risky):
        st.error(
            "Potential extraction bug: one or more values are not present in Qwen's own transcription. "
            "Treat them as unsupported."
        )
    st.dataframe(field_table, use_container_width=True, hide_index=True)

    with st.expander("Raw Qwen response and diagnostics", expanded=False):
        diagnostic_cols = st.columns(4)
        diagnostic_cols[0].metric("Status", clean(output.get("status")) or "missing")
        diagnostic_cols[1].metric("Raw chars", clean(output.get("raw_output_length")) or "0")
        diagnostic_cols[2].metric("Finish reason", clean(output.get("finish_reason")) or "unknown")
        diagnostic_cols[3].metric(
            "Tokens",
            clean((output.get("usage") or {}).get("total_tokens")) or "unknown",
        )
        st.code(clean(output.get("raw_output")), language="json")


# -----------------------------------------------------------------------------
# Existing 100-record evaluation browser
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def zip_names(path: str) -> list[str]:
    if not Path(path).exists():
        return []
    with zipfile.ZipFile(path) as zf:
        return zf.namelist()


@st.cache_data(show_spinner=False)
def read_csv_from_zip(path: str, name: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        with zf.open(name) as fh:
            return pd.read_csv(fh, dtype=str).fillna("")


@st.cache_data(show_spinner=False)
def read_text_from_zip(path: str, name: str) -> str:
    with zipfile.ZipFile(path) as zf:
        return zf.read(name).decode("utf-8")


@st.cache_data(show_spinner=False)
def read_jsonl_from_zip(path: str, name: str) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        text = zf.read(name).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_local_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("") if path.exists() else pd.DataFrame()


def read_local_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_bundle_csv(name: str) -> pd.DataFrame:
    if APP_BUNDLE.exists() and name in zip_names(str(APP_BUNDLE)):
        return read_csv_from_zip(str(APP_BUNDLE), name)
    return read_local_csv(ROOT / name)


def read_bundle_text(name: str) -> str:
    if APP_BUNDLE.exists() and name in zip_names(str(APP_BUNDLE)):
        return read_text_from_zip(str(APP_BUNDLE), name)
    path = ROOT / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_bundle_jsonl(name: str) -> list[dict[str, Any]]:
    if APP_BUNDLE.exists() and name in zip_names(str(APP_BUNDLE)):
        return read_jsonl_from_zip(str(APP_BUNDLE), name)
    return read_local_jsonl(ROOT / name)


def thumbnail_name(occurrence_id: str) -> str:
    digest = hashlib.sha1(occurrence_id.encode("utf-8")).hexdigest()[:16]
    return f"{THUMB_PREFIX}/{digest}.jpg"


@st.cache_data(show_spinner=False)
def read_thumbnail_from_zip(path: str, name: str) -> bytes:
    with zipfile.ZipFile(path) as zf:
        return zf.read(name)


def thumbnail_bytes(occurrence_id: str) -> bytes | None:
    if not APP_BUNDLE.exists():
        return None
    name = thumbnail_name(occurrence_id)
    if name not in zip_names(str(APP_BUNDLE)):
        return None
    return read_thumbnail_from_zip(str(APP_BUNDLE), name)


def show_source_image(image_url: str, occurrence_id: str = "") -> None:
    if not image_url:
        st.warning("No image URL found.")
        return
    thumb = thumbnail_bytes(occurrence_id) if occurrence_id else None
    if thumb:
        st.image(thumb, use_container_width=True)
        st.caption("Cached preview; open the source image for the full-resolution original.")
    else:
        st.image(image_url, use_container_width=True)
    st.link_button("Open source image", image_url)


@st.cache_data(show_spinner=False)
def load_eval100() -> dict[str, Any]:
    prefix = "data/processed/real_eval_100_"
    llm_prefix = "data/interim/llm/deepseek_v4_pro_eval100_"
    data = {
        "eval_set": read_bundle_csv(prefix + "eval_set.csv"),
        "demo_set": read_bundle_csv(prefix + "demo_set.csv"),
        "image_manifest": read_bundle_csv(prefix + "image_manifest.csv"),
        "ocr_by_region": read_bundle_csv(prefix + "ocr_by_region.csv"),
        "ocr_combined": read_bundle_csv(prefix + "ocr_combined.csv"),
        "evaluation_detail": read_bundle_csv(prefix + "evaluation_detail.csv"),
        "evaluation_summary": read_bundle_csv(prefix + "evaluation_summary.csv"),
        "predictions_rule": read_bundle_csv(prefix + "predictions_rule.csv"),
        "predictions_no_rag": read_bundle_csv(prefix + "predictions_deepseek_no_rag.csv"),
        "predictions_rag": read_bundle_csv(prefix + "predictions_deepseek_rag.csv"),
        "llm_no_rag": read_bundle_jsonl(llm_prefix + "no_rag_outputs.jsonl"),
        "llm_rag": read_bundle_jsonl(llm_prefix + "rag_outputs.jsonl"),
        "rag_contexts": read_bundle_jsonl(llm_prefix + "rag_contexts.jsonl"),
        "report": read_bundle_text("reports/real_eval_100_deepseek_v4_pro_report.md"),
        "source": str(APP_BUNDLE) if APP_BUNDLE.exists() else "local data/processed",
    }
    return data


def method_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    metrics = ["coverage", "exact_match", "token_f1", "parse_failure_rate", "not_evaluated_rate"]
    available = [col for col in metrics if col in summary.columns]
    out = summary.copy()
    for col in available:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.groupby("method", as_index=False)[available].mean().round(3)


def prediction_rows(data: dict[str, Any], occ: str) -> pd.DataFrame:
    frames = []
    for label, key in [
        ("rule_ocr", "predictions_rule"),
        ("deepseek_v4_pro_no_rag", "predictions_no_rag"),
        ("deepseek_v4_pro_rag", "predictions_rag"),
    ]:
        df = data[key]
        if not df.empty and "occurrenceID" in df.columns:
            sub = df[df["occurrenceID"] == occ].copy()
            if len(sub):
                sub["method"] = sub.get("method", label)
                frames.append(sub)
    if not frames:
        return pd.DataFrame()
    rows = []
    for _, row in pd.concat(frames, ignore_index=True).iterrows():
        for field in EXTRACTION_FIELDS:
            rows.append({
                "method": clean(row.get("method")),
                "field": field,
                "value": clean(row.get(field)),
                "confidence": clean(row.get(f"{field}_confidence")),
            })
    return pd.DataFrame(rows)


def llm_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {clean(row.get("occurrenceID")): row for row in rows}


def first_row(df: pd.DataFrame, occ: str) -> dict[str, Any]:
    if df.empty or "occurrenceID" not in df.columns:
        return {}
    match = df[df["occurrenceID"] == occ]
    return match.iloc[0].to_dict() if not match.empty else {}


def selected_record(data: dict[str, Any]) -> str:
    eval_set = data["eval_set"]
    if eval_set.empty:
        return ""
    occurrence_ids = [clean(value) for value in eval_set["occurrenceID"].tolist()]
    labels = []
    for _, row in eval_set.iterrows():
        bits = [clean(row.get("catalogNumber")), clean(row.get("institutionCode")), clean(row.get("scientificName"))]
        labels.append(" | ".join([item for item in bits if item]) or clean(row.get("occurrenceID")))
    default = 0
    current = clean(st.session_state.get("selected_occ"))
    if current in occurrence_ids:
        default = occurrence_ids.index(current)
    choice = st.sidebar.selectbox(
        "Record",
        list(range(len(labels))),
        index=default,
        format_func=lambda i: labels[i],
        key="record_choice",
    )
    occ = clean(eval_set.iloc[int(choice)]["occurrenceID"])
    st.session_state["selected_occ"] = occ
    return occ


def select_gallery_record(occ: str, index: int) -> None:
    st.session_state["selected_occ"] = occ
    st.session_state["record_choice"] = index
    st.session_state["page"] = "Pipeline Review"


def parsed_record_table(row: dict[str, Any]) -> pd.DataFrame:
    parsed = row.get("parsed_json") or {}
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except Exception:
            parsed = {}
    rows = []
    for field in EXTRACTION_FIELDS:
        item = parsed.get(field, {}) if isinstance(parsed, dict) else {}
        if not isinstance(item, dict):
            item = {"value": item, "confidence": "", "evidence_span": ""}
        evidence = item.get("evidence_span", "")
        if isinstance(evidence, (dict, list)):
            evidence = json.dumps(evidence, ensure_ascii=False)
        rows.append({
            "field": field,
            "value": clean(item.get("value")),
            "confidence": clean(item.get("confidence")),
            "evidence": clean(evidence),
        })
    return pd.DataFrame(rows)


def llm_row_for_method(data: dict[str, Any], occ: str, method: str) -> dict[str, Any]:
    rows = data["llm_no_rag"] if method == "deepseek_v4_pro_no_rag" else data["llm_rag"]
    return llm_index(rows).get(occ, {})


def comparison_table(data: dict[str, Any], occ: str) -> pd.DataFrame:
    eval_set = data["eval_set"]
    gold = first_row(eval_set, occ)
    preds = prediction_rows(data, occ)
    detail = data["evaluation_detail"]
    rows = []
    for field in EXTRACTION_FIELDS:
        row: dict[str, Any] = {"field": field, "gold": clean(gold.get(field))}
        for method in ["rule_ocr", "deepseek_v4_pro_no_rag", "deepseek_v4_pro_rag"]:
            match = preds[(preds["method"] == method) & (preds["field"] == field)] if not preds.empty else pd.DataFrame()
            row[method] = clean(match.iloc[0]["value"]) if not match.empty else ""
            if not detail.empty and "occurrenceID" in detail.columns:
                dmatch = detail[
                    (detail["occurrenceID"] == occ)
                    & (detail["method"] == method)
                    & (detail["field"] == field)
                ]
                row[f"{method}_f1"] = clean(dmatch.iloc[0].get("token_f1")) if not dmatch.empty else ""
        rows.append(row)
    return pd.DataFrame(rows)


def show_llm_result_panel(row: dict[str, Any], title: str) -> None:
    st.markdown(f"**{title}**")
    if not row:
        st.info("No LLM output found for this record.")
        return
    cols = st.columns(4)
    cols[0].metric("Raw chars", clean(row.get("raw_output_length")) or "0")
    cols[1].metric("Parsed", "no" if row.get("parse_failure") else "yes")
    cols[2].metric("RAG ctx", len(row.get("retrieved_context") or []))
    cols[3].metric("Model", clean(row.get("actual_model_if_available")) or clean(row.get("requested_model")))
    if row.get("error_message"):
        st.warning(clean(row.get("error_message")))
    st.dataframe(parsed_record_table(row), use_container_width=True, hide_index=True)
    with st.expander("Prompt, context, and raw output", expanded=False):
        st.text_area(f"{title} prompt", clean(row.get("prompt")), height=180, label_visibility="collapsed")
        context = row.get("retrieved_context") or []
        if context:
            st.json(context, expanded=False)
        st.code(clean(row.get("raw_output")), language="json")


def show_pipeline_review(data: dict[str, Any]) -> None:
    st.title("Pipeline Review")
    st.caption("One specimen at a time: original image, OCR text, LLM extraction, and evaluation side by side.")
    occ = selected_record(data)
    if not occ:
        st.info("No records available.")
        return

    eval_set = data["eval_set"]
    gold = first_row(eval_set, occ)
    image_row = first_row(data["image_manifest"], occ)
    ocr_row = first_row(data["ocr_combined"], occ)
    ocr_region = first_row(data["ocr_by_region"], occ)
    no_rag = llm_row_for_method(data, occ, "deepseek_v4_pro_no_rag")
    rag = llm_row_for_method(data, occ, "deepseek_v4_pro_rag")

    st.subheader(clean(gold.get("catalogNumber")) or occ)
    st.caption(occ)
    status = st.columns(5)
    status[0].metric("Image", clean(image_row.get("image_status")) or "unknown")
    status[1].metric("OCR", clean(ocr_region.get("ocr_status")) or "unknown")
    status[2].metric("OCR chars", clean(ocr_row.get("text_length")) or "0")
    status[3].metric("no-RAG raw", clean(no_rag.get("raw_output_length")) or "0")
    status[4].metric("RAG raw", clean(rag.get("raw_output_length")) or "0")

    image_col, ocr_col = st.columns([1.0, 1.15])
    with image_col:
        st.subheader("1. Original Image")
        image_url = clean(image_row.get("image_url"))
        show_source_image(image_url, occ)
        st.dataframe(
            pd.DataFrame([{
                "institutionCode": clean(gold.get("institutionCode")),
                "catalogNumber": clean(gold.get("catalogNumber")),
                "scientificName": clean(gold.get("scientificName")),
                "image_status": clean(image_row.get("image_status")),
            }]),
            use_container_width=True,
            hide_index=True,
        )

    with ocr_col:
        st.subheader("2. OCR Result")
        ocr_meta = pd.DataFrame([{
            "engine": clean(ocr_region.get("ocr_engine")),
            "status": clean(ocr_region.get("ocr_status")),
            "confidence": clean(ocr_region.get("ocr_confidence")),
            "used_fixture_text": clean(ocr_region.get("used_fixture_text")),
        }])
        st.dataframe(ocr_meta, use_container_width=True, hide_index=True)
        st.text_area("OCR text", clean(ocr_row.get("ocr_text")), height=460, label_visibility="collapsed")

    st.subheader("3. LLM Result")
    tab_no, tab_rag = st.tabs(["DeepSeek no-RAG", "DeepSeek RAG"])
    with tab_no:
        show_llm_result_panel(no_rag, "DeepSeek no-RAG")
    with tab_rag:
        show_llm_result_panel(rag, "DeepSeek RAG")

    st.subheader("4. Gold vs Predictions")
    table = comparison_table(data, occ)
    st.dataframe(table, use_container_width=True, hide_index=True)


def show_image_gallery(data: dict[str, Any]) -> None:
    st.title("Image Gallery")
    st.caption("Browse original specimen images first, then open one record in the full pipeline review.")
    eval_set = data["eval_set"].copy()
    if eval_set.empty:
        st.info("No records available.")
        return
    query = st.text_input("Filter by catalogue, institution, or taxon", "")
    if query:
        q = query.lower()
        mask = eval_set.apply(lambda row: q in " ".join(clean(row.get(col)).lower() for col in ["catalogNumber", "institutionCode", "scientificName", "occurrenceID"]), axis=1)
        eval_set = eval_set[mask]
    page_size = st.slider("Images per page", min_value=6, max_value=24, value=12, step=6)
    total = len(eval_set)
    pages = max((total - 1) // page_size + 1, 1)
    page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1)
    start = (int(page) - 1) * page_size
    subset = eval_set.iloc[start:start + page_size]
    st.caption(f"Showing {start + 1 if total else 0}-{min(start + page_size, total)} of {total}")

    manifest = data["image_manifest"]
    for row_start in range(0, len(subset), 3):
        cols = st.columns(3)
        for offset, (_, row) in enumerate(subset.iloc[row_start:row_start + 3].iterrows()):
            occ = clean(row.get("occurrenceID"))
            absolute_index = data["eval_set"].index[data["eval_set"]["occurrenceID"] == occ].tolist()[0]
            image_row = first_row(manifest, occ)
            image_url = clean(image_row.get("image_url"))
            thumb = thumbnail_bytes(occ)
            with cols[offset]:
                if thumb:
                    st.image(thumb, use_container_width=True)
                elif image_url:
                    st.image(image_url, use_container_width=True)
                else:
                    st.warning("No image URL")
                st.markdown(f"**{clean(row.get('catalogNumber')) or 'No catalogue number'}**")
                st.caption(clean(row.get("scientificName")) or occ)
                st.button(
                    "Review pipeline",
                    key=f"review-{absolute_index}",
                    on_click=select_gallery_record,
                    args=(occ, int(absolute_index)),
                    use_container_width=True,
                )


def show_overview(data: dict[str, Any]) -> None:
    st.title("Herbarium SCRIBE")
    st.caption("Interactive browser for the real-image OCR + DeepSeek + RAG evaluation.")
    eval_set = data["eval_set"]
    ocr = data["ocr_by_region"]
    llm_no = data["llm_no_rag"]
    llm_rag = data["llm_rag"]
    cols = st.columns(5)
    cols[0].metric("EVAL records", len(eval_set))
    cols[1].metric("OCR rows", len(ocr))
    cols[2].metric("no-RAG outputs", sum(bool(row.get("raw_output")) for row in llm_no))
    cols[3].metric("RAG outputs", sum(bool(row.get("raw_output")) for row in llm_rag))
    cols[4].metric("Data source", "eval100" if "real_eval_100" in data["source"] else "local")

    comparison = method_comparison(data["evaluation_summary"])
    if not comparison.empty:
        st.subheader("Method Comparison")
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        chart = comparison.set_index("method")[["exact_match", "token_f1"]]
        st.bar_chart(chart)

    with st.expander("Evaluation Summary By Field", expanded=False):
        st.dataframe(data["evaluation_summary"], use_container_width=True, hide_index=True)

    with st.expander("Markdown Report", expanded=False):
        st.markdown(data["report"] or "_No report found._")


def show_record_explorer(data: dict[str, Any]) -> None:
    st.title("Record Explorer")
    occ = selected_record(data)
    if not occ:
        st.info("No records available.")
        return

    eval_set = data["eval_set"]
    gold = eval_set[eval_set["occurrenceID"] == occ].iloc[0].to_dict()
    manifest = data["image_manifest"]
    image_row = manifest[manifest["occurrenceID"] == occ].iloc[0].to_dict() if not manifest[manifest["occurrenceID"] == occ].empty else {}
    ocr_combined = data["ocr_combined"]
    ocr_row = ocr_combined[ocr_combined["occurrenceID"] == occ].iloc[0].to_dict() if not ocr_combined[ocr_combined["occurrenceID"] == occ].empty else {}

    left, right = st.columns([0.95, 1.05])
    with left:
        st.subheader(clean(gold.get("catalogNumber")) or occ)
        image_url = clean(image_row.get("image_url"))
        show_source_image(image_url, occ)
        st.caption(f"occurrenceID: {occ}")

    with right:
        st.subheader("Gold Metadata")
        gold_view = pd.DataFrame([{"field": field, "gold": clean(gold.get(field))} for field in EXTRACTION_FIELDS])
        st.dataframe(gold_view, use_container_width=True, hide_index=True)

    st.subheader("OCR Text")
    st.text_area("Combined OCR", clean(ocr_row.get("ocr_text")), height=260, label_visibility="collapsed")

    st.subheader("Extraction Outputs")
    preds = prediction_rows(data, occ)
    if preds.empty:
        st.info("No predictions found for this record.")
    else:
        st.dataframe(preds, use_container_width=True, hide_index=True)

    st.subheader("Field-Level Evaluation")
    detail = data["evaluation_detail"]
    sub = detail[detail["occurrenceID"] == occ] if "occurrenceID" in detail.columns else pd.DataFrame()
    if sub.empty:
        st.info("No evaluation detail found for this record.")
    else:
        cols = ["method", "field", "prediction", "gold", "exact_match", "token_f1", "ocr_quality_tertile"]
        st.dataframe(sub[[col for col in cols if col in sub.columns]], use_container_width=True, hide_index=True)


def show_llm_trace(data: dict[str, Any]) -> None:
    st.title("LLM / RAG Trace")
    occ = selected_record(data)
    if not occ:
        return
    method = st.radio("Method", ["deepseek_v4_pro_no_rag", "deepseek_v4_pro_rag"], horizontal=True)
    source = llm_index(data["llm_no_rag"] if method.endswith("no_rag") else data["llm_rag"])
    row = source.get(occ)
    if not row:
        st.info("No LLM trace found for this record.")
        return

    cols = st.columns(5)
    cols[0].metric("Raw chars", row.get("raw_output_length", 0))
    cols[1].metric("Parsed", "yes" if not row.get("parse_failure") else "no")
    cols[2].metric("Not evaluated", "yes" if row.get("not_evaluated") else "no")
    cols[3].metric("Model", clean(row.get("actual_model_if_available")) or clean(row.get("requested_model")))
    cols[4].metric("RAG ctx", len(row.get("retrieved_context") or []))

    tab_prompt, tab_context, tab_output, tab_json = st.tabs(["Prompt", "Retrieved Context", "Raw Output", "Parsed JSON"])
    with tab_prompt:
        st.text_area("Prompt", clean(row.get("prompt")), height=360, label_visibility="collapsed")
    with tab_context:
        ctx = row.get("retrieved_context") or []
        if ctx:
            st.json(ctx, expanded=False)
        else:
            st.info("No retrieved context for this method.")
    with tab_output:
        st.code(clean(row.get("raw_output")), language="json")
    with tab_json:
        st.json(row.get("parsed_json") or {}, expanded=True)


def streamlit_secret(name: str, default: str = "") -> str:
    if name in os.environ:
        return os.environ[name]
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def configure_deepseek_env() -> bool:
    key = streamlit_secret("DEEPSEEK_API_KEY") or streamlit_secret("DEEPSEEK_API_KEY_SELF")
    if key:
        os.environ["DEEPSEEK_API_KEY"] = key
    os.environ.setdefault("DEEPSEEK_BASE_URL", streamlit_secret("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    os.environ.setdefault("DEEPSEEK_MODEL", streamlit_secret("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    return bool(key)


def run_live_llm(ocr_text: str, use_rag: bool, data: dict[str, Any]) -> dict[str, Any]:
    cfg = {
        "llm": {
            "backend": "deepseek_api",
            "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "model_name": os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
            "temperature": 0.0,
            "max_tokens": 4000,
            "timeout_seconds": 120,
            "retries": 1,
            "retry_backoff_seconds": 10,
            "min_interval_seconds": 0.0,
        }
    }
    retrieved = []
    if use_rag:
        corpus = build_rag_corpus(data["demo_set"])
        retrieved = retrieve_context(ocr_text, corpus, top_k=3)
    ctx = format_context_for_prompt(retrieved)
    prompt = f"Context:\n{ctx}\n\nOCR text:\n{ocr_text}" if retrieved else f"OCR text:\n{ocr_text}"
    messages = [
        {"role": "system", "content": "Extract herbarium label fields as JSON. Return one object with keys catalogNumber, scientificName, recordedBy, eventDate, country, stateProvince, decimalLatitude, decimalLongitude, and typeStatus. Each field must be an object with value, confidence, and evidence_span. Return JSON only."},
        {"role": "user", "content": prompt},
    ]
    meta = call_llm_with_metadata(messages, cfg)
    parsed = _parse_llm_json(clean(meta.get("content")))
    return {"meta": meta, "prompt": prompt, "retrieved_context": retrieved, "parsed": parsed}


def show_live_upload(data: dict[str, Any]) -> None:
    st.title("Live Upload")
    st.caption("Upload one herbarium image to run OCR, rule extraction, and optional DeepSeek extraction.")
    uploaded = st.file_uploader("Image", type=["jpg", "jpeg", "png", "tif", "tiff"])
    use_deepseek = st.checkbox("Run DeepSeek extraction", value=False)
    use_rag = st.checkbox("Use DEMO examples as RAG context", value=True)

    if uploaded is None:
        st.info("Upload an image to begin.")
        return

    image = Image.open(uploaded)
    st.image(image, caption=uploaded.name, use_container_width=True)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.convert("RGB").save(tmp.name)
        image_path = tmp.name

    with st.spinner("Running Tesseract OCR..."):
        ocr_text, confidence, status = ocr_image_tesseract(image_path)

    cols = st.columns(3)
    cols[0].metric("OCR status", status)
    cols[1].metric("OCR chars", len(ocr_text))
    cols[2].metric("Confidence", "" if confidence is None else f"{confidence:.2f}")
    st.text_area("OCR text", ocr_text, height=260)

    st.subheader("Rule Extraction")
    rule = validate_record(extract_rule_based(ocr_text))
    st.dataframe(pd.DataFrame([{"field": field, "value": rule[field]["value"], "confidence": rule[field]["confidence"]} for field in EXTRACTION_FIELDS]), use_container_width=True, hide_index=True)

    if use_deepseek:
        if not configure_deepseek_env():
            st.error("DEEPSEEK_API_KEY is not configured in Streamlit secrets.")
            return
        with st.spinner("Calling DeepSeek..."):
            result = run_live_llm(ocr_text, use_rag, data)
        meta = result["meta"]
        st.subheader("DeepSeek Extraction")
        st.caption(f"requested={meta.get('requested_model')} actual={meta.get('actual_model')} endpoint_reachable={meta.get('endpoint_reachable')}")
        if meta.get("error_message"):
            st.warning(clean(meta.get("error_message")))
        parsed = result["parsed"] or {}
        if parsed:
            flat = flatten_record(validate_record(parsed))
            st.dataframe(pd.DataFrame([{"field": field, "value": flat[field], "confidence": flat[f"{field}_confidence"]} for field in EXTRACTION_FIELDS]), use_container_width=True, hide_index=True)
        else:
            st.info("DeepSeek returned no parsed JSON.")
        with st.expander("Prompt and Raw Output"):
            st.text_area("Prompt", result["prompt"], height=260)
            st.code(clean(meta.get("content")), language="json")
        with st.expander("Retrieved Context"):
            st.json(result["retrieved_context"], expanded=False)


def main() -> None:
    data = load_eval100()
    st.sidebar.title("Herbarium SCRIBE")
    st.sidebar.caption(f"Data: {data['source']}")
    page = st.sidebar.radio(
        "View",
        [
            "Hespi v10 Results",
            "Qwen Vision Comparison",
            "Pipeline Review",
            "Image Gallery",
            "Overview",
            "Record Explorer",
            "LLM/RAG Trace",
            "Live Upload",
        ],
        key="page",
    )
    if page == "Hespi v10 Results":
        show_hespi_v10_home(HESPI_V10_REPORT)
    elif page == "Qwen Vision Comparison":
        show_qwen_vision_comparison(QWEN_VISION_REPORT)
    elif page == "Overview":
        show_overview(data)
    elif page == "Pipeline Review":
        show_pipeline_review(data)
    elif page == "Image Gallery":
        show_image_gallery(data)
    elif page == "Record Explorer":
        show_record_explorer(data)
    elif page == "LLM/RAG Trace":
        show_llm_trace(data)
    else:
        show_live_upload(data)


if __name__ == "__main__":
    main()
