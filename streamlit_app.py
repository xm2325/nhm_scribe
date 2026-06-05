from __future__ import annotations

import io
import json
import os
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
LOCAL_PROCESSED = ROOT / "data" / "processed"
LOCAL_LLM = ROOT / "data" / "interim" / "llm"


st.set_page_config(page_title="Herbarium SCRIBE", layout="wide")


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


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


def selected_record(data: dict[str, Any]) -> str:
    eval_set = data["eval_set"]
    if eval_set.empty:
        return ""
    labels = []
    for _, row in eval_set.iterrows():
        bits = [clean(row.get("catalogNumber")), clean(row.get("institutionCode")), clean(row.get("scientificName"))]
        labels.append(" | ".join([item for item in bits if item]) or clean(row.get("occurrenceID")))
    choice = st.sidebar.selectbox("Record", list(range(len(labels))), format_func=lambda i: labels[i])
    return clean(eval_set.iloc[int(choice)]["occurrenceID"])


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
        if image_url:
            st.image(image_url, use_container_width=True)
            st.link_button("Open Source Image", image_url)
        else:
            st.warning("No image URL found.")
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
    page = st.sidebar.radio("View", ["Overview", "Record Explorer", "LLM/RAG Trace", "Live Upload"])
    if page == "Overview":
        show_overview(data)
    elif page == "Record Explorer":
        show_record_explorer(data)
    elif page == "LLM/RAG Trace":
        show_llm_trace(data)
    else:
        show_live_upload(data)


if __name__ == "__main__":
    main()
