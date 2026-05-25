from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text not in {"", "0", "false", "f", "no", "n", "none", "nan"}


def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or len(df) == 0:
        return "_(empty)_\n"
    return df.head(max_rows).to_markdown(index=False) + "\n"


def read_llm_diagnostics(paths: dict[str, Path]) -> tuple[dict[str, Any], pd.DataFrame]:
    output_files = sorted(paths["llm"].glob("deepseek_v4_pro_*_outputs.jsonl"))
    if not output_files and (paths["llm"] / "raw_llm_outputs.jsonl").exists():
        output_files = [paths["llm"] / "raw_llm_outputs.jsonl"]

    rows: list[dict[str, Any]] = []
    for path in output_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            retrieved = item.get("retrieved_context", [])
            rows.append({
                "occurrenceID": item.get("occurrenceID", ""),
                "method": item.get("method", ""),
                "backend": item.get("backend", ""),
                "requested_model": item.get("requested_model", ""),
                "actual_model_if_available": item.get("actual_model_if_available", ""),
                "raw_output_length": item.get("raw_output_length", len(str(item.get("raw_output", "")))),
                "raw_output_nonempty": bool(item.get("raw_output", "")),
                "rag_context_count": len(retrieved) if isinstance(retrieved, list) else 0,
                "parse_failure": _truthy(item.get("parse_failure", False)),
                "error_message": item.get("error_message", ""),
                "output_file": str(path),
            })

    diag_files = sorted(paths["llm"].glob("deepseek_v4_pro_*_diagnostics.csv"))
    if (paths["processed"] / "llm_diagnostics.csv").exists():
        diag_files.append(paths["processed"] / "llm_diagnostics.csv")
    diag_frames = [pd.read_csv(path, dtype=str).fillna("") for path in diag_files if path.exists()]
    diag = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
    if len(diag) and {"occurrenceID", "method"}.issubset(diag.columns):
        diag = diag.drop_duplicates(["occurrenceID", "method"], keep="last")

    detail = pd.DataFrame(rows)
    if len(detail) and len(diag):
        enrich_cols = [
            col for col in ["occurrenceID", "method", "api_key_present", "base_url", "endpoint_reachable", "status"]
            if col in diag.columns
        ]
        if {"occurrenceID", "method"}.issubset(set(enrich_cols)):
            detail = detail.merge(diag[enrich_cols], on=["occurrenceID", "method"], how="left")
    elif len(diag):
        detail = diag

    if len(detail) == 0:
        return {}, detail

    actual_models = sorted(set(detail.get("actual_model_if_available", pd.Series(dtype=str)).dropna().astype(str)) - {""})
    requested_models = sorted(set(detail.get("requested_model", pd.Series(dtype=str)).dropna().astype(str)) - {""})
    backends = sorted(set(detail.get("backend", pd.Series(dtype=str)).dropna().astype(str)) - {""})
    summary = {
        "backend": ", ".join(backends),
        "requested_model": ", ".join(requested_models),
        "actual_models": actual_models,
        "api_key_present": bool(detail.get("api_key_present", pd.Series(dtype=str)).map(_truthy).any()) if "api_key_present" in detail else False,
        "base_url": ", ".join(sorted(set(detail.get("base_url", pd.Series(dtype=str)).dropna().astype(str)) - {""})) if "base_url" in detail else "",
        "endpoint_reachable": bool(detail.get("endpoint_reachable", pd.Series(dtype=str)).map(_truthy).any()) if "endpoint_reachable" in detail else False,
        "records": int(len(detail)),
        "raw_output_nonempty": int(detail["raw_output_nonempty"].sum()) if "raw_output_nonempty" in detail else 0,
        "empty_raw_outputs": int((~detail["raw_output_nonempty"]).sum()) if "raw_output_nonempty" in detail else 0,
        "parsed_records": int((~detail["parse_failure"]).sum()) if "parse_failure" in detail else 0,
        "parse_failures": int(detail["parse_failure"].sum()) if "parse_failure" in detail else 0,
        "raw_outputs_path": ", ".join(str(path) for path in output_files),
        "rag_contexts_path": str(paths["llm"] / "deepseek_v4_pro_rag_contexts.jsonl"),
    }
    return summary, detail


def _output_prefix(cfg: dict[str, Any]) -> str:
    return str(cfg.get("outputs", {}).get("prefix", "") or "").strip()


def _processed_csv(paths: dict[str, Path], cfg: dict[str, Any], suffix: str) -> pd.DataFrame:
    prefix = _output_prefix(cfg)
    candidates = []
    if prefix:
        candidates.append(paths["processed"] / f"{prefix}_{suffix}")
    candidates.append(paths["processed"] / suffix)
    for path in candidates:
        if path.exists():
            return pd.read_csv(path, dtype=str).fillna("")
    return pd.DataFrame()


def _method_comparison(eval_summary: pd.DataFrame) -> pd.DataFrame:
    if eval_summary is None or len(eval_summary) == 0:
        return pd.DataFrame()
    return eval_summary.groupby("method", as_index=False).agg(
        coverage=("coverage", "mean"),
        exact_match=("exact_match", "mean"),
        token_f1=("token_f1", "mean"),
        parse_failure_rate=("parse_failure_rate", "mean"),
        validation_warning_rate=("validation_warning_rate", "mean"),
    )


def _rag_delta_tables(paths: dict[str, Path], cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    detail = _processed_csv(paths, cfg, "evaluation_detail.csv")
    if len(detail) == 0:
        return pd.DataFrame(), pd.DataFrame(), "RAG comparison unavailable because evaluation detail was not found."
    no_rag = detail[detail["method"] == "deepseek_v4_pro_no_rag"]
    rag = detail[detail["method"] == "deepseek_v4_pro_rag"]
    if len(no_rag) == 0 or len(rag) == 0:
        return pd.DataFrame(), pd.DataFrame(), "RAG comparison unavailable because both DeepSeek no-RAG and RAG methods were not present."
    if pd.to_numeric(no_rag.get("parse_failure", pd.Series(dtype=float)), errors="coerce").fillna(0).eq(1).all() and pd.to_numeric(rag.get("parse_failure", pd.Series(dtype=float)), errors="coerce").fillna(0).eq(1).all():
        return pd.DataFrame(), pd.DataFrame(), "RAG comparison unavailable because DeepSeek produced no parsed outputs for either no-RAG or RAG."
    cols = ["occurrenceID", "field", "exact_match", "token_f1"]
    merged = no_rag[cols].merge(rag[cols], on=["occurrenceID", "field"], suffixes=("_no_rag", "_rag"))
    for col in ["exact_match_no_rag", "exact_match_rag", "token_f1_no_rag", "token_f1_rag"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["token_f1_delta"] = merged["token_f1_rag"] - merged["token_f1_no_rag"]
    helped = merged[merged["token_f1_delta"] > 0].head(10)
    hurt = merged[merged["token_f1_delta"] <= 0].head(10)
    mean_delta = merged["token_f1_delta"].mean()
    if pd.isna(mean_delta):
        verdict = "RAG comparison inconclusive because comparable token F1 values were unavailable."
    elif mean_delta > 0:
        verdict = f"RAG improved mean token F1 by {mean_delta:.3f} on this fixture smoke test."
    elif mean_delta < 0:
        verdict = f"RAG reduced mean token F1 by {abs(mean_delta):.3f} on this fixture smoke test."
    else:
        verdict = "RAG made no average token F1 difference on this fixture smoke test."
    return helped, hurt, verdict


def write_report(cfg: dict[str, Any], split_summary: pd.DataFrame, eval_summary: pd.DataFrame, stratified: pd.DataFrame, paths: dict[str, Path]) -> Path:
    report_name = cfg.get("outputs", {}).get("report_name", "herbarium_scribe_demo_report.md")
    out = paths["reports"] / report_name
    demo = _processed_csv(paths, cfg, "demo_set.csv")
    eval_set = _processed_csv(paths, cfg, "eval_set.csv")
    image_manifest = _processed_csv(paths, cfg, "image_manifest.csv")
    ocr = _processed_csv(paths, cfg, "ocr_by_region.csv")
    mode = cfg.get("experiment", {}).get("mode", "fixture_only" if cfg.get("ocr", {}).get("allow_fixture_text", True) else "real_image")
    demo_ids = set(demo.get("occurrenceID", [])) if len(demo) else set()
    eval_ids = set(eval_set.get("occurrenceID", [])) if len(eval_set) else set()
    overlap = sorted(demo_ids & eval_ids)
    image_success = int((image_manifest.get("image_path", pd.Series(dtype=str)).astype(str) != "").sum()) if len(image_manifest) else 0
    ocr_success = int((pd.to_numeric(ocr.get("text_length", pd.Series(dtype=str)), errors="coerce").fillna(0) > 0).sum()) if len(ocr) else 0
    fixture_used = int(ocr.get("ocr_engine", pd.Series(dtype=str)).astype(str).str.contains("fixture_text", na=False).sum()) if len(ocr) else 0
    lines = []
    title = cfg.get("experiment", {}).get("title", "Herbarium SCRIBE Demo Report")
    lines.append(f"# {title}\n")
    lines.append("\n## Run configuration\n")
    lines.append(f"- OCR backend: `{cfg.get('ocr', {}).get('backend', 'tesseract')}`\n")
    lines.append(f"- LLM backend: `{cfg.get('llm', {}).get('backend', 'none')}`\n")
    lines.append(f"- Random seed: `{cfg.get('project', {}).get('random_state', 42)}`\n")
    lines.append(f"- Experiment mode: `{mode}`\n")
    lines.append(f"- DEMO records: `{len(demo)}`\n")
    lines.append(f"- EVAL records: `{len(eval_set)}`\n")
    lines.append(f"- Image load success count: `{image_success}`\n")
    lines.append(f"- OCR success count: `{ocr_success}`\n")
    lines.append(f"- Fixture/fallback OCR text used count: `{fixture_used}`\n")
    if mode == "fixture_only":
        lines.append("- Real-image evaluation could not be run because no local images or downloadable image URLs were available.\n")
    lines.append("\n## Split summary\n")
    lines.append(md_table(split_summary))
    lines.append("\n## Leakage check\n")
    lines.append(f"- DEMO_SET occurrenceIDs: `{', '.join(sorted(demo_ids))}`\n")
    lines.append(f"- EVAL_SET occurrenceIDs: `{', '.join(sorted(eval_ids))}`\n")
    lines.append(f"- DEMO/EVAL occurrenceID overlap: `{len(overlap)}`\n")
    lines.append("- EVAL gold metadata is not included in DeepSeek prompts; RAG examples are built from DEMO_SET only when enabled.\n")
    lines.append("\n## Image and OCR manifest\n")
    if len(image_manifest):
        manifest = image_manifest.merge(
            ocr[["occurrenceID", "ocr_engine", "ocr_status", "text_length", "image_path", "crop_path"]]
            if len(ocr) else pd.DataFrame(columns=["occurrenceID"]),
            on="occurrenceID",
            how="left",
            suffixes=("_manifest", "_ocr"),
        )
        lines.append(md_table(manifest.head(20)))
    else:
        lines.append("_(no image manifest)_\n")
    lines.append("\n## Method comparison\n")
    lines.append(md_table(_method_comparison(eval_summary)))
    lines.append("\n## Evaluation summary\n")
    lines.append(md_table(eval_summary, max_rows=100))
    lines.append("\n## Stratified by OCR quality tertile\n")
    lines.append(md_table(stratified, max_rows=100))
    if len(eval_set) < 25:
        lines.append("\n_Only 10 or fewer EVAL records were used, so OCR quality tertiles are a smoke-test diagnostic rather than a stable stratification._\n")
    llm_summary, llm_detail = read_llm_diagnostics(paths)
    if llm_summary:
        lines.append("\n## LLM and RAG diagnostics\n")
        lines.append(f"- Backend: `{llm_summary.get('backend', '')}`\n")
        lines.append(f"- Requested model: `{llm_summary.get('requested_model', '')}`\n")
        lines.append(f"- Actual model(s): `{', '.join(llm_summary.get('actual_models', []))}`\n")
        lines.append(f"- Base URL: `{llm_summary.get('base_url', '')}`\n")
        lines.append(f"- API key present: `{llm_summary.get('api_key_present', False)}`\n")
        lines.append(f"- Endpoint reachable: `{llm_summary.get('endpoint_reachable', False)}`\n")
        lines.append(f"- LLM records attempted: `{llm_summary.get('records', 0)}`\n")
        lines.append(f"- Non-empty raw outputs: `{llm_summary.get('raw_output_nonempty', 0)}`\n")
        lines.append(f"- Empty raw outputs / likely API failures: `{llm_summary.get('empty_raw_outputs', 0)}`\n")
        lines.append(f"- Parsed records: `{llm_summary.get('parsed_records', 0)}`\n")
        lines.append(f"- Parse failures: `{llm_summary.get('parse_failures', 0)}`\n")
        lines.append(f"- Raw outputs: `{llm_summary.get('raw_outputs_path', '')}`\n")
        lines.append(f"- RAG contexts: `{llm_summary.get('rag_contexts_path', '')}`\n")
        lines.append("\n")
        lines.append(md_table(llm_detail))
    helped, hurt, verdict = _rag_delta_tables(paths, cfg)
    lines.append("\n## RAG vs no-RAG comparison\n")
    lines.append(verdict + "\n")
    lines.append("\n### Examples where RAG helped\n")
    lines.append(md_table(helped))
    lines.append("\n### Examples where RAG hurt or made no difference\n")
    lines.append(md_table(hurt))
    lines.append("\n## OCR evidence proxy note\n")
    lines.append("The OCR evidence proxy is not true OCR CER/WER. Catalogue metadata is not full label transcription, so the metric only checks whether catalogue field values appear in OCR text after normalisation.\n")
    lines.append("\n## Limitations\n")
    lines.append("The default demo uses fixture records and deterministic extraction. Optional PaddleOCR, Qwen, OpenAI, Anthropic, GBIF, and GeoNames paths are intentionally kept outside the default run.\n")
    out.write_text("".join(lines), encoding="utf-8")
    return out
