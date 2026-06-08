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


def _llm_artifact_family(cfg: dict[str, Any]) -> str:
    explicit = str(cfg.get("outputs", {}).get("llm_artifact_family", "") or "").strip()
    if explicit:
        return explicit
    method = str(cfg.get("method_name", "") or "")
    if method.endswith("_rag"):
        return method[:-4]
    if method.endswith("_no_rag"):
        return method[:-7]
    return method


def read_llm_diagnostics(paths: dict[str, Path], cfg: dict[str, Any] | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    cfg = cfg or {}
    family = _llm_artifact_family(cfg)
    output_files: list[Path] = []
    if family:
        output_files = sorted(paths["llm"].glob(f"{family}_*_outputs.jsonl"))
    if not output_files:
        output_files = sorted(path for path in paths["llm"].glob("*_outputs.jsonl") if path.name != "raw_llm_outputs.jsonl")
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
                "min_interval_seconds": item.get("min_interval_seconds", ""),
                "response_finish_reason": item.get("response_finish_reason", ""),
                "response_message_keys": item.get("response_message_keys", []),
                "reasoning_content_length": item.get("reasoning_content_length", 0),
                "raw_output_length": item.get("raw_output_length", len(str(item.get("raw_output", "")))),
                "raw_output_nonempty": bool(item.get("raw_output", "")),
                "llm_call_attempted": _truthy(item.get("llm_call_attempted", True)),
                "skip_reason": item.get("skip_reason", ""),
                "rag_context_count": len(retrieved) if isinstance(retrieved, list) else 0,
                "parse_failure": _truthy(item.get("parse_failure", False)),
                "not_evaluated": _truthy(item.get("not_evaluated", False)),
                "not_evaluated_reason": item.get("not_evaluated_reason", ""),
                "error_message": item.get("error_message", ""),
                "output_file": str(path),
            })

    diag_files = sorted(paths["llm"].glob(f"{family}_*_diagnostics.csv")) if family else []
    if not diag_files:
        diag_files = sorted(paths["llm"].glob("*_diagnostics.csv"))
    if (paths["processed"] / "llm_diagnostics.csv").exists():
        diag_files.append(paths["processed"] / "llm_diagnostics.csv")
    diag_frames = [pd.read_csv(path, dtype=str).fillna("") for path in diag_files if path.exists()]
    diag = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
    if len(diag) and {"occurrenceID", "method"}.issubset(diag.columns):
        diag = diag.drop_duplicates(["occurrenceID", "method"], keep="last")

    detail = pd.DataFrame(rows)
    if len(detail) and len(diag):
        if "not_evaluated" in diag.columns:
            diag = diag.rename(columns={"not_evaluated": "not_evaluated_diag"})
        enrich_cols = [
            col for col in ["occurrenceID", "method", "api_key_present", "base_url", "endpoint_reachable", "status", "not_evaluated_diag"]
            if col in diag.columns
        ]
        if "min_interval_seconds" in diag.columns and "min_interval_seconds" not in detail.columns:
            enrich_cols.append("min_interval_seconds")
        for col in ["response_finish_reason", "response_message_keys", "reasoning_content_length", "reasoning_content_nonempty", "not_evaluated_reason"]:
            if col in diag.columns and col not in detail.columns:
                enrich_cols.append(col)
        if {"occurrenceID", "method"}.issubset(set(enrich_cols)):
            detail = detail.merge(diag[enrich_cols], on=["occurrenceID", "method"], how="left")
            if "not_evaluated_diag" in detail.columns:
                detail["not_evaluated"] = detail["not_evaluated"] | detail["not_evaluated_diag"].map(_truthy)
                detail = detail.drop(columns=["not_evaluated_diag"])
    elif len(diag):
        detail = diag

    if len(detail) == 0:
        return {}, detail

    actual_models = sorted(set(detail.get("actual_model_if_available", pd.Series(dtype=str)).dropna().astype(str)) - {""})
    requested_models = sorted(set(detail.get("requested_model", pd.Series(dtype=str)).dropna().astype(str)) - {""})
    backends = sorted(set(detail.get("backend", pd.Series(dtype=str)).dropna().astype(str)) - {""})
    api_empty_responses = int((~detail["raw_output_nonempty"] & detail.get("endpoint_reachable", pd.Series(False, index=detail.index)).map(_truthy) & detail.get("not_evaluated_reason", pd.Series("", index=detail.index)).astype(str).eq("empty_raw_output")).sum()) if "raw_output_nonempty" in detail else 0
    summary = {
        "backend": ", ".join(backends),
        "requested_model": ", ".join(requested_models),
        "actual_models": actual_models,
        "api_key_present": bool(detail.get("api_key_present", pd.Series(dtype=str)).map(_truthy).any()) if "api_key_present" in detail else False,
        "base_url": ", ".join(sorted(set(detail.get("base_url", pd.Series(dtype=str)).dropna().astype(str)) - {""})) if "base_url" in detail else "",
        "endpoint_reachable": bool(detail.get("endpoint_reachable", pd.Series(dtype=str)).map(_truthy).any()) if "endpoint_reachable" in detail else False,
        "min_interval_seconds": ", ".join(sorted(set(detail.get("min_interval_seconds", pd.Series(dtype=str)).dropna().astype(str)) - {""})) if "min_interval_seconds" in detail else "",
        "records": int(len(detail)),
        "llm_calls_attempted": int(detail.get("llm_call_attempted", pd.Series(True, index=detail.index)).map(_truthy).sum()),
        "llm_calls_skipped": int((~detail.get("llm_call_attempted", pd.Series(True, index=detail.index)).map(_truthy)).sum()),
        "raw_output_nonempty": int(detail["raw_output_nonempty"].sum()) if "raw_output_nonempty" in detail else 0,
        "empty_raw_outputs": int((~detail["raw_output_nonempty"]).sum()) if "raw_output_nonempty" in detail else 0,
        "api_empty_responses": api_empty_responses,
        "api_empty_response_rate": api_empty_responses / max(int(len(detail)), 1),
        "parsed_records": int((detail["raw_output_nonempty"] & ~detail["parse_failure"] & ~detail.get("not_evaluated", pd.Series(False, index=detail.index))).sum()) if {"raw_output_nonempty", "parse_failure"}.issubset(detail.columns) else 0,
        "parse_failures": int(detail["parse_failure"].sum()) if "parse_failure" in detail else 0,
        "not_evaluated": int(detail["not_evaluated"].sum()) if "not_evaluated" in detail else 0,
        "raw_outputs_path": ", ".join(str(path) for path in output_files),
        "rag_contexts_path": str(paths["llm"] / str(cfg.get("outputs", {}).get("rag_contexts_name", ""))) if cfg.get("outputs", {}).get("rag_contexts_name") else "",
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
    agg = {
        "coverage": ("coverage", "mean"),
        "exact_match": ("exact_match", "mean"),
        "token_f1": ("token_f1", "mean"),
        "parse_failure_rate": ("parse_failure_rate", "mean"),
        "validation_warning_rate": ("validation_warning_rate", "mean"),
    }
    if "not_evaluated_rate" in eval_summary.columns:
        agg["not_evaluated_rate"] = ("not_evaluated_rate", "mean")
    for column in [
        "evidence_span_present_rate",
        "direct_evidence_support_rate",
        "unsupported_prediction_rate",
    ]:
        if column in eval_summary.columns:
            agg[column] = (column, "mean")
    return eval_summary.groupby("method", as_index=False).agg(**agg)


def _rag_delta_tables(paths: dict[str, Path], cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    detail = _processed_csv(paths, cfg, "evaluation_detail.csv")
    if len(detail) == 0:
        return pd.DataFrame(), pd.DataFrame(), "RAG comparison unavailable because evaluation detail was not found."
    method = str(cfg.get("method_name", "") or "")
    rag_method = method if method.endswith("_rag") and not method.endswith("_no_rag") else ""
    no_rag_method = f"{rag_method[:-4]}_no_rag" if rag_method else ""
    if method.endswith("_no_rag"):
        no_rag_method = method
        rag_method = f"{method[:-7]}_rag"
    if no_rag_method not in set(detail["method"]) or rag_method not in set(detail["method"]):
        methods = sorted(set(detail["method"].astype(str)) - {"rule_ocr"})
        no_rag_candidates = [item for item in methods if item.endswith("_no_rag")]
        rag_candidates = [item for item in methods if item.endswith("_rag") and not item.endswith("_no_rag")]
        no_rag_method = no_rag_candidates[0] if no_rag_candidates else no_rag_method
        rag_method = rag_candidates[0] if rag_candidates else rag_method
    no_rag = detail[detail["method"] == no_rag_method]
    rag = detail[detail["method"] == rag_method]
    if len(no_rag) and len(rag) == 0:
        return pd.DataFrame(), pd.DataFrame(), "RAG comparison unavailable because only no-RAG predictions were present; RAG was not run for this artifact."
    if len(rag) and len(no_rag) == 0:
        return pd.DataFrame(), pd.DataFrame(), "RAG comparison unavailable because only RAG predictions were present; no-RAG control was not run for this artifact."
    if len(no_rag) == 0 or len(rag) == 0:
        return pd.DataFrame(), pd.DataFrame(), "RAG comparison unavailable because both no-RAG and RAG LLM methods were not present."
    if pd.to_numeric(no_rag.get("not_evaluated", pd.Series(dtype=float)), errors="coerce").fillna(0).eq(1).all() and pd.to_numeric(rag.get("not_evaluated", pd.Series(dtype=float)), errors="coerce").fillna(0).eq(1).all():
        return pd.DataFrame(), pd.DataFrame(), "RAG comparison unavailable because the LLM methods were not evaluated, most likely because the API key was missing or rejected."
    if pd.to_numeric(no_rag.get("parse_failure", pd.Series(dtype=float)), errors="coerce").fillna(0).eq(1).all() and pd.to_numeric(rag.get("parse_failure", pd.Series(dtype=float)), errors="coerce").fillna(0).eq(1).all():
        return pd.DataFrame(), pd.DataFrame(), "RAG comparison unavailable because the LLM produced no parsed outputs for either no-RAG or RAG."
    if pd.to_numeric(no_rag.get("evaluable", pd.Series(dtype=float)), errors="coerce").fillna(0).sum() == 0 or pd.to_numeric(rag.get("evaluable", pd.Series(dtype=float)), errors="coerce").fillna(0).sum() == 0:
        return pd.DataFrame(), pd.DataFrame(), "RAG comparison unavailable because no-RAG and RAG did not both produce evaluable parsed outputs."
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
        verdict = f"RAG improved mean token F1 by {mean_delta:.3f} on this evaluation."
    elif mean_delta < 0:
        verdict = f"RAG reduced mean token F1 by {abs(mean_delta):.3f} on this evaluation."
    else:
        verdict = "RAG made no average token F1 difference on this evaluation."
    return helped, hurt, verdict


def write_report(cfg: dict[str, Any], split_summary: pd.DataFrame, eval_summary: pd.DataFrame, stratified: pd.DataFrame, paths: dict[str, Path]) -> Path:
    report_name = cfg.get("outputs", {}).get("report_name", "herbarium_scribe_demo_report.md")
    out = paths["reports"] / report_name
    demo = _processed_csv(paths, cfg, "demo_set.csv")
    eval_set = _processed_csv(paths, cfg, "eval_set.csv")
    image_manifest = _processed_csv(paths, cfg, "image_manifest.csv")
    layout = _processed_csv(paths, cfg, "layout_boxes.csv")
    ocr = _processed_csv(paths, cfg, "ocr_by_region.csv")
    mode = cfg.get("experiment", {}).get("mode", "fixture_only" if cfg.get("ocr", {}).get("allow_fixture_text", True) else "real_image")
    demo_ids = set(demo.get("occurrenceID", [])) if len(demo) else set()
    eval_ids = set(eval_set.get("occurrenceID", [])) if len(eval_set) else set()
    overlap = sorted(demo_ids & eval_ids)
    image_success = int((image_manifest.get("image_path", pd.Series(dtype=str)).astype(str) != "").sum()) if len(image_manifest) else 0
    image_url_count = int((image_manifest.get("image_url", pd.Series(dtype=str)).astype(str) != "").sum()) if len(image_manifest) else 0
    ocr_status_ok = int(ocr.get("ocr_status", pd.Series(dtype=str)).astype(str).str.contains("ok", na=False).sum()) if len(ocr) else 0
    ocr_nonempty = int((pd.to_numeric(ocr.get("text_length", pd.Series(dtype=str)), errors="coerce").fillna(0) > 0).sum()) if len(ocr) else 0
    ocr_success = ocr_nonempty
    fixture_used = int(ocr.get("ocr_engine", pd.Series(dtype=str)).astype(str).str.contains("fixture_text", na=False).sum()) if len(ocr) else 0
    if "used_fixture_text" in ocr.columns:
        fixture_used = int(ocr["used_fixture_text"].map(_truthy).sum())
    dataset_source = cfg.get("metadata", {}).get("dataset_label") or cfg.get("metadata", {}).get("source", "fixture")
    rbge_manifest = paths["processed"] / "rbge_e00633257_image_manifest.csv"
    rbge_status = "not run in this workspace"
    if rbge_manifest.exists():
        rbge_df = pd.read_csv(rbge_manifest, dtype=str).fillna("")
        rbge_downloaded = int((rbge_df.get("image_path", pd.Series(dtype=str)).astype(str) != "").sum()) if len(rbge_df) else 0
        rbge_status = "image available" if rbge_downloaded else "record created, direct image unavailable"
    lines = []
    title = cfg.get("experiment", {}).get("title", "Herbarium SCRIBE Demo Report")
    lines.append(f"# {title}\n")
    lines.append("\n## Run configuration\n")
    lines.append(f"- OCR backend: `{cfg.get('ocr', {}).get('backend', 'tesseract')}`\n")
    lines.append(f"- Layout strategy: `{cfg.get('layout', {}).get('strategy', 'auto')}`\n")
    lines.append(f"- LLM backend: `{cfg.get('llm', {}).get('backend', 'none')}`\n")
    if cfg.get("llm", {}).get("backend", "none") != "none":
        lines.append(f"- LLM max tokens: `{cfg.get('llm', {}).get('max_tokens', '')}`\n")
        lines.append(f"- LLM thinking: `{cfg.get('llm', {}).get('thinking', 'provider_default')}`\n")
        lines.append(f"- LLM response format: `{cfg.get('llm', {}).get('response_format', 'provider_default')}`\n")
    lines.append(f"- Random seed: `{cfg.get('project', {}).get('random_state', 42)}`\n")
    lines.append(f"- Experiment mode: `{mode}`\n")
    lines.append(f"- Dataset source: `{dataset_source}`\n")
    if mode == "real_image":
        lines.append(f"- RBGE E00633257 smoke test: `{rbge_status}`\n")
    lines.append(f"- DEMO records: `{len(demo)}`\n")
    lines.append(f"- EVAL records: `{len(eval_set)}`\n")
    lines.append(f"- Image URLs available: `{image_url_count}`\n")
    lines.append(f"- Image load success count: `{image_success}`\n")
    lines.append(f"- OCR status `ok` count: `{ocr_status_ok}`\n")
    lines.append(f"- OCR non-empty text count: `{ocr_nonempty}`\n")
    lines.append(f"- OCR success count: `{ocr_success}`\n")
    lines.append(f"- Fixture/fallback OCR text used count: `{fixture_used}`\n")
    if mode == "fixture_only":
        lines.append("- Real-image evaluation could not be run because no local images or downloadable image URLs were available.\n")
    if mode == "real_image":
        lines.append(f"- This is a `{len(eval_set)}`-record real-image evaluation.\n")
    lines.append("\n## Split summary\n")
    lines.append(md_table(split_summary))
    lines.append("\n## Leakage check\n")
    lines.append(f"- DEMO_SET occurrenceIDs: `{', '.join(sorted(demo_ids))}`\n")
    lines.append(f"- EVAL_SET occurrenceIDs: `{', '.join(sorted(eval_ids))}`\n")
    lines.append(f"- DEMO/EVAL occurrenceID overlap: `{len(overlap)}`\n")
    lines.append("- EVAL gold metadata is not included in LLM prompts; RAG examples are built from DEMO_SET only when enabled.\n")
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
    lines.append("\n## Layout diagnostics\n")
    if len(layout):
        layout_counts = layout.groupby("layout_method", as_index=False).agg(
            regions=("region_id", "count"),
            records=("occurrenceID", "nunique"),
        )
        lines.append(md_table(layout_counts))
    else:
        lines.append("_(no layout output)_\n")
    hespi_diag = _processed_csv(paths, cfg, "hespi_layout_diagnostics.csv")
    if len(hespi_diag):
        fallback_count = int(hespi_diag.get("fallback_used", pd.Series(dtype=str)).map(_truthy).sum())
        primary_count = int(pd.to_numeric(
            hespi_diag.get("primary_label_count", pd.Series(0, index=hespi_diag.index)),
            errors="coerce",
        ).fillna(0).sum())
        field_count = int(pd.to_numeric(
            hespi_diag.get("label_field_count", pd.Series(0, index=hespi_diag.index)),
            errors="coerce",
        ).fillna(0).sum())
        lines.append(f"- Hespi records attempted: `{len(hespi_diag)}`\n")
        lines.append(f"- Primary labels detected: `{primary_count}`\n")
        lines.append(f"- Label fields detected: `{field_count}`\n")
        lines.append(f"- Records using fallback: `{fallback_count}`\n")
        lines.append(md_table(hespi_diag))
    lines.append("\n## Method comparison\n")
    lines.append(md_table(_method_comparison(eval_summary)))
    lines.append("\n## Evaluation summary\n")
    lines.append(md_table(eval_summary, max_rows=100))
    lines.append("\n## Stratified by OCR quality tertile\n")
    lines.append(md_table(stratified, max_rows=100))
    if len(eval_set) < 25:
        lines.append("\n_Fewer than 25 EVAL records were used, so OCR quality tertiles are a smoke-test diagnostic rather than a stable stratification._\n")
    llm_summary, llm_detail = read_llm_diagnostics(paths, cfg)
    if llm_summary:
        lines.append("\n## LLM and RAG diagnostics\n")
        lines.append(f"- Backend: `{llm_summary.get('backend', '')}`\n")
        lines.append(f"- Requested model: `{llm_summary.get('requested_model', '')}`\n")
        lines.append(f"- Actual model(s): `{', '.join(llm_summary.get('actual_models', []))}`\n")
        lines.append(f"- Base URL: `{llm_summary.get('base_url', '')}`\n")
        lines.append(f"- API key present: `{llm_summary.get('api_key_present', False)}`\n")
        lines.append(f"- Endpoint reachable: `{llm_summary.get('endpoint_reachable', False)}`\n")
        lines.append(f"- Minimum request interval seconds: `{llm_summary.get('min_interval_seconds', '')}`\n")
        lines.append(f"- LLM records considered: `{llm_summary.get('records', 0)}`\n")
        lines.append(f"- LLM API calls attempted: `{llm_summary.get('llm_calls_attempted', llm_summary.get('records', 0))}`\n")
        lines.append(f"- LLM API calls skipped by evidence gate: `{llm_summary.get('llm_calls_skipped', 0)}`\n")
        lines.append(f"- Non-empty raw outputs: `{llm_summary.get('raw_output_nonempty', 0)}`\n")
        lines.append(
            f"- Empty raw outputs (including evidence-gated skips): "
            f"`{llm_summary.get('empty_raw_outputs', 0)}`\n"
        )
        lines.append(f"- API empty responses: `{llm_summary.get('api_empty_responses', 0)}`\n")
        lines.append(f"- API empty response rate: `{llm_summary.get('api_empty_response_rate', 0)}`\n")
        lines.append(f"- Parsed records: `{llm_summary.get('parsed_records', 0)}`\n")
        lines.append(f"- Parse failures: `{llm_summary.get('parse_failures', 0)}`\n")
        lines.append(f"- Not evaluated: `{llm_summary.get('not_evaluated', 0)}`\n")
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
    eval_detail = _processed_csv(paths, cfg, "evaluation_detail.csv")
    success_cols = [col for col in ["occurrenceID", "method", "field", "prediction", "gold", "token_f1"] if col in eval_detail.columns]
    lines.append("\n## Examples of success\n")
    if len(eval_detail) and success_cols:
        success = eval_detail[pd.to_numeric(eval_detail.get("token_f1", 0), errors="coerce").fillna(0) >= 1][success_cols].head(10)
        lines.append(md_table(success))
    else:
        lines.append("_(empty)_\n")
    lines.append("\n## Examples of failure\n")
    if len(eval_detail) and success_cols:
        failure = eval_detail[pd.to_numeric(eval_detail.get("token_f1", 1), errors="coerce").fillna(1) < 1][success_cols].head(10)
        lines.append(md_table(failure))
    else:
        lines.append("_(empty)_\n")
    lines.append("\n## Limitations\n")
    lines.append("Results show feasibility only, not production-level accuracy. Optional PaddleOCR, Qwen, OpenAI, Anthropic, GBIF, and GeoNames paths are intentionally kept outside the default run.\n")
    if mode == "real_image":
        lines.append("\n## Recommendation\n")
        llm_ok = llm_summary.get("not_evaluated", 1) == 0 and llm_summary.get("parse_failures", 1) / max(llm_summary.get("records", 1), 1) <= 0.2 if llm_summary else False
        image_ok = image_success / max(len(image_manifest), 1) >= 0.8
        ocr_ok = ocr_nonempty / max(len(ocr), 1) >= 0.7
        overlap_ok = len(overlap) == 0
        if image_ok and ocr_ok and overlap_ok and (llm_ok or not llm_summary):
            lines.append("Proceed to 25 records only after confirming LLM parsed outputs are present in the artifact.\n")
        else:
            lines.append("Stop and debug before 25 records unless image download, OCR, and LLM parse rates meet the configured thresholds.\n")
    out.write_text("".join(lines), encoding="utf-8")
    return out
