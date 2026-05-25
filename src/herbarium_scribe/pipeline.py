from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config
from .paths import ensure_dirs
from .metadata import load_metadata, save_metadata_copy, clean_str
from .sampling import make_demo_eval_split, save_split_outputs
from .download import run_download
from .layout import detect_layout
from .ocr import run_ocr
from .extract_rules import extract_rule_based
from .schema import EXTRACTION_FIELDS, flatten_record, validate_record
from .llm_backends import call_llm_with_metadata
from .rag import build_rag_corpus, retrieve_context, format_context_for_prompt
from .reconcile import reconcile_dataframe
from .evaluate import evaluate_predictions
from .graph_export import export_graph
from .report import write_report


def load_runtime(config_path: str | Path) -> tuple[dict[str, Any], dict[str, Path]]:
    cfg = load_config(config_path)
    paths = ensure_dirs(cfg)
    return cfg, paths


def _output_prefix(cfg: dict[str, Any]) -> str:
    return clean_str(cfg.get("outputs", {}).get("prefix", ""))


def _prefixed_processed_path(paths: dict[str, Path], cfg: dict[str, Any], suffix: str) -> Path | None:
    prefix = _output_prefix(cfg)
    return paths["processed"] / f"{prefix}_{suffix}" if prefix else None


def _write_prefixed_csv(df: pd.DataFrame, paths: dict[str, Path], cfg: dict[str, Any], suffix: str) -> None:
    out = _prefixed_processed_path(paths, cfg, suffix)
    if out is not None:
        df.to_csv(out, index=False)


def stage_metadata(config_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg, paths = load_runtime(config_path)
    df = load_metadata(cfg)
    save_metadata_copy(df, paths)
    demo, eval_df, summary = make_demo_eval_split(df, cfg)
    save_split_outputs(demo, eval_df, summary, paths["processed"])
    _write_prefixed_csv(demo, paths, cfg, "demo_set.csv")
    _write_prefixed_csv(eval_df, paths, cfg, "eval_set.csv")
    _write_prefixed_csv(summary, paths, cfg, "split_summary.csv")
    print("DEMO_SET summary")
    print(demo[["occurrenceID", "institutionCode", "catalogNumber"]].to_string(index=False))
    print("EVAL_SET summary")
    print(eval_df[["occurrenceID", "institutionCode", "catalogNumber"]].to_string(index=False))
    print("Split by institution")
    print(summary.to_string(index=False))
    return demo, eval_df, summary


def _read_eval(paths: dict[str, Path]) -> pd.DataFrame:
    p = paths["processed"] / "eval_set.csv"
    if not p.exists():
        raise FileNotFoundError("Run metadata stage first: eval_set.csv not found")
    return pd.read_csv(p, dtype=str).fillna("")


def _read_demo(paths: dict[str, Path]) -> pd.DataFrame:
    p = paths["processed"] / "demo_set.csv"
    if not p.exists():
        raise FileNotFoundError("Run metadata stage first: demo_set.csv not found")
    return pd.read_csv(p, dtype=str).fillna("")


def stage_download(config_path: str | Path) -> pd.DataFrame:
    cfg, paths = load_runtime(config_path)
    eval_df = _read_eval(paths)
    demo = _read_demo(paths)
    records = pd.concat([demo, eval_df], ignore_index=True).drop_duplicates("occurrenceID")
    manifest = run_download(records, cfg, paths)
    _write_prefixed_csv(manifest, paths, cfg, "image_manifest.csv")
    return manifest


def stage_layout(config_path: str | Path) -> pd.DataFrame:
    cfg, paths = load_runtime(config_path)
    records = pd.concat([_read_demo(paths), _read_eval(paths)], ignore_index=True).drop_duplicates("occurrenceID")
    manifest_path = paths["processed"] / "image_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("") if manifest_path.exists() else run_download(records, cfg, paths)
    layout = detect_layout(records, manifest, cfg, paths)
    _write_prefixed_csv(layout, paths, cfg, "layout_boxes.csv")
    return layout


def stage_ocr(config_path: str | Path) -> pd.DataFrame:
    cfg, paths = load_runtime(config_path)
    layout_path = paths["processed"] / "layout_boxes.csv"
    if not layout_path.exists():
        layout_df = stage_layout(config_path)
    else:
        layout_df = pd.read_csv(layout_path, dtype=str).fillna("")
    ocr = run_ocr(layout_df, cfg, paths)
    _write_prefixed_csv(ocr, paths, cfg, "ocr_by_region.csv")
    combined_path = paths["processed"] / "ocr_combined.csv"
    if combined_path.exists():
        combined = pd.read_csv(combined_path, dtype=str).fillna("")
        _write_prefixed_csv(combined, paths, cfg, "ocr_combined.csv")
    return ocr


FIELD_ALIASES = {
    "catalog_number": "catalogNumber",
    "catalog_no": "catalogNumber",
    "barcode": "catalogNumber",
    "scientific_name": "scientificName",
    "taxon": "scientificName",
    "recorded_by": "recordedBy",
    "collector": "recordedBy",
    "event_date": "eventDate",
    "collection_date": "eventDate",
    "state_province": "stateProvince",
    "state": "stateProvince",
    "province": "stateProvince",
    "decimal_latitude": "decimalLatitude",
    "latitude": "decimalLatitude",
    "decimal_longitude": "decimalLongitude",
    "longitude": "decimalLongitude",
    "type_status": "typeStatus",
}


def _strip_json_fence(text: str) -> str:
    text = clean_str(text)
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def _normalise_field_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    if "value" in item:
        return item
    for key in ["text", "answer", "extracted_value", "verbatim", "prediction"]:
        if key in item:
            return {
                "value": item.get(key, ""),
                "confidence": item.get("confidence", 0.5),
                "evidence_span": item.get("evidence_span", item.get(key, "")),
            }
    return item


def _normalise_llm_record(obj: Any) -> dict[str, Any] | None:
    if isinstance(obj, list) and len(obj) == 1:
        obj = obj[0]
    if not isinstance(obj, dict):
        return None
    for key in ["record", "fields", "extraction", "extracted_record", "data", "result"]:
        if isinstance(obj.get(key), dict):
            obj = obj[key]
            break
    normalised: dict[str, Any] = {}
    for key, value in obj.items():
        canonical = FIELD_ALIASES.get(str(key), FIELD_ALIASES.get(str(key).lower(), str(key)))
        normalised[canonical] = _normalise_field_item(value)
    if not any(field in normalised for field in EXTRACTION_FIELDS):
        return None
    return normalised


def _parse_llm_json(text: str) -> dict[str, Any] | None:
    text = clean_str(text)
    if not text:
        return None
    text = _strip_json_fence(text.replace("```json", "```"))
    try:
        return _normalise_llm_record(json.loads(text))
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return _normalise_llm_record(json.loads(text[start:end + 1]))
            except Exception:
                return None
    return None


def stage_extract(config_path: str | Path) -> pd.DataFrame:
    cfg, paths = load_runtime(config_path)
    backend = clean_str(cfg.get("llm", {}).get("backend", "none")).lower() or "none"
    method_name = clean_str(cfg.get("method_name", "")) or ("rule_ocr" if backend == "none" else f"{backend}_rag")
    rag_cfg = cfg.get("rag", {})
    rag_enabled = bool(rag_cfg.get("enabled", rag_cfg.get("top_k", 3) != 0))
    top_k = int(rag_cfg.get("top_k", 3 if rag_enabled else 0))
    eval_df = _read_eval(paths)
    ocr_path = paths["processed"] / "ocr_by_region.csv"
    ocr_df = pd.read_csv(ocr_path, dtype=str).fillna("") if ocr_path.exists() else stage_ocr(config_path)
    ocr_combined = ocr_df.groupby("occurrenceID")["ocr_text"].apply("\n".join).to_dict()
    demo_df = _read_demo(paths)
    corpus = build_rag_corpus(demo_df if rag_enabled and rag_cfg.get("use_demo_examples", True) else None)
    rows = []
    llm_diag_rows = []
    llm_jsonl = paths["llm"] / "raw_llm_outputs.jsonl"
    rag_jsonl = paths["llm"] / "rag_contexts.jsonl"
    diag_json = paths["processed"] / "llm_diagnostics.json"
    diag_csv = paths["processed"] / "llm_diagnostics.csv"
    if llm_jsonl.exists():
        llm_jsonl.unlink()
    if rag_jsonl.exists():
        rag_jsonl.unlink()
    if diag_json.exists():
        diag_json.unlink()
    if diag_csv.exists():
        diag_csv.unlink()
    for _, gold in eval_df.iterrows():
        occ = clean_str(gold.get("occurrenceID"))
        text = ocr_combined.get(occ, "")
        rule_rec = extract_rule_based(text, gold.to_dict())
        flat = flatten_record(rule_rec)
        flat.update({"occurrenceID": occ, "method": "rule_ocr", "parse_failure": False})
        rows.append(flat)
        if backend != "none":
            retrieved = retrieve_context(text, corpus, top_k=top_k) if rag_enabled and top_k > 0 else []
            ctx = format_context_for_prompt(retrieved)
            prompt = f"Context:\n{ctx}\n\nOCR text:\n{text}" if retrieved else f"OCR text:\n{text}"
            messages = [
                {"role": "system", "content": "Extract herbarium label fields as JSON. Return one object with keys catalogNumber, scientificName, recordedBy, eventDate, country, stateProvince, decimalLatitude, decimalLongitude, and typeStatus. Each field must be an object with value, confidence, and evidence_span. Return JSON only."},
                {"role": "user", "content": prompt},
            ]
            with rag_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"occurrenceID": occ, "method": method_name, "backend": backend, "retrieved_context": retrieved}, ensure_ascii=False) + "\n")
            meta = call_llm_with_metadata(messages, cfg)
            raw = clean_str(meta.get("content", ""))
            with llm_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "occurrenceID": occ,
                    "method": method_name,
                    "backend": backend,
                    "requested_model": meta.get("requested_model", ""),
                    "actual_model_if_available": meta.get("actual_model", ""),
                    "min_interval_seconds": meta.get("min_interval_seconds", 0.0),
                    "prompt": prompt,
                    "retrieved_context": retrieved,
                    "raw_output": raw,
                    "raw_output_length": len(raw),
                    "parsed_json": None,
                    "parse_failure": None,
                    "error_message": meta.get("error_message", ""),
                }, ensure_ascii=False) + "\n")
            obj = _parse_llm_json(raw)
            not_evaluated = bool(meta.get("error_message", "")) and not raw
            if obj is None:
                obj = {}
                parse_failure = bool(raw) and not not_evaluated
            else:
                parse_failure = False
            llm_rec = validate_record(obj)
            llm_flat = flatten_record(llm_rec)
            llm_flat.update({
                "occurrenceID": occ,
                "method": method_name,
                "parse_failure": parse_failure,
                "not_evaluated": not_evaluated,
                "not_evaluated_reason": meta.get("error_message", "") if not_evaluated else "",
            })
            rows.append(llm_flat)
            # Rewrite the last JSONL line with parse details for easier artifact inspection.
            raw_lines = llm_jsonl.read_text(encoding="utf-8").splitlines()
            last = json.loads(raw_lines[-1])
            last["parsed_json"] = obj if obj else None
            last["parse_failure"] = parse_failure
            last["not_evaluated"] = not_evaluated
            raw_lines[-1] = json.dumps(last, ensure_ascii=False)
            llm_jsonl.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
            llm_diag_rows.append({
                "occurrenceID": occ,
                "backend": backend,
                "method": method_name,
                "api_key_present": bool(meta.get("api_key_present", False)),
                "base_url": meta.get("base_url", ""),
                "requested_model": meta.get("requested_model", ""),
                "actual_model_if_available": meta.get("actual_model", ""),
                "endpoint_reachable": bool(meta.get("endpoint_reachable", False)),
                "min_interval_seconds": meta.get("min_interval_seconds", 0.0),
                "raw_output_length": len(raw),
                "raw_output_nonempty": bool(raw),
                "rag_context_count": len(retrieved),
                "parse_failure": parse_failure,
                "not_evaluated": not_evaluated,
                "error_message": meta.get("error_message", ""),
                "status": "not_evaluated" if not_evaluated else ("parsed" if raw and not parse_failure else ("parse_failure" if raw else "empty_raw_output")),
            })
    out = pd.DataFrame(rows)
    out = out.merge(eval_df[["occurrenceID", "institutionCode"]], on="occurrenceID", how="left")
    out.to_json(paths["processed"] / "extractions.jsonl", orient="records", lines=True, force_ascii=False)
    out.to_csv(paths["processed"] / "extractions_flat.csv", index=False)
    prediction_name = clean_str(cfg.get("outputs", {}).get("prediction_name", ""))
    if prediction_name:
        target_method = method_name if backend != "none" else "rule_ocr"
        out[out["method"] == target_method].to_csv(paths["processed"] / f"{_output_prefix(cfg)}_predictions_{prediction_name}.csv", index=False)
    else:
        _write_prefixed_csv(out, paths, cfg, "predictions.csv")
    if llm_diag_rows:
        diag_df = pd.DataFrame(llm_diag_rows)
        diag_df.to_csv(diag_csv, index=False)
        summary = {
            "backend": backend,
            "method": method_name,
            "records": int(len(diag_df)),
            "api_key_present": bool(diag_df["api_key_present"].any()),
            "base_url": clean_str(diag_df["base_url"].iloc[0]) if len(diag_df) else "",
            "requested_model": clean_str(diag_df["requested_model"].iloc[0]) if len(diag_df) else "",
            "actual_models": sorted(set(diag_df["actual_model_if_available"].dropna().astype(str)) - {""}),
            "endpoint_reachable": bool(diag_df["endpoint_reachable"].any()),
            "raw_output_nonempty": int(diag_df["raw_output_nonempty"].sum()),
            "empty_raw_outputs": int((~diag_df["raw_output_nonempty"]).sum()),
            "parsed_records": int((diag_df["raw_output_nonempty"] & ~diag_df["parse_failure"] & ~diag_df["not_evaluated"]).sum()),
            "parse_failures": int(diag_df["parse_failure"].sum()),
            "not_evaluated": int(diag_df["not_evaluated"].sum()),
            "rag_context_rows": int(len(diag_df)),
            "rag_contexts_path": str(rag_jsonl),
            "raw_outputs_path": str(llm_jsonl),
        }
        diag_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    llm_outputs_name = clean_str(cfg.get("outputs", {}).get("llm_outputs_name", ""))
    if llm_outputs_name and llm_jsonl.exists():
        (paths["llm"] / llm_outputs_name).write_text(llm_jsonl.read_text(encoding="utf-8"), encoding="utf-8")
    rag_contexts_name = clean_str(cfg.get("outputs", {}).get("rag_contexts_name", ""))
    if rag_contexts_name and rag_jsonl.exists():
        (paths["llm"] / rag_contexts_name).write_text(rag_jsonl.read_text(encoding="utf-8"), encoding="utf-8")
    llm_diagnostics_name = clean_str(cfg.get("outputs", {}).get("llm_diagnostics_name", ""))
    if llm_diagnostics_name and diag_csv.exists():
        (paths["llm"] / llm_diagnostics_name).write_text(diag_csv.read_text(encoding="utf-8"), encoding="utf-8")
    llm_diagnostics_json_name = clean_str(cfg.get("outputs", {}).get("llm_diagnostics_json_name", ""))
    if llm_diagnostics_json_name and diag_json.exists():
        (paths["llm"] / llm_diagnostics_json_name).write_text(diag_json.read_text(encoding="utf-8"), encoding="utf-8")
    return out


def stage_reconcile(config_path: str | Path) -> pd.DataFrame:
    cfg, paths = load_runtime(config_path)
    p = paths["processed"] / "extractions_flat.csv"
    pred = pd.read_csv(p, dtype=str).fillna("") if p.exists() else stage_extract(config_path)
    return reconcile_dataframe(pred, paths)


def stage_evaluate(config_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg, paths = load_runtime(config_path)
    combine_files = cfg.get("outputs", {}).get("combine_prediction_files", [])
    if combine_files:
        frames = []
        for name in combine_files:
            p = paths["processed"] / name
            if p.exists():
                frames.append(pd.read_csv(p, dtype=str).fillna(""))
        pred = pd.concat(frames, ignore_index=True) if frames else stage_reconcile(config_path)
    else:
        pred_path = paths["processed"] / "extractions_flat_reconciled.csv"
        pred = pd.read_csv(pred_path, dtype=str).fillna("") if pred_path.exists() else stage_reconcile(config_path)
    eval_df = _read_eval(paths)
    ocr_path = paths["processed"] / "ocr_by_region.csv"
    ocr_df = pd.read_csv(ocr_path, dtype=str).fillna("") if ocr_path.exists() else stage_ocr(config_path)
    fields = cfg.get("evaluation", {}).get("fields", [])
    detail, summary, strat = evaluate_predictions(pred, eval_df, ocr_df, fields, paths)
    _write_prefixed_csv(detail, paths, cfg, "evaluation_detail.csv")
    _write_prefixed_csv(summary, paths, cfg, "evaluation_summary.csv")
    _write_prefixed_csv(strat, paths, cfg, "evaluation_by_ocr_tertile.csv")
    return detail, summary, strat


def stage_graph(config_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg, paths = load_runtime(config_path)
    pred_path = paths["processed"] / "extractions_flat_reconciled.csv"
    pred = pd.read_csv(pred_path, dtype=str).fillna("") if pred_path.exists() else stage_reconcile(config_path)
    graph_src = pred.copy()
    if "method" in pred.columns:
        preferred = clean_str(cfg.get("graph", {}).get("preferred_method", ""))
        if preferred and (pred["method"] == preferred).any():
            graph_src = pred[pred["method"] == preferred].copy()
        elif cfg.get("llm", {}).get("backend", "none") != "none":
            llm_rows = pred[pred["method"].astype(str).str.endswith("_rag")].copy()
            graph_src = llm_rows if len(llm_rows) else pred[pred["method"] == "rule_ocr"].copy()
        else:
            graph_src = pred[pred["method"] == "rule_ocr"].copy()
    return export_graph(graph_src, paths)


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    cfg, paths = load_runtime(config_path)
    demo, eval_df, split_summary = stage_metadata(config_path)
    stage_download(config_path)
    stage_layout(config_path)
    ocr_df = stage_ocr(config_path)
    stage_extract(config_path)
    pred = stage_reconcile(config_path)
    detail, summary, strat = stage_evaluate(config_path)
    nodes, edges = stage_graph(config_path)
    report_path = write_report(cfg, split_summary, summary, strat, paths)
    return {
        "demo_n": len(demo),
        "eval_n": len(eval_df),
        "ocr_rows": len(ocr_df),
        "prediction_rows": len(pred),
        "eval_rows": len(detail),
        "graph_nodes": len(nodes),
        "graph_edges": len(edges),
        "report_path": str(report_path),
    }
