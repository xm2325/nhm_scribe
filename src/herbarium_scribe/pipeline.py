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
from .schema import flatten_record
from .llm_backends import call_llm
from .rag import build_rag_corpus, retrieve_context, format_context_for_prompt
from .reconcile import reconcile_dataframe
from .evaluate import evaluate_predictions
from .graph_export import export_graph
from .report import write_report


def load_runtime(config_path: str | Path) -> tuple[dict[str, Any], dict[str, Path]]:
    cfg = load_config(config_path)
    paths = ensure_dirs(cfg)
    return cfg, paths


def stage_metadata(config_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg, paths = load_runtime(config_path)
    df = load_metadata(cfg)
    save_metadata_copy(df, paths)
    demo, eval_df, summary = make_demo_eval_split(df, cfg)
    save_split_outputs(demo, eval_df, summary, paths["processed"])
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
    return run_download(records, cfg, paths)


def stage_layout(config_path: str | Path) -> pd.DataFrame:
    cfg, paths = load_runtime(config_path)
    records = pd.concat([_read_demo(paths), _read_eval(paths)], ignore_index=True).drop_duplicates("occurrenceID")
    manifest_path = paths["processed"] / "image_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("") if manifest_path.exists() else run_download(records, cfg, paths)
    return detect_layout(records, manifest, cfg, paths)


def stage_ocr(config_path: str | Path) -> pd.DataFrame:
    cfg, paths = load_runtime(config_path)
    layout_path = paths["processed"] / "layout_boxes.csv"
    if not layout_path.exists():
        layout_df = stage_layout(config_path)
    else:
        layout_df = pd.read_csv(layout_path, dtype=str).fillna("")
    return run_ocr(layout_df, cfg, paths)


def _parse_llm_json(text: str) -> dict[str, Any] | None:
    text = clean_str(text)
    if not text:
        return None
    text = text.replace("```json", "```").strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return None
    return None


def stage_extract(config_path: str | Path) -> pd.DataFrame:
    cfg, paths = load_runtime(config_path)
    eval_df = _read_eval(paths)
    ocr_path = paths["processed"] / "ocr_by_region.csv"
    ocr_df = pd.read_csv(ocr_path, dtype=str).fillna("") if ocr_path.exists() else stage_ocr(config_path)
    ocr_combined = ocr_df.groupby("occurrenceID")["ocr_text"].apply("\n".join).to_dict()
    demo_df = _read_demo(paths)
    corpus = build_rag_corpus(demo_df if cfg.get("rag", {}).get("use_demo_examples", True) else None)
    rows = []
    llm_jsonl = paths["llm"] / "raw_llm_outputs.jsonl"
    if llm_jsonl.exists():
        llm_jsonl.unlink()
    for _, gold in eval_df.iterrows():
        occ = clean_str(gold.get("occurrenceID"))
        text = ocr_combined.get(occ, "")
        rule_rec = extract_rule_based(text, gold.to_dict())
        flat = flatten_record(rule_rec)
        flat.update({"occurrenceID": occ, "method": "rule_ocr", "parse_failure": False})
        rows.append(flat)
        if cfg.get("llm", {}).get("backend", "none") != "none":
            ctx = format_context_for_prompt(retrieve_context(text, corpus, top_k=int(cfg.get("rag", {}).get("top_k", 3))))
            prompt = f"Context:\n{ctx}\n\nOCR text:\n{text}"
            messages = [
                {"role": "system", "content": "Extract herbarium label fields as JSON with value, confidence, evidence_span."},
                {"role": "user", "content": prompt},
            ]
            raw = call_llm(messages, cfg)
            with llm_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"occurrenceID": occ, "raw_output": raw}, ensure_ascii=False) + "\n")
            obj = _parse_llm_json(raw)
            if obj is None:
                obj = {}
                parse_failure = True
            else:
                parse_failure = False
            from .schema import validate_record
            llm_rec = validate_record(obj)
            llm_flat = flatten_record(llm_rec)
            llm_flat.update({"occurrenceID": occ, "method": "llm", "parse_failure": parse_failure})
            rows.append(llm_flat)
    out = pd.DataFrame(rows)
    out = out.merge(eval_df[["occurrenceID", "institutionCode"]], on="occurrenceID", how="left")
    out.to_json(paths["processed"] / "extractions.jsonl", orient="records", lines=True, force_ascii=False)
    out.to_csv(paths["processed"] / "extractions_flat.csv", index=False)
    return out


def stage_reconcile(config_path: str | Path) -> pd.DataFrame:
    cfg, paths = load_runtime(config_path)
    p = paths["processed"] / "extractions_flat.csv"
    pred = pd.read_csv(p, dtype=str).fillna("") if p.exists() else stage_extract(config_path)
    return reconcile_dataframe(pred, paths)


def stage_evaluate(config_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg, paths = load_runtime(config_path)
    pred_path = paths["processed"] / "extractions_flat_reconciled.csv"
    pred = pd.read_csv(pred_path, dtype=str).fillna("") if pred_path.exists() else stage_reconcile(config_path)
    eval_df = _read_eval(paths)
    ocr_path = paths["processed"] / "ocr_by_region.csv"
    ocr_df = pd.read_csv(ocr_path, dtype=str).fillna("") if ocr_path.exists() else stage_ocr(config_path)
    fields = cfg.get("evaluation", {}).get("fields", [])
    return evaluate_predictions(pred, eval_df, ocr_df, fields, paths)


def stage_graph(config_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg, paths = load_runtime(config_path)
    pred_path = paths["processed"] / "extractions_flat_reconciled.csv"
    pred = pd.read_csv(pred_path, dtype=str).fillna("") if pred_path.exists() else stage_reconcile(config_path)
    graph_src = pred[pred["method"] == "rule_ocr"].copy() if "method" in pred.columns else pred.copy()
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
