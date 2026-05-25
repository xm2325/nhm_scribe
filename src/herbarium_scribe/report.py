from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or len(df) == 0:
        return "_(empty)_\n"
    return df.head(max_rows).to_markdown(index=False) + "\n"


def read_llm_diagnostics(paths: dict[str, Path]) -> tuple[dict[str, Any], pd.DataFrame]:
    summary_path = paths["processed"] / "llm_diagnostics.json"
    detail_path = paths["processed"] / "llm_diagnostics.csv"
    summary: dict[str, Any] = {}
    detail = pd.DataFrame()
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if detail_path.exists():
        detail = pd.read_csv(detail_path)
    return summary, detail


def write_report(cfg: dict[str, Any], split_summary: pd.DataFrame, eval_summary: pd.DataFrame, stratified: pd.DataFrame, paths: dict[str, Path]) -> Path:
    out = paths["reports"] / "herbarium_scribe_demo_report.md"
    lines = []
    lines.append("# Herbarium SCRIBE Demo Report\n")
    lines.append("\n## Run configuration\n")
    lines.append(f"- OCR backend: `{cfg.get('ocr', {}).get('backend', 'tesseract')}`\n")
    lines.append(f"- LLM backend: `{cfg.get('llm', {}).get('backend', 'none')}`\n")
    lines.append(f"- Random seed: `{cfg.get('project', {}).get('random_state', 42)}`\n")
    lines.append("\n## Split summary\n")
    lines.append(md_table(split_summary))
    lines.append("\n## Evaluation summary\n")
    lines.append(md_table(eval_summary))
    lines.append("\n## Stratified by OCR quality tertile\n")
    lines.append(md_table(stratified))
    llm_summary, llm_detail = read_llm_diagnostics(paths)
    if llm_summary:
        lines.append("\n## LLM and RAG diagnostics\n")
        lines.append(f"- Backend: `{llm_summary.get('backend', '')}`\n")
        lines.append(f"- LLM records attempted: `{llm_summary.get('records', 0)}`\n")
        lines.append(f"- Non-empty raw outputs: `{llm_summary.get('raw_output_nonempty', 0)}`\n")
        lines.append(f"- Empty raw outputs / likely API failures: `{llm_summary.get('empty_raw_outputs', 0)}`\n")
        lines.append(f"- Parsed records: `{llm_summary.get('parsed_records', 0)}`\n")
        lines.append(f"- Parse failures: `{llm_summary.get('parse_failures', 0)}`\n")
        lines.append(f"- Raw outputs: `{llm_summary.get('raw_outputs_path', '')}`\n")
        lines.append(f"- RAG contexts: `{llm_summary.get('rag_contexts_path', '')}`\n")
        lines.append("\n")
        lines.append(md_table(llm_detail))
    lines.append("\n## OCR evidence proxy note\n")
    lines.append("The OCR evidence proxy is not true OCR CER/WER. Catalogue metadata is not full label transcription, so the metric only checks whether catalogue field values appear in OCR text after normalisation.\n")
    lines.append("\n## Limitations\n")
    lines.append("The default demo uses fixture records and deterministic extraction. Optional PaddleOCR, Qwen, OpenAI, Anthropic, GBIF, and GeoNames paths are intentionally kept outside the default run.\n")
    out.write_text("".join(lines), encoding="utf-8")
    return out
