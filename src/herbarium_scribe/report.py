from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or len(df) == 0:
        return "_(empty)_\n"
    return df.head(max_rows).to_markdown(index=False) + "\n"


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
    lines.append("\n## OCR evidence proxy note\n")
    lines.append("The OCR evidence proxy is not true OCR CER/WER. Catalogue metadata is not full label transcription, so the metric only checks whether catalogue field values appear in OCR text after normalisation.\n")
    lines.append("\n## Limitations\n")
    lines.append("The default demo uses fixture records and deterministic extraction. Optional PaddleOCR, Qwen, OpenAI, Anthropic, GBIF, and GeoNames paths are intentionally kept outside the default run.\n")
    out.write_text("".join(lines), encoding="utf-8")
    return out
