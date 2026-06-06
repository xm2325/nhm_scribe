from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("") if path.exists() else pd.DataFrame()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def method_summary(data_dir: Path, prefix: str, label: str) -> pd.DataFrame:
    summary = read_csv(data_dir / "processed" / f"{prefix}_evaluation_summary.csv")
    if summary.empty:
        return summary
    metrics = [
        "coverage",
        "exact_match",
        "token_f1",
        "parse_failure_rate",
        "not_evaluated_rate",
    ]
    summary = numeric(summary, metrics)
    available = [column for column in metrics if column in summary.columns]
    out = summary.groupby("method", as_index=False)[available].mean()
    out.insert(0, "pipeline", label)
    return out


def ocr_summary(data_dir: Path, prefix: str, label: str) -> dict[str, Any]:
    ocr = read_csv(data_dir / "processed" / f"{prefix}_ocr_by_region.csv")
    proxy = read_csv(data_dir / "processed" / "ocr_proxy_field_presence.csv")
    if ocr.empty:
        return {"pipeline": label, "records": 0, "regions": 0}
    lengths = pd.to_numeric(ocr.get("text_length", 0), errors="coerce").fillna(0)
    record_text = ocr.assign(_length=lengths).groupby("occurrenceID")["_length"].sum()
    proxy_scores = pd.to_numeric(
        proxy.get("ocr_evidence_proxy_score", pd.Series(dtype=float)),
        errors="coerce",
    )
    return {
        "pipeline": label,
        "records": int(ocr["occurrenceID"].nunique()),
        "regions": int(len(ocr)),
        "nonempty_regions": int((lengths > 0).sum()),
        "mean_regions_per_record": len(ocr) / max(ocr["occurrenceID"].nunique(), 1),
        "mean_ocr_chars_per_record": float(record_text.mean()),
        "mean_ocr_evidence_proxy": float(proxy_scores.mean()) if proxy_scores.notna().any() else float("nan"),
    }


def llm_summary(data_dir: Path, output_name: str, label: str) -> dict[str, Any]:
    rows = read_jsonl(data_dir / "interim" / "llm" / output_name)
    return {
        "pipeline": label,
        "attempted": len(rows),
        "nonempty": sum(bool(row.get("raw_output")) for row in rows),
        "parsed": sum(
            bool(row.get("raw_output"))
            and row.get("parse_failure") is False
            and not bool(row.get("not_evaluated"))
            for row in rows
        ),
        "not_evaluated": sum(bool(row.get("not_evaluated")) for row in rows),
    }


def field_deltas(
    baseline_dir: Path,
    hespi_dir: Path,
    baseline_prefix: str,
    hespi_prefix: str,
) -> pd.DataFrame:
    baseline = numeric(
        read_csv(baseline_dir / "processed" / f"{baseline_prefix}_evaluation_summary.csv"),
        ["coverage", "exact_match", "token_f1"],
    )
    hespi = numeric(
        read_csv(hespi_dir / "processed" / f"{hespi_prefix}_evaluation_summary.csv"),
        ["coverage", "exact_match", "token_f1"],
    )
    if baseline.empty or hespi.empty:
        return pd.DataFrame()
    merged = baseline.merge(hespi, on=["method", "field"], suffixes=("_baseline", "_hespi"))
    for metric in ["coverage", "exact_match", "token_f1"]:
        merged[f"{metric}_delta"] = merged[f"{metric}_hespi"] - merged[f"{metric}_baseline"]
    return merged


def markdown_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    return "_(empty)_\n" if df.empty else df.head(max_rows).to_markdown(index=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", default="data/experiments/hespi_lite_baseline")
    parser.add_argument("--hespi-dir", default="data/experiments/hespi_lite")
    parser.add_argument("--output", default="reports/hespi_lite_poc/hespi_lite_poc_comparison.md")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    hespi_dir = Path(args.hespi_dir)
    baseline_prefix = "hespi_lite_eval20_baseline"
    hespi_prefix = "hespi_lite_eval20"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    baseline_eval = read_csv(baseline_dir / "processed" / f"{baseline_prefix}_eval_set.csv")
    hespi_eval = read_csv(hespi_dir / "processed" / f"{hespi_prefix}_eval_set.csv")
    baseline_ids = set(baseline_eval.get("occurrenceID", []))
    hespi_ids = set(hespi_eval.get("occurrenceID", []))
    same_eval_set = baseline_ids == hespi_ids and len(baseline_ids) > 0

    methods = pd.concat([
        method_summary(baseline_dir, baseline_prefix, "full_image_baseline"),
        method_summary(hespi_dir, hespi_prefix, "hespi_lite"),
    ], ignore_index=True)
    ocr = pd.DataFrame([
        ocr_summary(baseline_dir, baseline_prefix, "full_image_baseline"),
        ocr_summary(hespi_dir, hespi_prefix, "hespi_lite"),
    ])
    llm = pd.DataFrame([
        llm_summary(baseline_dir, "hespi_lite_eval20_baseline_no_rag_outputs.jsonl", "full_image_baseline"),
        llm_summary(hespi_dir, "hespi_lite_eval20_no_rag_outputs.jsonl", "hespi_lite"),
    ])
    deltas = field_deltas(baseline_dir, hespi_dir, baseline_prefix, hespi_prefix)
    hespi_diag = read_csv(hespi_dir / "processed" / f"{hespi_prefix}_hespi_layout_diagnostics.csv")

    methods.to_csv(output.parent / "hespi_lite_method_comparison.csv", index=False)
    ocr.to_csv(output.parent / "hespi_lite_ocr_comparison.csv", index=False)
    llm.to_csv(output.parent / "hespi_lite_llm_comparison.csv", index=False)
    deltas.to_csv(output.parent / "hespi_lite_field_deltas.csv", index=False)

    llm_deltas = deltas[deltas["method"] == "deepseek_v4_pro_no_rag"] if not deltas.empty else pd.DataFrame()
    mean_delta = pd.to_numeric(llm_deltas.get("token_f1_delta", pd.Series(dtype=float)), errors="coerce").mean()
    if pd.isna(mean_delta):
        verdict = "DeepSeek comparison is unavailable because both pipelines did not produce comparable evaluable outputs."
    elif mean_delta > 0:
        verdict = f"Hespi-lite improved mean DeepSeek field token F1 by {mean_delta:.3f} in this POC."
    elif mean_delta < 0:
        verdict = f"Hespi-lite reduced mean DeepSeek field token F1 by {abs(mean_delta):.3f} in this POC."
    else:
        verdict = "Hespi-lite made no mean DeepSeek field token F1 difference in this POC."

    fallback_count = 0
    if not hespi_diag.empty:
        fallback_count = hespi_diag.get("fallback_used", pd.Series(dtype=str)).astype(str).str.lower().isin(
            {"true", "1", "yes"}
        ).sum()

    lines = [
        "# Hespi-lite POC Comparison\n",
        "\n## Design\n",
        "- Dataset: Zenodo 6372393 real herbarium images.\n",
        "- EVAL size: 20 records, with a 2-record DEMO set.\n",
        "- Baseline: full image -> Tesseract -> DeepSeek V4 Pro no-RAG.\n",
        "- Hespi-lite: sheet component detection -> primary label field detection -> Tesseract -> DeepSeek V4 Pro no-RAG.\n",
        "- Hespi HTR, fuzzy matching, and Hespi LLM correction are disabled.\n",
        f"- Baseline and Hespi use the same EVAL occurrenceIDs: `{same_eval_set}`.\n",
        f"- EVAL occurrenceID intersection: `{len(baseline_ids & hespi_ids)}`.\n",
        "\n## Layout result\n",
        f"- Hespi records with an explicit fallback: `{int(fallback_count)}` of `{len(hespi_diag)}`.\n",
        markdown_table(hespi_diag),
        "\n## OCR comparison\n",
        markdown_table(ocr),
        "\n## LLM call health\n",
        markdown_table(llm),
        "\n## Method comparison\n",
        markdown_table(methods),
        "\n## Field-level deltas\n",
        "Positive delta means Hespi-lite outperformed full-image OCR on the same field and method.\n\n",
        markdown_table(deltas),
        "\n## POC verdict\n",
        verdict + "\n",
        "\n## Interpretation guardrails\n",
        "- This is a 20-record feasibility test, not a production accuracy estimate.\n",
        "- The OCR evidence score is a metadata-presence proxy, not CER or WER.\n",
        "- Hespi uses pretrained herbarium models; possible overlap or institutional similarity with the benchmark must be checked before claiming generalisation.\n",
        "- A layout improvement is useful only if OCR evidence and downstream extraction improve without unacceptable fallback rates.\n",
    ]
    output.write_text("".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
