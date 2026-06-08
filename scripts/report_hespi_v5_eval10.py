from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("data/experiments/hespi_v5_eval10")
PROCESSED = ROOT / "processed"
LLM = ROOT / "interim" / "llm"
REPORT_DIR = Path("reports/hespi_v5_eval10")
METHOD = "deepseek_v4_pro_nonthinking_barcode_gbif"
V4_REFERENCE = {
    "run_id": 27160844627,
    "coverage": 0.30357142857142855,
    "exact_match": 0.125,
    "token_f1": 0.15625,
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("") if path.exists() else pd.DataFrame()


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(
        frame.get(column, pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    )


def markdown(frame: pd.DataFrame, max_rows: int = 30) -> str:
    return "_(empty)_\n" if frame.empty else frame.head(max_rows).to_markdown(index=False) + "\n"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    detail = read_csv(PROCESSED / "eval_detail.csv")
    llm_detail = detail[detail.get("method", pd.Series(dtype=str)).eq(METHOD)].copy()
    if llm_detail.empty:
        raise SystemExit(f"No evaluation rows found for {METHOD}")

    filled = llm_detail[llm_detail["prediction"].astype(str).str.strip().ne("")]
    field_metrics = llm_detail.groupby("field", as_index=False).agg(
        evaluable_field_units=("evaluable", lambda values: numeric(pd.DataFrame({"v": values}), "v").sum()),
        coverage=("coverage", lambda values: pd.to_numeric(values, errors="coerce").mean()),
        exact_match=("exact_match", lambda values: pd.to_numeric(values, errors="coerce").mean()),
        token_f1=("token_f1", lambda values: pd.to_numeric(values, errors="coerce").mean()),
        direct_evidence_support_rate=(
            "direct_evidence_supported",
            lambda values: pd.to_numeric(values, errors="coerce").mean(),
        ),
        unsupported_prediction_rate=(
            "unsupported_prediction",
            lambda values: pd.to_numeric(values, errors="coerce").mean(),
        ),
        review_required_rate=(
            "review_required",
            lambda values: pd.to_numeric(values, errors="coerce").mean(),
        ),
    )

    ocr = read_csv(PROCESSED / "ocr_by_region.csv")
    barcode = ocr[ocr.get("ocr_engine", pd.Series(dtype=str)).eq("zxingcpp")].copy()
    eval_set = read_csv(PROCESSED / "eval_set.csv")
    if not eval_set.empty:
        eval_ids = set(eval_set["occurrenceID"].astype(str))
        barcode = barcode[barcode["occurrenceID"].astype(str).isin(eval_ids)].copy()
    if not barcode.empty:
        barcode["barcode_count"] = numeric(barcode, "barcode_count").fillna(0).astype(int)
        barcode_view = barcode[[
            "occurrenceID",
            "ocr_status",
            "ocr_text",
            "barcode_format",
            "barcode_count",
            "barcode_ambiguous",
            "barcode_elapsed_seconds",
        ]]
    else:
        barcode_view = pd.DataFrame()

    reconciliation = read_csv(PROCESSED / "reconciliation.csv")
    correction_flags = reconciliation.get(
        "taxonomy_correction_applied",
        pd.Series("", index=reconciliation.index, dtype=str),
    )
    corrections = reconciliation[
        correction_flags
        .astype(str)
        .str.lower()
        .isin({"true", "1"})
    ].copy()

    output_path = LLM / "hespi_v5_eval10_outputs.jsonl"
    outputs = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if output_path.exists() else []
    parsed = [
        row for row in outputs
        if row.get("raw_output")
        and row.get("parse_failure") is False
        and not row.get("not_evaluated")
    ]
    total_tokens = sum(
        int((row.get("response_usage", {}) or {}).get("total_tokens", 0) or 0)
        for row in outputs
    )

    summary = pd.DataFrame([{
        "records": llm_detail["occurrenceID"].nunique(),
        "evaluable_field_units": int(numeric(llm_detail, "evaluable").sum()),
        "coverage": numeric(llm_detail, "coverage").mean(),
        "exact_match": numeric(llm_detail, "exact_match").mean(),
        "token_f1": numeric(llm_detail, "token_f1").mean(),
        "exact_match_among_filled": numeric(filled, "exact_match").mean(),
        "token_f1_among_filled": numeric(filled, "token_f1").mean(),
        "unsupported_prediction_rate": numeric(filled, "unsupported_prediction").mean(),
        "review_required_rate": numeric(filled, "review_required").mean(),
        "parsed_records": len(parsed),
        "parse_success_rate": len(parsed) / len(outputs) if outputs else float("nan"),
        "barcode_records": len(barcode),
        "barcode_records_with_values": int(barcode["barcode_count"].gt(0).sum()) if len(barcode) else 0,
        "barcode_ambiguous_records": int(barcode["barcode_count"].gt(1).sum()) if len(barcode) else 0,
        "taxonomy_corrections": len(corrections),
        "total_tokens": total_tokens,
    }])
    summary["coverage_delta_vs_v4"] = summary["coverage"] - V4_REFERENCE["coverage"]
    summary["exact_match_delta_vs_v4"] = summary["exact_match"] - V4_REFERENCE["exact_match"]
    summary["token_f1_delta_vs_v4"] = summary["token_f1"] - V4_REFERENCE["token_f1"]
    summary.to_csv(REPORT_DIR / "hespi_v5_summary.csv", index=False)
    field_metrics.to_csv(REPORT_DIR / "hespi_v5_field_metrics.csv", index=False)
    barcode_view.to_csv(REPORT_DIR / "hespi_v5_barcode_diagnostics.csv", index=False)
    corrections.to_csv(REPORT_DIR / "hespi_v5_taxonomy_corrections.csv", index=False)

    row = summary.iloc[0]
    scale_ready = (
        float(row["parse_success_rate"]) >= 0.95
        and float(row["exact_match"]) >= 0.20
        and float(row["token_f1"]) >= 0.25
        and float(row["unsupported_prediction_rate"]) <= 0.15
        and float(row["review_required_rate"]) <= 0.30
    )
    lines = [
        "# Hespi v5 Eval10 Barcode + GBIF Report\n",
        "\n## Experiment\n",
        "- The EVAL set is the same frozen ten-record set used by Hespi v4.\n",
        "- ZXing-C++ barcode decoding augments, but does not replace, Tesseract OCR.\n",
        "- GBIF correction is allowed only when confidence is at least 90 and the parsed binomial is unchanged.\n",
        "- RAG is disabled and DeepSeek thinking is disabled.\n",
        f"- Historical reference: Hespi v4 repeat 1, GitHub Actions run `{V4_REFERENCE['run_id']}`.\n",
        "\n## Summary\n",
        markdown(summary),
        "\n## Field metrics\n",
        markdown(field_metrics),
        "\n## Barcode diagnostics\n",
        markdown(barcode_view),
        "\n## Taxonomy corrections\n",
        markdown(corrections),
        "\n## Scale decision\n",
        f"- Ready for a larger evaluation: `{scale_ready}`.\n",
        "- Gates: parse success >=95%, exact match >=20%, token F1 >=25%, unsupported predictions <=15%, human review <=30%.\n",
        "\n## Interpretation guardrails\n",
        "- A decoded barcode is direct image evidence, but multiple barcodes on one sheet remain ambiguous.\n",
        "- A GBIF correction can repair authorship spelling; it cannot recover a missing or badly corrupted binomial.\n",
        "- Ten records are a debugging sample, not a production performance estimate.\n",
    ]
    report = REPORT_DIR / "hespi_v5_improvement_report.md"
    report.write_text("".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
