from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("data/experiments/hespi_v6_eval10")
PROCESSED = ROOT / "processed"
LLM = ROOT / "interim" / "llm"
REPORT_DIR = Path("reports/hespi_v6_eval10")
METHOD = "deepseek_v4_pro_nonthinking_barcode_gbif_number_ensemble"
V5_REFERENCE = {
    "run_id": 27163416125,
    "coverage": 0.30357142857142855,
    "exact_match": 0.19642857142857142,
    "token_f1": 0.20535714285714285,
    "catalog_exact_match": 0.7,
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
        evaluable_field_units=("evaluable", lambda values: pd.to_numeric(values, errors="coerce").sum()),
        coverage=("coverage", lambda values: pd.to_numeric(values, errors="coerce").mean()),
        exact_match=("exact_match", lambda values: pd.to_numeric(values, errors="coerce").mean()),
        token_f1=("token_f1", lambda values: pd.to_numeric(values, errors="coerce").mean()),
        unsupported_prediction_rate=(
            "unsupported_prediction",
            lambda values: pd.to_numeric(values, errors="coerce").mean(),
        ),
        review_required_rate=(
            "review_required",
            lambda values: pd.to_numeric(values, errors="coerce").mean(),
        ),
    )
    catalog_rows = llm_detail[llm_detail["field"].eq("catalogNumber")]
    catalog_exact = numeric(catalog_rows, "exact_match").mean()

    eval_set = read_csv(PROCESSED / "eval_set.csv")
    eval_ids = set(eval_set.get("occurrenceID", pd.Series(dtype=str)).astype(str))
    ocr = read_csv(PROCESSED / "ocr_by_region.csv")
    if eval_ids and not ocr.empty:
        ocr = ocr[ocr["occurrenceID"].astype(str).isin(eval_ids)].copy()
    ensemble = ocr[
        ocr.get("ocr_engine", pd.Series(dtype=str)).eq("tesseract_catalog_number_ensemble")
    ].copy()
    if not ensemble.empty:
        ensemble["ocr_ensemble_candidate_count"] = numeric(
            ensemble,
            "ocr_ensemble_candidate_count",
        ).fillna(0).astype(int)
        ensemble["ocr_ensemble_attempts"] = numeric(
            ensemble,
            "ocr_ensemble_attempts",
        ).fillna(0).astype(int)
        ensemble["ocr_ensemble_elapsed_seconds"] = numeric(
            ensemble,
            "ocr_ensemble_elapsed_seconds",
        ).fillna(0.0)
        ensemble_view = ensemble[[
            "occurrenceID",
            "region_id",
            "region_type",
            "ocr_status",
            "ocr_text",
            "ocr_ensemble_candidate_count",
            "ocr_ensemble_ambiguous",
            "ocr_ensemble_attempts",
            "ocr_ensemble_elapsed_seconds",
            "ocr_ensemble_candidates_json",
        ]]
    else:
        ensemble_view = pd.DataFrame()

    output_path = LLM / "hespi_v6_eval10_outputs.jsonl"
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
        "catalog_exact_match": catalog_exact,
        "parsed_records": len(parsed),
        "parse_success_rate": len(parsed) / len(outputs) if outputs else float("nan"),
        "ensemble_records": ensemble["occurrenceID"].nunique() if len(ensemble) else 0,
        "ensemble_regions": len(ensemble),
        "ensemble_regions_with_candidates": (
            int(ensemble["ocr_ensemble_candidate_count"].gt(0).sum())
            if len(ensemble)
            else 0
        ),
        "ensemble_ambiguous_regions": (
            int(ensemble["ocr_ensemble_candidate_count"].gt(1).sum())
            if len(ensemble)
            else 0
        ),
        "ensemble_tesseract_attempts": (
            int(ensemble["ocr_ensemble_attempts"].sum())
            if len(ensemble)
            else 0
        ),
        "ensemble_elapsed_seconds": (
            float(ensemble["ocr_ensemble_elapsed_seconds"].sum())
            if len(ensemble)
            else 0.0
        ),
        "total_tokens": total_tokens,
    }])
    summary["coverage_delta_vs_v5"] = summary["coverage"] - V5_REFERENCE["coverage"]
    summary["exact_match_delta_vs_v5"] = summary["exact_match"] - V5_REFERENCE["exact_match"]
    summary["token_f1_delta_vs_v5"] = summary["token_f1"] - V5_REFERENCE["token_f1"]
    summary["catalog_exact_delta_vs_v5"] = (
        summary["catalog_exact_match"] - V5_REFERENCE["catalog_exact_match"]
    )

    summary.to_csv(REPORT_DIR / "hespi_v6_summary.csv", index=False)
    field_metrics.to_csv(REPORT_DIR / "hespi_v6_field_metrics.csv", index=False)
    ensemble_view.to_csv(REPORT_DIR / "hespi_v6_number_ensemble_diagnostics.csv", index=False)

    row = summary.iloc[0]
    scale_ready = (
        float(row["parse_success_rate"]) >= 0.95
        and float(row["exact_match"]) >= 0.20
        and float(row["token_f1"]) >= 0.25
        and float(row["unsupported_prediction_rate"]) <= 0.15
        and float(row["review_required_rate"]) <= 0.30
    )
    lines = [
        "# Hespi v6 Eval10 Catalog Number OCR Ensemble Report\n",
        "\n## Experiment\n",
        "- The EVAL set is the same frozen ten-record set used by Hespi v5.\n",
        "- Only the OCR treatment of high-confidence number, barcode, and database-label crops changed.\n",
        "- Each selected crop is enlarged and read with two image variants and Tesseract PSM 7/13.\n",
        "- Candidate values are ranked hypotheses, not independent confirmations or automatic truth.\n",
        "- ZXing barcode decoding, GBIF reconciliation, DeepSeek settings, and all other evidence channels remain unchanged.\n",
        f"- Historical reference: Hespi v5, GitHub Actions run `{V5_REFERENCE['run_id']}`.\n",
        "\n## Summary\n",
        markdown(summary),
        "\n## Field metrics\n",
        markdown(field_metrics),
        "\n## Number ensemble diagnostics\n",
        markdown(ensemble_view),
        "\n## Scale decision\n",
        f"- Ready for a larger evaluation: `{scale_ready}`.\n",
        "- Gates: parse success >=95%, exact match >=20%, token F1 >=25%, unsupported predictions <=15%, human review <=30%.\n",
        "\n## Interpretation guardrails\n",
        "- More OCR candidates can improve recall while also increasing ambiguity and review workload.\n",
        "- A catalog-number gain is useful only if overall unsupported predictions do not increase materially.\n",
        "- Ten records are a debugging sample, not a production performance estimate.\n",
    ]
    report = REPORT_DIR / "hespi_v6_improvement_report.md"
    report.write_text("".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
