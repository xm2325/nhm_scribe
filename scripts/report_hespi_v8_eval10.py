from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from herbarium_scribe.evaluate import field_token_f1


ROOT = Path("data/experiments/hespi_v8_eval10")
PROCESSED = ROOT / "processed"
LLM = ROOT / "interim" / "llm"
REPORT_DIR = Path("reports/hespi_v8_eval10")
METHOD = "deepseek_v4_pro_nonthinking_htr"
TARGET_REGION_TYPES = {"collector", "year", "month", "day"}
V7_REFERENCE = {
    "run_id": 27197202840,
    "coverage": 0.30357142857142855,
    "exact_match": 0.19642857142857142,
    "token_f1": 0.20535714285714285,
    "recorded_by_token_f1": 0.05,
    "event_date_token_f1": 0.0,
}
MONTH_ALIASES = {
    1: {"1", "01", "jan", "january"},
    2: {"2", "02", "feb", "february"},
    3: {"3", "03", "mar", "march"},
    4: {"4", "04", "apr", "april"},
    5: {"5", "05", "may"},
    6: {"6", "06", "jun", "june"},
    7: {"7", "07", "jul", "july"},
    8: {"8", "08", "aug", "august"},
    9: {"9", "09", "sep", "sept", "september"},
    10: {"10", "oct", "october"},
    11: {"11", "nov", "november"},
    12: {"12", "dec", "december"},
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("") if path.exists() else pd.DataFrame()


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(
        frame.get(column, pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    )


def markdown(frame: pd.DataFrame, max_rows: int = 50) -> str:
    return "_(empty)_\n" if frame.empty else frame.head(max_rows).to_markdown(index=False) + "\n"


def engine_family(value: str) -> str:
    return "trocr" if str(value).startswith("hespi_trocr_") else "tesseract"


def date_component_match(region_type: str, text: str, event_date: str) -> bool | None:
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(event_date))
    if not match:
        return None
    year, month, day = [int(value) for value in match.groups()]
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z]+|\d+", str(text))
    }
    if region_type == "year":
        return str(year) in tokens
    if region_type == "month":
        return bool(tokens & MONTH_ALIASES[month])
    if region_type == "day":
        return str(day) in tokens or f"{day:02d}" in tokens
    return None


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

    eval_set = read_csv(PROCESSED / "eval_set.csv")
    eval_ids = set(eval_set.get("occurrenceID", pd.Series(dtype=str)).astype(str))
    gold = eval_set.set_index("occurrenceID") if not eval_set.empty else pd.DataFrame()
    ocr = read_csv(PROCESSED / "ocr_by_region.csv")
    if eval_ids and not ocr.empty:
        ocr = ocr[ocr["occurrenceID"].astype(str).isin(eval_ids)].copy()
    target = ocr[ocr.get("region_type", pd.Series(dtype=str)).isin(TARGET_REGION_TYPES)].copy()
    target["engine_family"] = target.get("ocr_engine", pd.Series(dtype=str)).map(engine_family)
    target["base_region_id"] = target.apply(
        lambda row: (
            row.get("htr_source_region_id", "")
            if row.get("engine_family") == "trocr"
            else row.get("region_id", "")
        ),
        axis=1,
    )
    paired = target.pivot_table(
        index=["occurrenceID", "base_region_id", "region_type"],
        columns="engine_family",
        values="ocr_text",
        aggfunc=lambda values: " | ".join(
            value for value in dict.fromkeys(str(item) for item in values) if value
        ),
        fill_value="",
    ).reset_index()
    paired.columns.name = None
    for column in ("tesseract", "trocr"):
        if column not in paired.columns:
            paired[column] = ""

    evidence_rows = []
    for _, row in paired.iterrows():
        occurrence_id = str(row["occurrenceID"])
        region_type = str(row["region_type"])
        gold_row = gold.loc[occurrence_id] if occurrence_id in gold.index else {}
        if region_type == "collector":
            gold_value = str(gold_row.get("recordedBy", ""))
            for family in ("tesseract", "trocr"):
                text = str(row.get(family, ""))
                evidence_rows.append({
                    "occurrenceID": occurrence_id,
                    "region_type": region_type,
                    "engine_family": family,
                    "gold": gold_value,
                    "text": text,
                    "component_match": (
                        field_token_f1("recordedBy", text, gold_value)
                        if text and gold_value
                        else 0.0
                    ),
                })
        else:
            gold_value = str(gold_row.get("eventDate", ""))
            for family in ("tesseract", "trocr"):
                text = str(row.get(family, ""))
                matched = date_component_match(region_type, text, gold_value)
                if matched is not None:
                    evidence_rows.append({
                        "occurrenceID": occurrence_id,
                        "region_type": region_type,
                        "engine_family": family,
                        "gold": gold_value,
                        "text": text,
                        "component_match": float(matched),
                    })
    evidence = pd.DataFrame(evidence_rows)
    if not evidence.empty:
        evidence_summary = evidence.groupby(
            ["engine_family", "region_type"],
            as_index=False,
        ).agg(
            component_match_rate=("component_match", "mean"),
            nonempty_rate=("text", lambda values: values.astype(str).str.strip().ne("").mean()),
            n=("occurrenceID", "size"),
        )
    else:
        evidence_summary = pd.DataFrame()

    htr = target[target["engine_family"].eq("trocr")].copy()
    output_path = LLM / "hespi_v8_eval10_outputs.jsonl"
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
    field_lookup = field_metrics.set_index("field") if not field_metrics.empty else pd.DataFrame()
    recorded_by_f1 = (
        float(field_lookup.loc["recordedBy", "token_f1"])
        if "recordedBy" in field_lookup.index
        else float("nan")
    )
    event_date_f1 = (
        float(field_lookup.loc["eventDate", "token_f1"])
        if "eventDate" in field_lookup.index
        else float("nan")
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
        "recorded_by_token_f1": recorded_by_f1,
        "event_date_token_f1": event_date_f1,
        "htr_regions": len(htr),
        "htr_nonempty_regions": int(htr["ocr_text"].astype(str).str.strip().ne("").sum()) if len(htr) else 0,
        "htr_error_regions": int(
            htr["ocr_status"]
            .astype(str)
            .str.startswith(("error:", "missing", "unavailable:", "model_load_error:"))
            .sum()
        ) if len(htr) else 0,
        "htr_elapsed_seconds": numeric(htr, "htr_elapsed_seconds").fillna(0).sum() if len(htr) else 0.0,
        "parsed_records": len(parsed),
        "parse_success_rate": len(parsed) / len(outputs) if outputs else float("nan"),
        "total_tokens": total_tokens,
    }])
    summary["coverage_delta_vs_v7"] = summary["coverage"] - V7_REFERENCE["coverage"]
    summary["exact_match_delta_vs_v7"] = summary["exact_match"] - V7_REFERENCE["exact_match"]
    summary["token_f1_delta_vs_v7"] = summary["token_f1"] - V7_REFERENCE["token_f1"]
    summary["recorded_by_f1_delta_vs_v7"] = (
        summary["recorded_by_token_f1"] - V7_REFERENCE["recorded_by_token_f1"]
    )
    summary["event_date_f1_delta_vs_v7"] = (
        summary["event_date_token_f1"] - V7_REFERENCE["event_date_token_f1"]
    )

    summary.to_csv(REPORT_DIR / "hespi_v8_summary.csv", index=False)
    field_metrics.to_csv(REPORT_DIR / "hespi_v8_field_metrics.csv", index=False)
    paired.to_csv(REPORT_DIR / "hespi_v8_htr_tesseract_pairs.csv", index=False)
    evidence.to_csv(REPORT_DIR / "hespi_v8_htr_evidence_detail.csv", index=False)
    evidence_summary.to_csv(REPORT_DIR / "hespi_v8_htr_evidence_summary.csv", index=False)

    row = summary.iloc[0]
    scale_ready = (
        float(row["parse_success_rate"]) >= 0.95
        and float(row["exact_match"]) >= 0.20
        and float(row["token_f1"]) >= 0.25
        and float(row["unsupported_prediction_rate"]) <= 0.15
        and float(row["review_required_rate"]) <= 0.30
    )
    lines = [
        "# Hespi v8 Eval10 Supplementary HTR Report\n",
        "\n## Experiment\n",
        "- The EVAL set, whole-sheet OCR, barcode decoder, catalogue resolver, GBIF checks, and DeepSeek settings are unchanged from v7.\n",
        "- Hespi label-field detection adds only collector, year, month, and day crops.\n",
        "- Each crop keeps its Tesseract text and receives a supplementary Microsoft TrOCR-small handwritten reading.\n",
        "- TrOCR output is a hypothesis, not an automatic replacement or source of filled values.\n",
        f"- Historical reference: Hespi v7, GitHub Actions run `{V7_REFERENCE['run_id']}`.\n",
        "\n## Summary\n",
        markdown(summary),
        "\n## Final field metrics\n",
        markdown(field_metrics),
        "\n## HTR evidence proxy by engine and field component\n",
        markdown(evidence_summary),
        "\n## Paired crop readings\n",
        markdown(paired),
        "\n## Scale decision\n",
        f"- Ready for a larger evaluation: `{scale_ready}`.\n",
        "- Gates: parse success >=95%, exact match >=20%, token F1 >=25%, unsupported predictions <=15%, human review <=30%.\n",
        "\n## Interpretation guardrails\n",
        "- Component-match rates measure whether the gold collector/date component appears in crop text; they are not true HTR CER/WER.\n",
        "- A field-level improvement must appear in the final recordedBy or eventDate metrics, not only in one attractive crop example.\n",
        "- Ten records are a debugging sample, not a production performance estimate.\n",
    ]
    report = REPORT_DIR / "hespi_v8_improvement_report.md"
    report.write_text("".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
