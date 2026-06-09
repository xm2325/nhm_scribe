from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from herbarium_scribe.evaluate import field_token_f1


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


def date_component_match(region_type: str, text: str, event_date: str) -> float | None:
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(event_date))
    if not match:
        return None
    year, month, day = [int(value) for value in match.groups()]
    tokens = {token.lower() for token in re.findall(r"[A-Za-z]+|\d+", str(text))}
    if region_type == "year":
        return float(str(year) in tokens)
    if region_type == "month":
        return float(bool(tokens & MONTH_ALIASES[month]))
    if region_type == "day":
        return float(str(day) in tokens or f"{day:02d}" in tokens)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--label", default="TrOCR base")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    processed = data_dir / "processed"
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    eval_set = pd.read_csv(processed / "eval_set.csv", dtype=str).fillna("")
    gold = eval_set.set_index("occurrenceID")
    eval_ids = set(eval_set["occurrenceID"].astype(str))
    ocr = pd.read_csv(processed / "ocr_by_region.csv", dtype=str).fillna("")
    target = ocr[
        ocr["occurrenceID"].astype(str).isin(eval_ids)
        & ocr["region_type"].astype(str).isin({"collector", "year", "month", "day"})
    ].copy()
    target["engine_family"] = target["ocr_engine"].astype(str).map(
        lambda value: "trocr" if value.startswith("hespi_trocr_") else "tesseract"
    )
    target["base_region_id"] = target.apply(
        lambda row: (
            row["htr_source_region_id"]
            if row["engine_family"] == "trocr"
            else row["region_id"]
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
        if column not in paired:
            paired[column] = ""

    evidence_rows = []
    for _, row in paired.iterrows():
        occurrence_id = str(row["occurrenceID"])
        region_type = str(row["region_type"])
        gold_row = gold.loc[occurrence_id]
        for engine in ("tesseract", "trocr"):
            text = str(row[engine])
            if region_type == "collector":
                score = field_token_f1("recordedBy", text, str(gold_row.get("recordedBy", "")))
            else:
                score = date_component_match(
                    region_type,
                    text,
                    str(gold_row.get("eventDate", "")),
                )
            if score is not None:
                evidence_rows.append({
                    "occurrenceID": occurrence_id,
                    "region_type": region_type,
                    "engine_family": engine,
                    "text": text,
                    "component_match": score,
                })
    evidence = pd.DataFrame(evidence_rows)
    summary = evidence.groupby(
        ["engine_family", "region_type"],
        as_index=False,
    ).agg(
        component_match_rate=("component_match", "mean"),
        nonempty_rate=("text", lambda values: values.astype(str).str.strip().ne("").mean()),
        n=("occurrenceID", "size"),
    )
    htr = target[target["engine_family"].eq("trocr")].copy()
    accepted = htr["htr_prompt_accepted"].astype(str).str.lower().eq("true")
    run_summary = pd.DataFrame([{
        "records": len(eval_set),
        "htr_regions": len(htr),
        "htr_nonempty_regions": htr["ocr_text"].astype(str).str.strip().ne("").sum(),
        "htr_prompt_accepted_regions": accepted.sum(),
        "htr_prompt_acceptance_rate": accepted.mean(),
        "htr_elapsed_seconds": pd.to_numeric(
            htr["htr_elapsed_seconds"],
            errors="coerce",
        ).fillna(0).sum(),
        "llm_calls": 0,
    }])

    paired.to_csv(report_dir / "htr_tesseract_pairs.csv", index=False)
    evidence.to_csv(report_dir / "htr_evidence_detail.csv", index=False)
    summary.to_csv(report_dir / "htr_evidence_summary.csv", index=False)
    run_summary.to_csv(report_dir / "htr_probe_summary.csv", index=False)
    report = report_dir / "htr_probe_report.md"
    report.write_text(
        "\n".join([
            f"# {args.label} Eval10 HTR Probe",
            "",
            "This probe evaluates handwriting OCR evidence only. It makes no LLM calls.",
            "",
            "## Run summary",
            run_summary.to_markdown(index=False),
            "",
            "## Evidence proxy",
            summary.to_markdown(index=False),
            "",
            "## Paired readings",
            paired.to_markdown(index=False),
            "",
            "Component match is a field-specific evidence proxy, not true HTR CER/WER.",
            "",
        ]),
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()
