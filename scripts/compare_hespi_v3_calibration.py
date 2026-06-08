from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PIPELINES = {
    "thinking_hybrid": {
        "root": Path("data/experiments/hespi_v3_thinking_hybrid"),
        "prefix": "hespi_v3_thinking_hybrid",
        "llm": "hespi_v3_thinking_hybrid_outputs.jsonl",
    },
    "nonthinking_hybrid": {
        "root": Path("data/experiments/hespi_v3_nonthinking_hybrid"),
        "prefix": "hespi_v3_nonthinking_hybrid",
        "llm": "hespi_v3_nonthinking_hybrid_outputs.jsonl",
    },
    "nonthinking_lean": {
        "root": Path("data/experiments/hespi_v3_nonthinking_lean"),
        "prefix": "hespi_v3_nonthinking_lean",
        "llm": "hespi_v3_nonthinking_lean_outputs.jsonl",
    },
}


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


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def usage_value(row: dict[str, Any], key: str) -> int:
    usage = row.get("response_usage", {})
    if not isinstance(usage, dict):
        return 0
    return int(number(usage.get(key, 0)))


def reasoning_tokens(row: dict[str, Any]) -> int:
    usage = row.get("response_usage", {})
    if not isinstance(usage, dict):
        return 0
    details = usage.get("completion_tokens_details", {})
    if not isinstance(details, dict):
        return 0
    return int(number(details.get("reasoning_tokens", 0)))


def pipeline_metrics(name: str, spec: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    detail = read_csv(spec["root"] / "processed" / f"{spec['prefix']}_evaluation_detail.csv")
    if detail.empty:
        return {}, pd.DataFrame()
    llm = detail[~detail["method"].eq("rule_ocr")].copy()
    llm = llm[pd.to_numeric(llm["evaluable"], errors="coerce").fillna(0).eq(1)]
    for column in [
        "coverage",
        "exact_match",
        "token_f1",
        "direct_evidence_supported",
        "unsupported_prediction",
    ]:
        if column in llm:
            llm[column] = pd.to_numeric(llm[column], errors="coerce")
    filled = llm[llm["prediction"].astype(str).str.strip().ne("")]
    rows = read_jsonl(spec["root"] / "interim" / "llm" / spec["llm"])
    attempted = [row for row in rows if bool(row.get("llm_call_attempted", True))]
    parsed = [
        row for row in attempted
        if row.get("raw_output") and row.get("parse_failure") is False and not row.get("not_evaluated")
    ]
    prompt_tokens = sum(usage_value(row, "prompt_tokens") for row in attempted)
    completion_tokens = sum(usage_value(row, "completion_tokens") for row in attempted)
    thought_tokens = sum(reasoning_tokens(row) for row in attempted)
    total_tokens = sum(usage_value(row, "total_tokens") for row in attempted)
    metrics = {
        "pipeline": name,
        "records": llm["occurrenceID"].nunique(),
        "evaluable_field_units": len(llm),
        "coverage": llm["coverage"].mean(),
        "exact_match": llm["exact_match"].mean(),
        "token_f1": llm["token_f1"].mean(),
        "exact_match_among_filled": filled["exact_match"].mean() if len(filled) else float("nan"),
        "direct_evidence_support_rate": filled["direct_evidence_supported"].mean() if len(filled) else float("nan"),
        "unsupported_prediction_rate": filled["unsupported_prediction"].mean() if len(filled) else float("nan"),
        "llm_calls": len(attempted),
        "parsed_records": len(parsed),
        "parse_success_rate": len(parsed) / len(attempted) if attempted else float("nan"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": thought_tokens,
        "visible_output_tokens": max(0, completion_tokens - thought_tokens),
        "total_tokens": total_tokens,
        "tokens_per_parsed_record": total_tokens / len(parsed) if parsed else float("nan"),
    }
    return metrics, llm


def comparison_row(metrics: pd.DataFrame, candidate: str, baseline: str) -> dict[str, Any] | None:
    if candidate not in set(metrics["pipeline"]) or baseline not in set(metrics["pipeline"]):
        return None
    left = metrics.set_index("pipeline").loc[candidate]
    right = metrics.set_index("pipeline").loc[baseline]
    baseline_tokens = number(right["total_tokens"])
    return {
        "candidate": candidate,
        "baseline": baseline,
        "coverage_delta": number(left["coverage"]) - number(right["coverage"]),
        "exact_match_delta": number(left["exact_match"]) - number(right["exact_match"]),
        "token_f1_delta": number(left["token_f1"]) - number(right["token_f1"]),
        "unsupported_prediction_rate_delta": (
            number(left["unsupported_prediction_rate"]) - number(right["unsupported_prediction_rate"])
        ),
        "total_token_reduction": (
            1 - number(left["total_tokens"]) / baseline_tokens
            if baseline_tokens else float("nan")
        ),
    }


def markdown(frame: pd.DataFrame) -> str:
    return "_(not run)_\n" if frame.empty else frame.to_markdown(index=False) + "\n"


def main() -> None:
    output = Path("reports/hespi_v3_calibration")
    output.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    details: dict[str, pd.DataFrame] = {}
    for name, spec in PIPELINES.items():
        metrics, detail = pipeline_metrics(name, spec)
        if metrics:
            metric_rows.append(metrics)
            details[name] = detail
    if not metric_rows:
        raise SystemExit("No Hespi v3 calibration outputs were found.")

    metrics = pd.DataFrame(metric_rows)
    comparisons = pd.DataFrame([
        row
        for row in [
            comparison_row(metrics, "nonthinking_hybrid", "thinking_hybrid"),
            comparison_row(metrics, "nonthinking_lean", "nonthinking_hybrid"),
        ]
        if row is not None
    ])
    shared_ids = set.intersection(
        *(set(frame["occurrenceID"]) for frame in details.values())
    ) if details else set()
    if len(details) > 1:
        expected_sizes = {name: frame["occurrenceID"].nunique() for name, frame in details.items()}
        if any(size != len(shared_ids) for size in expected_sizes.values()):
            raise SystemExit(
                f"Calibration pipelines did not use identical occurrenceIDs: "
                f"shared={len(shared_ids)}, sizes={expected_sizes}"
            )
    selected_ids = pd.DataFrame({"occurrenceID": sorted(shared_ids)})

    metrics.to_csv(output / "hespi_v3_calibration_metrics.csv", index=False)
    comparisons.to_csv(output / "hespi_v3_calibration_deltas.csv", index=False)
    selected_ids.to_csv(output / "hespi_v3_shared_occurrence_ids.csv", index=False)

    lines = [
        "# Hespi v3 Five-Record Calibration\n",
        "\n## Scope\n",
        f"- Pipelines found: `{', '.join(metrics['pipeline'])}`.\n",
        f"- occurrenceIDs shared by every completed pipeline: `{len(shared_ids)}`.\n",
        "- Results use field-specific normalization for catalogue numbers, dates, collector names, coordinates, and type status.\n",
        "- Direct evidence support checks both OCR occurrence and prediction/span alignment; it is not a semantic entailment score.\n",
        "\n## Metrics\n",
        markdown(metrics),
        "\n## Comparisons\n",
        markdown(comparisons),
        "\n## Decision rules\n",
        "- Prefer non-thinking if total tokens fall by at least 75% and token F1 falls by no more than 0.02.\n",
        "- Prefer lean hybrid if it stays within 0.01 token F1 of full hybrid while reducing prompt tokens by at least 25%.\n",
        "- Do not scale when parsed-record rate is below 95% or unsupported prediction rate exceeds 15%.\n",
        "\n## Limitations\n",
        "- Five records are a calibration smoke test, not a performance claim.\n",
        "- Thinking-mode variability requires repeated calls before making a final model-setting decision.\n",
        "- RAG is not included in this calibration.\n",
    ]
    report = output / "hespi_v3_calibration_report.md"
    report.write_text("".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
