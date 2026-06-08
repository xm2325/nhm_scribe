from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from herbarium_scribe.evaluate import (
    field_exact_match,
    field_token_f1,
    normalize_field_value,
)


REPEATS = {
    "repeat_1": {
        "root": Path("data/experiments/hespi_v4_eval10_repeat_1"),
        "prefix": "hespi_v4_eval10_repeat_1",
        "llm": "hespi_v4_eval10_repeat_1_outputs.jsonl",
    },
    "repeat_2": {
        "root": Path("data/experiments/hespi_v4_eval10_repeat_2"),
        "prefix": "hespi_v4_eval10_repeat_2",
        "llm": "hespi_v4_eval10_repeat_2_outputs.jsonl",
    },
    "repeat_3": {
        "root": Path("data/experiments/hespi_v4_eval10_repeat_3"),
        "prefix": "hespi_v4_eval10_repeat_3",
        "llm": "hespi_v4_eval10_repeat_3_outputs.jsonl",
    },
}

AUTO_ACCEPT_FIELDS = {"catalogNumber", "typeStatus"}
TYPE_STATUS_VALUES = {
    "holotype",
    "isotype",
    "lectotype",
    "isolectotype",
    "neotype",
    "isoneotype",
    "syntype",
    "isosyntype",
    "paratype",
    "isoparatype",
    "type",
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


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(index=frame.index, dtype=float)), errors="coerce")


def usage_value(row: dict[str, Any], key: str) -> int:
    usage = row.get("response_usage", {})
    if not isinstance(usage, dict):
        return 0
    try:
        return int(usage.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def run_inputs(name: str, spec: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, str]]:
    detail_path = spec["root"] / "processed" / f"{spec['prefix']}_evaluation_detail.csv"
    detail = read_csv(detail_path)
    if detail.empty:
        raise SystemExit(f"{name}: missing evaluation detail: {detail_path}")
    detail = detail[
        ~detail["method"].eq("rule_ocr")
        & numeric(detail, "evaluable").eq(1)
    ].copy()
    detail["repeat"] = name

    rows = read_jsonl(spec["root"] / "interim" / "llm" / spec["llm"])
    if len(rows) != 10:
        raise SystemExit(f"{name}: expected 10 LLM rows, got {len(rows)}")
    prompt_hashes = {}
    for row in rows:
        messages = row.get("messages", [])
        stored = str(row.get("prompt_sha256", ""))
        messages_json = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        actual = hashlib.sha256(messages_json.encode("utf-8")).hexdigest()
        if not stored or stored != actual:
            raise SystemExit(f"{name}: invalid prompt SHA-256 for {row.get('occurrenceID')}")
        prompt_hashes[str(row.get("occurrenceID"))] = stored
    return detail, rows, prompt_hashes


def run_metric(name: str, detail: pd.DataFrame, rows: list[dict[str, Any]]) -> dict[str, Any]:
    filled = detail[detail["prediction"].astype(str).str.strip().ne("")]
    parsed = [
        row for row in rows
        if row.get("raw_output") and row.get("parse_failure") is False and not row.get("not_evaluated")
    ]
    return {
        "repeat": name,
        "records": detail["occurrenceID"].nunique(),
        "evaluable_field_units": len(detail),
        "coverage": numeric(detail, "coverage").mean(),
        "exact_match": numeric(detail, "exact_match").mean(),
        "token_f1": numeric(detail, "token_f1").mean(),
        "exact_match_among_filled": numeric(filled, "exact_match").mean(),
        "direct_evidence_support_rate": numeric(filled, "direct_evidence_supported").mean(),
        "unsupported_prediction_rate": numeric(filled, "unsupported_prediction").mean(),
        "review_required_rate": numeric(filled, "review_required").mean(),
        "parsed_records": len(parsed),
        "parse_success_rate": len(parsed) / len(rows) if rows else float("nan"),
        "prompt_tokens": sum(usage_value(row, "prompt_tokens") for row in rows),
        "completion_tokens": sum(usage_value(row, "completion_tokens") for row in rows),
        "total_tokens": sum(usage_value(row, "total_tokens") for row in rows),
    }


def _catalog_auto_accept_policy(
    occurrence_id: str,
    candidate: str,
    ocr_rows: pd.DataFrame | None,
) -> tuple[bool, str]:
    normalized = normalize_field_value("catalogNumber", candidate)
    if not re.search(r"[A-Z]", normalized) or not re.search(r"\d", normalized):
        return False, "catalog_requires_letters_and_digits"
    if ocr_rows is None or ocr_rows.empty:
        return False, "catalog_requires_targeted_region_support"
    record_rows = ocr_rows[ocr_rows["occurrenceID"].astype(str).eq(occurrence_id)]
    for _, row in record_rows.iterrows():
        if str(row.get("region_type", "")).strip() == "whole_sheet":
            continue
        region_text = normalize_field_value("catalogNumber", str(row.get("ocr_text", "")))
        if normalized and normalized in region_text:
            return True, ""
    return False, "catalog_requires_targeted_region_support"


def _auto_accept_policy(
    *,
    occurrence_id: str,
    field: str,
    candidate: str,
    ocr_rows: pd.DataFrame | None,
) -> tuple[bool, str]:
    if field not in AUTO_ACCEPT_FIELDS:
        return False, "field_requires_human_review"
    if field == "catalogNumber":
        return _catalog_auto_accept_policy(occurrence_id, candidate, ocr_rows)
    normalized = normalize_field_value(field, candidate)
    if field == "typeStatus" and normalized not in TYPE_STATUS_VALUES:
        return False, "unrecognized_type_status"
    return True, ""


def consensus_row(
    group: pd.DataFrame,
    ocr_rows: pd.DataFrame | None = None,
) -> dict[str, Any]:
    group = group.sort_values("repeat")
    values = group["normalised_prediction"].astype(str).tolist()
    counter = Counter(values)
    consensus_value, consensus_count = counter.most_common(1)[0]
    raw_match = group[group["normalised_prediction"].astype(str).eq(consensus_value)]
    consensus_prediction = str(raw_match.iloc[0]["prediction"]) if len(raw_match) else ""
    pair_matches = [
        int(values[left] == values[right])
        for left, right in combinations(range(len(values)), 2)
    ]
    filled = bool(consensus_value)
    direct_count = int(numeric(group, "direct_evidence_supported").fillna(0).sum())
    review_count = int(numeric(group, "review_required").fillna(0).sum())
    confidences = numeric(group, "prediction_confidence")
    unanimous = consensus_count == len(values)
    occurrence_id = str(group.iloc[0]["occurrenceID"])
    field = str(group.iloc[0]["field"])
    policy_pass, policy_reason = _auto_accept_policy(
        occurrence_id=occurrence_id,
        field=field,
        candidate=consensus_prediction,
        ocr_rows=ocr_rows,
    )
    auto_accept = (
        filled
        and unanimous
        and direct_count == len(values)
        and review_count == 0
        and confidences.notna().all()
        and confidences.min() >= 0.75
        and policy_pass
    )
    if auto_accept:
        consensus_action = "auto_accept"
        consensus_priority = ""
    elif not filled and unanimous:
        consensus_action = "empty_consensus"
        consensus_priority = ""
    else:
        consensus_action = "human_review"
        consensus_priority = (
            "high"
            if consensus_count == 1 or direct_count == 0 or field == "catalogNumber"
            else "medium"
        )
    gold = str(group.iloc[0]["gold"])
    return {
        "occurrenceID": occurrence_id,
        "field": field,
        "gold": gold,
        "normalised_gold": group.iloc[0]["normalised_gold"],
        "repeat_1_prediction": group.iloc[0]["prediction"],
        "repeat_2_prediction": group.iloc[1]["prediction"],
        "repeat_3_prediction": group.iloc[2]["prediction"],
        "consensus_prediction": consensus_prediction,
        "normalised_consensus_prediction": consensus_value,
        "consensus_count": consensus_count,
        "unanimous": unanimous,
        "pairwise_agreement": sum(pair_matches) / len(pair_matches),
        "filled_repeat_count": int(sum(bool(value) for value in values)),
        "direct_evidence_repeat_count": direct_count,
        "review_required_repeat_count": review_count,
        "minimum_confidence": confidences.min() if confidences.notna().any() else float("nan"),
        "mean_exact_match": numeric(group, "exact_match").mean(),
        "mean_token_f1": numeric(group, "token_f1").mean(),
        "consensus_exact_match": field_exact_match(field, consensus_prediction, gold),
        "consensus_token_f1": field_token_f1(field, consensus_prediction, gold),
        "auto_accept_policy_pass": policy_pass,
        "auto_accept_policy_reason": policy_reason,
        "consensus_action": consensus_action,
        "consensus_review_priority": consensus_priority,
    }


def markdown(frame: pd.DataFrame, max_rows: int = 200) -> str:
    return "_(empty)_\n" if frame.empty else frame.head(max_rows).to_markdown(index=False) + "\n"


def build_scale_gates(metrics: pd.DataFrame, overall: dict[str, Any]) -> pd.DataFrame:
    gates = [
        ("parse_success_rate", float(metrics["parse_success_rate"].min()), ">=", 0.95),
        ("pairwise_agreement_rate", float(overall["pairwise_agreement_rate"]), ">=", 0.90),
        (
            "mean_unsupported_prediction_rate",
            float(overall["mean_unsupported_prediction_rate"]),
            "<=",
            0.15,
        ),
        ("human_review_rate", float(overall["human_review_rate"]), "<=", 0.30),
        ("consensus_exact_match", float(overall["consensus_exact_match"]), ">=", 0.20),
        ("consensus_token_f1", float(overall["consensus_token_f1"]), ">=", 0.25),
        (
            "auto_accept_exact_match",
            float(overall["auto_accept_exact_match"]),
            ">=",
            0.95,
        ),
    ]
    rows = []
    for gate, value, operator, threshold in gates:
        passed = value >= threshold if operator == ">=" else value <= threshold
        rows.append({
            "gate": gate,
            "value": value,
            "operator": operator,
            "threshold": threshold,
            "passed": bool(passed),
        })
    return pd.DataFrame(rows)


def main() -> None:
    output = Path("reports/hespi_v4_repeat10")
    output.mkdir(parents=True, exist_ok=True)
    details = []
    run_metrics = []
    prompt_maps: dict[str, dict[str, str]] = {}

    for name, spec in REPEATS.items():
        detail, rows, prompt_hashes = run_inputs(name, spec)
        if detail["occurrenceID"].nunique() != 10:
            raise SystemExit(f"{name}: expected 10 evaluated records")
        details.append(detail)
        run_metrics.append(run_metric(name, detail, rows))
        prompt_maps[name] = prompt_hashes

    shared_ids = set.intersection(*(set(item) for item in prompt_maps.values()))
    if len(shared_ids) != 10:
        raise SystemExit(f"Only {len(shared_ids)} occurrenceIDs are shared by every repeat")
    prompt_mismatches = [
        occurrence_id
        for occurrence_id in sorted(shared_ids)
        if len({mapping[occurrence_id] for mapping in prompt_maps.values()}) != 1
    ]
    if prompt_mismatches:
        raise SystemExit(f"Prompt hashes differ across repeats: {prompt_mismatches}")

    combined = pd.concat(details, ignore_index=True)
    counts = combined.groupby(["occurrenceID", "field"])["repeat"].nunique()
    if not counts.eq(3).all():
        raise SystemExit("Not every occurrenceID/field unit is present in all three repeats")

    shared_ocr_path = (
        Path("data/experiments/hespi_v4_eval10_shared")
        / "processed"
        / "ocr_by_region.csv"
    )
    shared_ocr = read_csv(shared_ocr_path)
    stability_rows = [
        consensus_row(group, shared_ocr)
        for _, group in combined.groupby(["occurrenceID", "field"], sort=True)
    ]
    stability = pd.DataFrame(stability_rows)
    field_stability = stability.groupby("field", as_index=False).agg(
        evaluable_field_units=("occurrenceID", "count"),
        unanimous_prediction_rate=("unanimous", "mean"),
        pairwise_agreement_rate=("pairwise_agreement", "mean"),
        mean_exact_match=("mean_exact_match", "mean"),
        mean_token_f1=("mean_token_f1", "mean"),
        auto_accept_rate=("consensus_action", lambda values: (values == "auto_accept").mean()),
        human_review_rate=("consensus_action", lambda values: (values == "human_review").mean()),
    )
    review_queue = stability[stability["consensus_action"].eq("human_review")].copy()
    prompt_manifest = pd.DataFrame([
        {
            "occurrenceID": occurrence_id,
            "prompt_sha256": prompt_maps["repeat_1"][occurrence_id],
            "identical_across_repeats": True,
        }
        for occurrence_id in sorted(shared_ids)
    ])
    metrics = pd.DataFrame(run_metrics)
    filled_consensus = stability[
        stability["normalised_consensus_prediction"].astype(str).str.strip().ne("")
    ]
    auto_accept = stability[stability["consensus_action"].eq("auto_accept")]

    overall = {
        "records": len(shared_ids),
        "field_units": len(stability),
        "unanimous_prediction_rate": stability["unanimous"].mean(),
        "pairwise_agreement_rate": stability["pairwise_agreement"].mean(),
        "auto_accept_rate": stability["consensus_action"].eq("auto_accept").mean(),
        "human_review_rate": stability["consensus_action"].eq("human_review").mean(),
        "mean_exact_match": metrics["exact_match"].mean(),
        "mean_token_f1": metrics["token_f1"].mean(),
        "consensus_coverage": len(filled_consensus) / len(stability),
        "consensus_exact_match": stability["consensus_exact_match"].mean(),
        "consensus_token_f1": stability["consensus_token_f1"].mean(),
        "consensus_exact_match_among_filled": filled_consensus["consensus_exact_match"].mean(),
        "consensus_token_f1_among_filled": filled_consensus["consensus_token_f1"].mean(),
        "auto_accept_count": len(auto_accept),
        "auto_accept_exact_match": auto_accept["consensus_exact_match"].mean(),
        "auto_accept_token_f1": auto_accept["consensus_token_f1"].mean(),
        "wrong_auto_accept_count": int(auto_accept["consensus_exact_match"].eq(0).sum()),
        "mean_unsupported_prediction_rate": metrics["unsupported_prediction_rate"].mean(),
        "mean_review_required_rate": metrics["review_required_rate"].mean(),
        "total_llm_calls": int(metrics["parsed_records"].sum()),
        "total_tokens": int(metrics["total_tokens"].sum()),
    }
    overall_frame = pd.DataFrame([overall])
    scale_gates = build_scale_gates(metrics, overall)

    metrics.to_csv(output / "hespi_v4_repeat_metrics.csv", index=False)
    overall_frame.to_csv(output / "hespi_v4_stability_summary.csv", index=False)
    stability.to_csv(output / "hespi_v4_stability_detail.csv", index=False)
    field_stability.to_csv(output / "hespi_v4_field_stability.csv", index=False)
    review_queue.to_csv(output / "hespi_v4_consensus_review_queue.csv", index=False)
    prompt_manifest.to_csv(output / "hespi_v4_prompt_hash_manifest.csv", index=False)
    scale_gates.to_csv(output / "hespi_v4_scale_gates.csv", index=False)

    scale_ready = bool(scale_gates["passed"].all())
    lines = [
        "# Hespi v4 Eval10 Three-Repeat Stability Report\n",
        "\n## Experimental integrity\n",
        "- Ten fixed EVAL records used the same shared image, layout, OCR, and prompt evidence.\n",
        f"- Identical prompt SHA-256 across all repeats: `{not prompt_mismatches}`.\n",
        "- DeepSeek thinking was disabled and JSON mode was enabled.\n",
        "- RAG was disabled.\n",
        "\n## Repeat metrics\n",
        markdown(metrics),
        "\n## Stability summary\n",
        markdown(overall_frame),
        "\n## Field stability\n",
        markdown(field_stability),
        "\n## Consensus review queue\n",
        f"- Field units requiring human review: `{len(review_queue)}`.\n",
        markdown(review_queue, max_rows=40),
        "\n## Scale decision\n",
        f"- Ready to scale to 50 records: `{scale_ready}`.\n",
        markdown(scale_gates),
        "- Accuracy and auto-accept precision are required in addition to operational stability.\n",
        "\n## Limitations\n",
        "- Ten records and three repeats measure operational stability, not production accuracy.\n",
        "- Direct evidence checks OCR support, not whether OCR itself is correct.\n",
        "- Consensus does not make an unsupported prediction trustworthy; review gating remains necessary.\n",
    ]
    report = output / "hespi_v4_repeat10_report.md"
    report.write_text("".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
