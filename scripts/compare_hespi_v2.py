from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_PIPELINES = {
    "full_image": {
        "dir": "data/experiments/hespi_v2_full_image",
        "prefix": "hespi_v2_full_image",
        "llm": "hespi_v2_full_image_outputs.jsonl",
    },
    "field_only": {
        "dir": "data/experiments/hespi_v2_field_only",
        "prefix": "hespi_v2_field_only",
        "llm": "hespi_v2_field_only_outputs.jsonl",
    },
    "hybrid": {
        "dir": "data/experiments/hespi_v2_hybrid",
        "prefix": "hespi_v2_hybrid",
        "llm": "hespi_v2_hybrid_outputs.jsonl",
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


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def usage_value(row: dict[str, Any], key: str) -> int:
    usage = row.get("response_usage", {})
    if isinstance(usage, str):
        try:
            usage = json.loads(usage)
        except Exception:
            usage = {}
    try:
        return int(usage.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def paired_ids(pipelines: dict[str, dict[str, Any]], shared_dir: Path) -> tuple[set[str], pd.DataFrame]:
    shared = read_csv(shared_dir / "paired_eval_manifest.csv")
    eligible = shared[
        shared.get("split", "").eq("EVAL_SET")
        & shared.get("paired_eligible", "").map(truthy)
        & shared.get("sha256", "").astype(str).ne("")
    ].copy()
    ids = set(eligible["occurrenceID"])
    for name, spec in pipelines.items():
        manifest = read_csv(spec["dir"] / "processed" / f"{spec['prefix']}_image_manifest.csv")
        subset = manifest[manifest["occurrenceID"].isin(ids)][["occurrenceID", "sha256", "image_path"]]
        merged = eligible[["occurrenceID", "sha256"]].merge(
            subset,
            on="occurrenceID",
            how="left",
            suffixes=("_shared", f"_{name}"),
        )
        valid = merged[
            merged["sha256_shared"].eq(merged[f"sha256_{name}"])
            & merged["image_path"].astype(str).ne("")
        ]
        ids &= set(valid["occurrenceID"])
    return ids, eligible


def performance_metrics(detail: pd.DataFrame, ids: set[str], pipeline: str) -> tuple[dict[str, Any], pd.DataFrame]:
    llm = detail[
        detail["occurrenceID"].isin(ids)
        & detail["method"].eq("deepseek_v4_pro_no_rag")
        & detail["gold"].astype(str).ne("")
    ].copy()
    llm["filled"] = llm["prediction"].astype(str).str.strip().ne("")
    llm["exact_value"] = pd.to_numeric(llm["exact_match"], errors="coerce").fillna(0)
    llm["token_f1_value"] = pd.to_numeric(llm["token_f1"], errors="coerce").fillna(0)
    llm["parse_failure_value"] = pd.to_numeric(llm["parse_failure"], errors="coerce").fillna(0)
    field = llm.groupby("field", as_index=False).agg(
        evaluable_fields=("gold", "count"),
        coverage=("filled", "mean"),
        exact_match=("exact_value", "mean"),
        token_f1=("token_f1_value", "mean"),
    )
    field.insert(0, "pipeline", pipeline)
    filled = llm[llm["filled"]]
    record_failures = llm.drop_duplicates("occurrenceID")
    metrics = {
        "pipeline": pipeline,
        "paired_records": llm["occurrenceID"].nunique(),
        "evaluable_field_units": len(llm),
        "field_macro_coverage": field["coverage"].mean(),
        "field_macro_exact_match": field["exact_match"].mean(),
        "field_macro_token_f1": field["token_f1"].mean(),
        "field_micro_coverage": llm["filled"].mean(),
        "field_micro_exact_match": llm["exact_value"].mean(),
        "field_micro_token_f1": llm["token_f1_value"].mean(),
        "exact_match_among_filled": filled["exact_value"].mean() if len(filled) else float("nan"),
        "token_f1_among_filled": filled["token_f1_value"].mean() if len(filled) else float("nan"),
        "parse_failure_rate": record_failures["parse_failure_value"].mean(),
    }
    return metrics, field


def evidence_and_cost(spec: dict[str, Any], ids: set[str], pipeline: str) -> tuple[dict[str, Any], pd.DataFrame]:
    ocr = read_csv(spec["dir"] / "processed" / f"{spec['prefix']}_ocr_by_region.csv")
    ocr = ocr[ocr["occurrenceID"].isin(ids)].copy()
    ocr["text_length_value"] = pd.to_numeric(ocr["text_length"], errors="coerce").fillna(0)
    per_record = (
        ocr.groupby("occurrenceID")["text_length_value"]
        .sum()
        .reindex(sorted(ids), fill_value=0)
        .rename_axis("occurrenceID")
        .reset_index()
    )
    proxy = read_csv(spec["dir"] / "processed" / "ocr_proxy_field_presence.csv")
    proxy = proxy[proxy["occurrenceID"].isin(ids)].copy() if not proxy.empty else proxy
    proxy_scores = pd.to_numeric(
        proxy.get("ocr_evidence_proxy_score", pd.Series(dtype=float)),
        errors="coerce",
    )
    empty_ids = set(per_record.loc[per_record["text_length_value"].eq(0), "occurrenceID"])
    rows = [row for row in read_jsonl(spec["dir"] / "interim" / "llm" / spec["llm"]) if row.get("occurrenceID") in ids]
    attempted = [row for row in rows if bool(row.get("llm_call_attempted", True))]
    parsed = [
        row for row in attempted
        if bool(row.get("raw_output")) and row.get("parse_failure") is False and not bool(row.get("not_evaluated"))
    ]
    total_tokens = sum(usage_value(row, "total_tokens") for row in attempted)
    channel = ocr.groupby("evidence_source", as_index=False).agg(
        regions=("region_id", "count"),
        records=("occurrenceID", "nunique"),
        characters=("text_length_value", "sum"),
        mean_characters_per_region=("text_length_value", "mean"),
    )
    channel.insert(0, "pipeline", pipeline)
    return {
        "pipeline": pipeline,
        "ocr_characters_per_record": per_record["text_length_value"].mean(),
        "mean_ocr_evidence_proxy": proxy_scores.mean() if proxy_scores.notna().any() else float("nan"),
        "empty_ocr_records": len(empty_ids),
        "llm_calls_attempted": len(attempted),
        "llm_calls_skipped": len(rows) - len(attempted),
        "empty_ocr_llm_calls": sum(row.get("occurrenceID") in empty_ids for row in attempted),
        "parsed_records": len(parsed),
        "prompt_tokens": sum(usage_value(row, "prompt_tokens") for row in attempted),
        "completion_tokens": sum(usage_value(row, "completion_tokens") for row in attempted),
        "total_tokens": total_tokens,
        "tokens_per_parsed_record": total_tokens / len(parsed) if parsed else float("nan"),
    }, channel


def markdown(df: pd.DataFrame, max_rows: int = 200) -> str:
    return "_(empty)_\n" if df.empty else df.head(max_rows).to_markdown(index=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-dir", default="data/experiments/hespi_v2_shared/processed")
    parser.add_argument("--full-image-dir", default=DEFAULT_PIPELINES["full_image"]["dir"])
    parser.add_argument("--field-only-dir", default=DEFAULT_PIPELINES["field_only"]["dir"])
    parser.add_argument("--hybrid-dir", default=DEFAULT_PIPELINES["hybrid"]["dir"])
    parser.add_argument("--output-dir", default="reports/hespi_v2")
    args = parser.parse_args()
    shared_dir = Path(args.shared_dir)
    output = Path(args.output_dir)
    pipelines = {
        name: {
            **spec,
            "dir": Path(getattr(args, f"{name}_dir")),
        }
        for name, spec in DEFAULT_PIPELINES.items()
    }
    output.mkdir(parents=True, exist_ok=True)
    ids, eligible = paired_ids(pipelines, shared_dir)
    if len(ids) < 20:
        raise SystemExit(f"Only {len(ids)} SHA-matched records are available; at least 20 are required.")

    performance_rows = []
    field_frames = []
    cost_rows = []
    channel_frames = []
    fallback_frames = []
    availability_rows = []
    requested = len(read_csv(shared_dir / "eval_set.csv"))
    demo_records = len(read_csv(shared_dir / "demo_set.csv"))

    for name, spec in pipelines.items():
        manifest = read_csv(spec["dir"] / "processed" / f"{spec['prefix']}_image_manifest.csv")
        success = manifest[manifest["occurrenceID"].isin(set(eligible["occurrenceID"]))]["image_path"].astype(str).ne("").sum()
        availability_rows.append({
            "pipeline": name,
            "demo_records": demo_records,
            "requested_eval_records": requested,
            "successful_images": int(success),
            "sha256_matched_paired_records": len(ids),
            "excluded_records": requested - len(ids),
        })
        detail = read_csv(spec["dir"] / "processed" / f"{spec['prefix']}_evaluation_detail.csv")
        metrics, field = performance_metrics(detail, ids, name)
        performance_rows.append(metrics)
        field_frames.append(field)
        cost, channels = evidence_and_cost(spec, ids, name)
        cost_rows.append(cost)
        channel_frames.append(channels)
        diagnostic_path = spec["dir"] / "processed" / f"{spec['prefix']}_hespi_layout_diagnostics.csv"
        diagnostics = read_csv(diagnostic_path)
        if diagnostics.empty:
            fallback_frames.append(pd.DataFrame([{
                "pipeline": name,
                "fallback_reason": "not_applicable",
                "records": len(ids),
            }]))
        else:
            diagnostics = diagnostics[diagnostics["occurrenceID"].isin(ids)].copy()
            diagnostics["fallback_reason"] = diagnostics["fallback_reason"].replace("", "none")
            fallback = diagnostics.groupby("fallback_reason", as_index=False).agg(
                records=("occurrenceID", "nunique"),
            )
            fallback.insert(0, "pipeline", name)
            fallback_frames.append(fallback)

    availability = pd.DataFrame(availability_rows)
    performance = pd.DataFrame(performance_rows)
    field_metrics = pd.concat(field_frames, ignore_index=True)
    evidence_cost = pd.DataFrame(cost_rows)
    channels = pd.concat(channel_frames, ignore_index=True)
    fallbacks = pd.concat(fallback_frames, ignore_index=True)

    baseline = field_metrics[field_metrics["pipeline"] == "full_image"]
    deltas = []
    for candidate in ["field_only", "hybrid"]:
        candidate_fields = field_metrics[field_metrics["pipeline"] == candidate]
        merged = baseline.merge(candidate_fields, on="field", suffixes=("_full_image", f"_{candidate}"))
        for _, row in merged.iterrows():
            deltas.append({
                "candidate": candidate,
                "field": row["field"],
                "coverage_delta": row[f"coverage_{candidate}"] - row["coverage_full_image"],
                "exact_match_delta": row[f"exact_match_{candidate}"] - row["exact_match_full_image"],
                "token_f1_delta": row[f"token_f1_{candidate}"] - row["token_f1_full_image"],
            })
    field_deltas = pd.DataFrame(deltas)

    availability.to_csv(output / "hespi_v2_data_availability.csv", index=False)
    performance.to_csv(output / "hespi_v2_performance.csv", index=False)
    field_metrics.to_csv(output / "hespi_v2_field_metrics.csv", index=False)
    field_deltas.to_csv(output / "hespi_v2_field_deltas.csv", index=False)
    evidence_cost.to_csv(output / "hespi_v2_evidence_and_cost.csv", index=False)
    channels.to_csv(output / "hespi_v2_ocr_channels.csv", index=False)
    fallbacks.to_csv(output / "hespi_v2_fallback_reasons.csv", index=False)

    hybrid = performance[performance["pipeline"] == "hybrid"].iloc[0]
    full = performance[performance["pipeline"] == "full_image"].iloc[0]
    delta = hybrid["field_micro_token_f1"] - full["field_micro_token_f1"]
    verdict = (
        f"Hybrid improved paired micro token F1 by {delta:.3f}."
        if delta > 0 else
        f"Hybrid reduced paired micro token F1 by {abs(delta):.3f}."
        if delta < 0 else
        "Hybrid made no paired micro token F1 difference."
    )
    gate_ok = evidence_cost["empty_ocr_llm_calls"].eq(0).all()
    lines = [
        "# Hespi v2 Strict Paired Comparison\n",
        "\n## Experimental design\n",
        "- A shared acquisition stage downloaded, validated, and SHA-256 hashed all images.\n",
        "- Twenty successful EVAL images were selected with fixed-seed stratified sampling.\n",
        "- Full-image, field-only, and hybrid pipelines read the same immutable local files.\n",
        f"- SHA-256 matched paired records: `{len(ids)}`.\n",
        "- DEMO records are excluded from all performance summaries below.\n",
        "\n## Data availability\n",
        markdown(availability),
        "\n## Extraction performance\n",
        markdown(performance),
        "\n## Evidence quality and cost\n",
        markdown(evidence_cost),
        f"\n- Empty-OCR LLM call gate passed: `{gate_ok}`.\n",
        "\n## OCR channels\n",
        markdown(channels),
        "\n## Layout fallback reasons\n",
        markdown(fallbacks),
        "\n## Field metrics\n",
        markdown(field_metrics),
        "\n## Field deltas against full-image baseline\n",
        markdown(field_deltas),
        "\n## Verdict\n",
        verdict + "\n",
        "\n## Guardrails\n",
        "- Macro metrics weight fields equally; micro metrics weight evaluable field units equally.\n",
        "- Filled-field metrics measure correctness only where the pipeline returned a value.\n",
        "- The OCR evidence proxy is not CER or WER.\n",
        "- No claim that Hespi improves extraction should be made unless the strict paired hybrid metrics support it.\n",
    ]
    (output / "hespi_v2_paired_report.md").write_text("".join(lines), encoding="utf-8")
    print(output / "hespi_v2_paired_report.md")


if __name__ == "__main__":
    main()
