from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from herbarium_scribe.config import load_config
from herbarium_scribe.pipeline import stage_download, stage_metadata
from herbarium_scribe.sampling import save_split_outputs, stratified_random_sample


def split_summary(demo: pd.DataFrame, eval_df: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for split_name, frame in [("DEMO_SET", demo), ("EVAL_SET", eval_df)]:
        counts = frame[by].value_counts(dropna=False).to_dict() if by in frame.columns else {"all": len(frame)}
        for stratum, count in counts.items():
            rows.append({
                "split": split_name,
                "stratify_by": by,
                "stratum": stratum,
                "n": int(count),
                "split_mode": "frozen_after_validated_download",
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hespi_v2_acquisition.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(cfg["paths"]["data_dir"])
    processed = data_dir / "processed"
    target_size = int(cfg.get("paired_eval", {}).get("target_eval_size", 20))
    by = cfg.get("sampling", {}).get("stratify_by", "institutionCode")
    seed = int(cfg.get("project", {}).get("random_state", 42))

    demo, candidate_eval, _ = stage_metadata(args.config)
    manifest = stage_download(args.config)
    candidate_eval.to_csv(processed / "candidate_eval_set.csv", index=False)

    eval_manifest = candidate_eval[["occurrenceID"]].merge(
        manifest,
        on="occurrenceID",
        how="left",
    )
    valid_ids = set(
        eval_manifest.loc[
            eval_manifest["image_path"].astype(str).ne("")
            & eval_manifest["sha256"].astype(str).ne(""),
            "occurrenceID",
        ]
    )
    successful = candidate_eval[candidate_eval["occurrenceID"].isin(valid_ids)].copy()
    if len(successful) < target_size:
        raise SystemExit(
            f"Only {len(successful)} validated EVAL images were downloaded; "
            f"at least {target_size} are required for the paired pilot."
        )

    frozen_eval = stratified_random_sample(successful, target_size, by, seed + 101).reset_index(drop=True)
    summary = split_summary(demo, frozen_eval, by)
    save_split_outputs(demo, frozen_eval, summary, processed)
    prefix = str(cfg.get("outputs", {}).get("prefix", "") or "").strip()
    if prefix:
        demo.to_csv(processed / f"{prefix}_demo_set.csv", index=False)
        frozen_eval.to_csv(processed / f"{prefix}_eval_set.csv", index=False)
        summary.to_csv(processed / f"{prefix}_split_summary.csv", index=False)

    demo_ids = set(demo["occurrenceID"])
    eval_ids = set(frozen_eval["occurrenceID"])
    paired = manifest.copy()
    paired["split"] = paired["occurrenceID"].map(
        lambda value: "DEMO_SET" if value in demo_ids else ("EVAL_SET" if value in eval_ids else "CANDIDATE_EXCLUDED")
    )
    paired["selected_for_paired_eval"] = paired["occurrenceID"].isin(eval_ids)
    paired["paired_eligible"] = (
        paired["selected_for_paired_eval"]
        & paired["image_path"].astype(str).ne("")
        & paired["sha256"].astype(str).ne("")
    )
    paired["excluded_reason"] = paired.apply(
        lambda row: ""
        if row["paired_eligible"]
        else (
            str(row.get("download_error", "") or row.get("image_status", ""))
            if not str(row.get("image_path", ""))
            else ("demo_record" if row["split"] == "DEMO_SET" else "not_selected_after_download")
        ),
        axis=1,
    )
    paired.to_csv(processed / "paired_eval_manifest.csv", index=False)
    if prefix:
        paired.to_csv(processed / f"{prefix}_paired_eval_manifest.csv", index=False)
    frozen_eval[["occurrenceID"]].merge(
        paired,
        on="occurrenceID",
        how="left",
    ).to_csv(processed / "paired_eval20_manifest.csv", index=False)

    print(f"Candidate EVAL records: {len(candidate_eval)}")
    print(f"Validated downloaded EVAL images: {len(successful)}")
    print(f"Frozen paired EVAL records: {len(frozen_eval)}")
    print(processed / "paired_eval_manifest.csv")


if __name__ == "__main__":
    main()
