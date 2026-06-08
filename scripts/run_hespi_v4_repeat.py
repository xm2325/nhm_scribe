from __future__ import annotations

import argparse

import pandas as pd

from herbarium_scribe.pipeline import (
    load_runtime,
    stage_evaluate,
    stage_extract,
    stage_reconcile,
)
from herbarium_scribe.report import write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg, paths = load_runtime(args.config)
    required = [
        paths["processed"] / "eval_set.csv",
        paths["processed"] / "ocr_by_region.csv",
        paths["processed"] / "image_manifest.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Run prepare_hespi_v4_repeats.py first; missing: {missing}")

    stage_extract(args.config)
    stage_reconcile(args.config)
    _, summary, stratified = stage_evaluate(args.config)
    split_summary = pd.read_csv(paths["processed"] / "split_summary.csv", dtype=str).fillna("")
    report = write_report(cfg, split_summary, summary, stratified, paths)
    print(report)


if __name__ == "__main__":
    main()
