from __future__ import annotations

import argparse

from herbarium_scribe.pipeline import (
    load_runtime,
    require_htr_evidence,
    stage_download,
    stage_layout,
    stage_metadata,
    stage_ocr,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config, _ = load_runtime(args.config)
    _, eval_df, _ = stage_metadata(args.config)
    stage_download(args.config)
    stage_layout(args.config)
    ocr = stage_ocr(args.config)
    require_htr_evidence(
        ocr,
        set(eval_df["occurrenceID"].astype(str)),
        config,
    )
    print({
        "eval_n": len(eval_df),
        "ocr_rows": len(ocr),
        "llm_calls": 0,
    })


if __name__ == "__main__":
    main()
