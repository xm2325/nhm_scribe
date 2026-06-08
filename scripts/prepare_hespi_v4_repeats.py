from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REQUIRED_FILES = [
    "demo_set.csv",
    "eval_set.csv",
    "split_summary.csv",
    "image_manifest.csv",
    "layout_boxes.csv",
    "ocr_by_region.csv",
    "ocr_combined.csv",
]

OPTIONAL_FILES = [
    "metadata_loaded.csv",
    "hespi_layout_diagnostics.csv",
    "hespi_sheet_components.csv",
    "hespi_label_fields.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-data-dir", default="data/experiments/hespi_v4_eval10_shared")
    parser.add_argument(
        "--repeat-data-dirs",
        nargs="+",
        default=[
            "data/experiments/hespi_v4_eval10_repeat_1",
            "data/experiments/hespi_v4_eval10_repeat_2",
            "data/experiments/hespi_v4_eval10_repeat_3",
        ],
    )
    args = parser.parse_args()

    source = Path(args.shared_data_dir) / "processed"
    missing = [name for name in REQUIRED_FILES if not (source / name).exists()]
    if missing:
        raise SystemExit(f"Shared OCR package is missing required files: {missing}")

    for value in args.repeat_data_dirs:
        target = Path(value) / "processed"
        target.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_FILES:
            shutil.copy2(source / name, target / name)
        for name in OPTIONAL_FILES:
            if (source / name).exists():
                shutil.copy2(source / name, target / name)
        print(f"Prepared shared OCR inputs in {target}")


if __name__ == "__main__":
    main()
