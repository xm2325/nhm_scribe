from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from herbarium_scribe.download import safe_filename


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir",
        default="data/experiments/hespi_v4_eval10_shared/processed",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/hespi_v4_repeat10/review_bundle",
    )
    args = parser.parse_args()

    root = Path(args.processed_dir)
    output = Path(args.output_dir)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    eval_set = pd.read_csv(root / "eval_set.csv", dtype=str).fillna("")
    manifest = pd.read_csv(root / "image_manifest.csv", dtype=str).fillna("")
    selected = eval_set.merge(manifest, on="occurrenceID", how="left", suffixes=("", "_manifest"))
    rows = []
    for _, row in selected.iterrows():
        source = Path(str(row.get("image_path", "")))
        if not source.exists():
            raise SystemExit(f"Missing selected review image: {source}")
        name = safe_filename(str(row.get("catalogNumber", "")) or str(row["occurrenceID"]))
        target = image_dir / f"{name}{source.suffix.lower() or '.jpg'}"
        shutil.copy2(source, target)
        rows.append({
            "occurrenceID": row["occurrenceID"],
            "catalogNumber": row.get("catalogNumber", ""),
            "institutionCode": row.get("institutionCode", ""),
            "source_image_path": str(source),
            "artifact_image_path": str(target.relative_to(output.parent)),
            "sha256": row.get("sha256", ""),
        })
    pd.DataFrame(rows).to_csv(output / "review_image_manifest.csv", index=False)
    print(output)


if __name__ == "__main__":
    main()
