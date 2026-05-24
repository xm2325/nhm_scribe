from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .metadata import clean_str


def detect_layout(records: pd.DataFrame, image_manifest: pd.DataFrame, cfg: dict[str, Any], paths: dict[str, Path]) -> pd.DataFrame:
    manifest = image_manifest.set_index("occurrenceID") if len(image_manifest) else pd.DataFrame()
    rows = []
    for _, row in records.iterrows():
        occ = clean_str(row.get("occurrenceID"))
        image_path = ""
        if len(manifest) and occ in manifest.index:
            image_path = clean_str(manifest.loc[occ].get("image_path", ""))
        safe = occ.replace(":", "_").replace("/", "_")
        crop_path = ""
        method = "fixture_text"
        bbox = [0, 0, 0, 0]
        if image_path and Path(image_path).exists():
            crop_path = str(paths["crops"] / f"{safe}_label.jpg")
            try:
                from PIL import Image
                img = Image.open(image_path)
                img.save(crop_path)
                bbox = [0, 0, int(img.width), int(img.height)]
                method = "full_image_fallback"
            except Exception:
                crop_path = ""
                method = "image_read_failed_fixture_text"
        rows.append({
            "occurrenceID": occ,
            "region_id": f"{occ}::label_0",
            "region_label": "label",
            "layout_method": method,
            "bbox": json.dumps(bbox),
            "crop_path": crop_path,
            "fixture_label_text": clean_str(row.get("fixture_label_text", "")),
        })
    out = pd.DataFrame(rows)
    out.to_csv(paths["processed"] / "layout_boxes.csv", index=False)
    with (paths["processed"] / "layout_boxes.jsonl").open("w", encoding="utf-8") as f:
        for rec in out.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out
