from __future__ import annotations

import csv
import zipfile
from pathlib import Path

BUNDLE = Path("app_data/hespi_v10_ocr_visual_report.zip")
OUT = Path("reports/yoloe26_hespi_compare/hespi_v10_overview_members.csv")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BUNDLE) as archive:
        names = sorted(
            name for name in archive.namelist()
            if "/assets/overviews/" in name and name.lower().endswith((".jpg", ".jpeg", ".png"))
        )
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["member"])
        writer.writerows([[name] for name in names])
    print({"bundle": str(BUNDLE), "overview_count": len(names), "manifest": str(OUT)})


if __name__ == "__main__":
    main()
