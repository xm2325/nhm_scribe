from __future__ import annotations

import argparse
import csv
import hashlib
import io
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image


MANIFEST_PATH = "data/processed/real_eval_100_image_manifest.csv"
THUMB_PREFIX = "app_data/thumbnails/real_eval_100"


def thumbnail_name(occurrence_id: str) -> str:
    digest = hashlib.sha1(occurrence_id.encode("utf-8")).hexdigest()[:16]
    return f"{THUMB_PREFIX}/{digest}.jpg"


def read_manifest(bundle: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(bundle) as zf:
        with zf.open(MANIFEST_PATH) as fh:
            text = fh.read().decode("utf-8").splitlines()
    return list(csv.DictReader(text))


def download_image(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "HerbariumSCRIBEStreamlit/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def make_thumbnail(image_bytes: bytes, max_size: int, quality: int) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=quality, optimize=True)
        return out.getvalue()


def copy_existing_entries(source: Path, target: Path, skip_prefix: str) -> None:
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            if info.is_dir() or info.filename.startswith(skip_prefix):
                continue
            dst.writestr(info, src.read(info.filename))


def add_thumbnails(bundle: Path, max_size: int, quality: int, timeout: int, limit: int | None) -> None:
    rows = read_manifest(bundle)
    if limit is not None:
        rows = rows[:limit]

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    copy_existing_entries(bundle, tmp_path, THUMB_PREFIX)
    ok = 0
    failures = 0
    with zipfile.ZipFile(tmp_path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        for row in rows:
            occurrence_id = (row.get("occurrenceID") or "").strip()
            image_url = (row.get("image_url") or "").strip()
            if not occurrence_id or not image_url:
                failures += 1
                continue
            try:
                thumb = make_thumbnail(download_image(image_url, timeout), max_size=max_size, quality=quality)
            except (OSError, urllib.error.URLError, TimeoutError) as exc:
                failures += 1
                print(f"thumbnail failed: {occurrence_id} {exc}", flush=True)
                continue
            zf.writestr(thumbnail_name(occurrence_id), thumb)
            ok += 1
            print(f"thumbnail ok: {ok}/{len(rows)} {occurrence_id}", flush=True)

    tmp_path.replace(bundle)
    print(f"thumbnail summary: ok={ok} failed={failures} bundle={bundle}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add cached Streamlit thumbnails to the eval100 bundle.")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--max-size", type=int, default=900)
    parser.add_argument("--quality", type=int, default=72)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    add_thumbnails(args.bundle, max_size=args.max_size, quality=args.quality, timeout=args.timeout, limit=args.limit)


if __name__ == "__main__":
    main()
