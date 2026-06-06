from pathlib import Path

from PIL import Image

import pandas as pd

from herbarium_scribe.download import download_one_diagnostic, file_sha256, run_download


class Response:
    def __init__(self, status_code: int, content: bytes = b"", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error


def test_download_retries_and_hashes_valid_image(tmp_path, monkeypatch):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (20, 20), "white").save(source)
    content = source.read_bytes()
    responses = [
        Response(503),
        Response(200, content, {"Content-Type": "image/jpeg"}),
    ]
    sleeps = []
    monkeypatch.setattr("requests.get", lambda *_args, **_kwargs: responses.pop(0))
    monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))
    target = tmp_path / "target.jpg"

    result = download_one_diagnostic(
        "https://example.test/image.jpg",
        target,
        retries=2,
        retry_backoff_seconds=0.1,
        min_bytes=10,
    )

    assert result["ok"] is True
    assert result["attempts"] == 2
    assert result["sha256"] == file_sha256(target)
    assert len(sleeps) == 1


def test_shared_manifest_is_read_without_redownloading(tmp_path, monkeypatch):
    image_path = tmp_path / "shared.jpg"
    Image.new("RGB", (20, 20), "white").save(image_path)
    digest = file_sha256(image_path)
    manifest_path = tmp_path / "paired_manifest.csv"
    pd.DataFrame([{
        "occurrenceID": "id1",
        "image_path": str(image_path),
        "sha256": digest,
        "selected_for_paired_eval": True,
    }]).to_csv(manifest_path, index=False)
    records = pd.DataFrame([{
        "occurrenceID": "id1",
        "catalogNumber": "C1",
        "image_url": "https://example.test/shared.jpg",
    }])
    paths = {
        "raw_images": tmp_path / "raw_images",
        "processed": tmp_path / "processed",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "requests.get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not download")),
    )

    out = run_download(
        records,
        {"image": {"shared_manifest_csv": str(manifest_path), "min_bytes": 10}},
        paths,
    )

    assert out.loc[0, "image_status"] == "shared_manifest"
    assert out.loc[0, "sha256"] == digest
    assert bool(out.loc[0, "paired_eligible"]) is True
