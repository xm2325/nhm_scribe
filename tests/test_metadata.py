import pandas as pd

from herbarium_scribe.metadata import load_zenodo_metadata


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


def test_zenodo_metadata_retries_gateway_timeout_and_caches_atomically(tmp_path, monkeypatch):
    csv_path = tmp_path / "source.csv"
    pd.DataFrame([{
        "persistentID": "urn:test:1",
        "jpegURL": "https://example.test/image.jpg",
        "catalogNumber": "TEST1",
    }]).to_csv(csv_path, index=False)
    responses = [
        Response(504, headers={"Retry-After": "0"}),
        Response(200, csv_path.read_bytes(), {"Content-Type": "text/csv"}),
    ]
    sleeps = []
    monkeypatch.setattr("requests.get", lambda *_args, **_kwargs: responses.pop(0))
    monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))
    cache_path = tmp_path / "metadata.csv"

    out = load_zenodo_metadata({
        "metadata": {
            "source": "zenodo",
            "record_id": "6372393",
            "cache_csv": str(cache_path),
            "retries": 2,
            "retry_backoff_seconds": 0,
        }
    })

    assert cache_path.exists()
    assert not cache_path.with_suffix(".csv.part").exists()
    assert len(sleeps) == 1
    assert out.loc[0, "occurrenceID"] == "urn:test:1"
    assert out.loc[0, "image_url"] == "https://example.test/image.jpg"
