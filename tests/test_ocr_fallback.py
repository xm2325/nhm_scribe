import sys
from herbarium_scribe.ocr import resolve_ocr_backend


def test_auto_ocr_falls_back_when_paddle_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "paddleocr", None)
    assert resolve_ocr_backend("auto") in {"tesseract", "paddle"}


def test_requested_paddle_does_not_crash_when_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "paddleocr", None)
    assert resolve_ocr_backend("paddle") == "tesseract"
