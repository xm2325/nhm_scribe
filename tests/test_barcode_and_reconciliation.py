from pathlib import Path

import pandas as pd

from herbarium_scribe.ocr import run_ocr
from herbarium_scribe.reconcile import _safe_gbif_name, reconcile_dataframe


def test_run_ocr_adds_barcode_decoder_row(tmp_path: Path, monkeypatch):
    image = tmp_path / "sheet.jpg"
    image.write_bytes(b"image")
    layout = pd.DataFrame([{
        "occurrenceID": "eval:1",
        "region_id": "eval:1::whole",
        "region_label": "whole_sheet",
        "region_type": "whole_sheet",
        "evidence_source": "whole_sheet",
        "prompt_header": "SOURCE=whole_sheet",
        "image_path": str(image),
        "crop_path": str(image),
        "fixture_label_text": "",
    }])
    monkeypatch.setattr(
        "herbarium_scribe.ocr.ocr_image_tesseract",
        lambda *args, **kwargs: ("label text", None, "ok"),
    )
    monkeypatch.setattr(
        "herbarium_scribe.ocr.decode_barcodes",
        lambda *args, **kwargs: (
            [
                {"value": "K0001", "format": "Code 39"},
                {"value": "K0002", "format": "Data Matrix"},
            ],
            "ok",
            0.25,
        ),
    )
    paths = {
        "ocr": tmp_path / "ocr",
        "processed": tmp_path / "processed",
    }
    paths["ocr"].mkdir()
    paths["processed"].mkdir()

    out = run_ocr(
        layout,
        {
            "ocr": {
                "backend": "tesseract",
                "allow_fixture_text": False,
                "barcode_decoder": {"enabled": True},
            }
        },
        paths,
    )

    barcode = out[out["ocr_engine"].eq("zxingcpp")].iloc[0]
    assert barcode["ocr_text"] == "K0001\nK0002"
    assert barcode["barcode_count"] == 2
    assert bool(barcode["barcode_ambiguous"]) is True
    assert "ambiguous=true" in barcode["prompt_header"]


def test_safe_gbif_name_repairs_authorship_only():
    corrected, applied = _safe_gbif_name(
        "Ceriops tagal (Perr.) C.B.Robin.",
        {
            "scientificName": "Ceriops tagal (Perr.) C.B.Rob.",
            "confidence": 100,
        },
        90,
    )
    assert corrected == "Ceriops tagal (Perr.) C.B.Rob."
    assert applied is True

    unchanged, applied = _safe_gbif_name(
        "Borleric se. 8 Somos a",
        {
            "scientificName": "Barleria L.",
            "confidence": 100,
        },
        90,
    )
    assert unchanged == "Borleric se. 8 Somos a"
    assert applied is False


def test_reconcile_dataframe_applies_audited_gbif_correction(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "herbarium_scribe.reconcile._gbif_match",
        lambda *args, **kwargs: {
            "scientificName": "Ceriops tagal (Perr.) C.B.Rob.",
            "confidence": 100,
            "matchType": "EXACT",
        },
    )
    frame = pd.DataFrame([{
        "occurrenceID": "eval:1",
        "method": "llm",
        "scientificName": "Ceriops tagal (Perr.) C.B.Robin.",
        "country": "",
        "stateProvince": "",
    }])
    paths = {"processed": tmp_path}

    out = reconcile_dataframe(
        frame,
        paths,
        {
            "reconciliation": {
                "use_gbif_api": True,
                "gbif_methods": ["llm"],
                "gbif_cache_path": str(tmp_path / "gbif.json"),
            }
        },
    )

    assert out.iloc[0]["scientificName"] == "Ceriops tagal (Perr.) C.B.Rob."
    assert bool(out.iloc[0]["taxonomy_correction_applied"]) is True
    assert out.iloc[0]["scientificName_verbatim"].endswith("Robin.")


def test_reconcile_skips_gbif_for_non_binomial_or_unselected_method(tmp_path: Path, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("GBIF should not be called")

    monkeypatch.setattr("herbarium_scribe.reconcile._gbif_match", fail_if_called)
    frame = pd.DataFrame([
        {
            "occurrenceID": "eval:1",
            "method": "llm",
            "scientificName": "Carex",
            "country": "",
            "stateProvince": "",
        },
        {
            "occurrenceID": "eval:2",
            "method": "rule_ocr",
            "scientificName": "Rosa canina",
            "country": "",
            "stateProvince": "",
        },
    ])

    out = reconcile_dataframe(
        frame,
        None,
        {
            "reconciliation": {
                "use_gbif_api": True,
                "gbif_methods": ["llm"],
                "gbif_cache_path": str(tmp_path / "gbif.json"),
            }
        },
    )

    assert out["taxonomy_correction_applied"].eq(False).all()
