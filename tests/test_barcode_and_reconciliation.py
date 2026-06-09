from pathlib import Path

import pandas as pd

from herbarium_scribe.ocr import rank_catalog_ocr_candidates, run_ocr
from herbarium_scribe.reconcile import _safe_gbif_name, reconcile_dataframe


def test_catalog_ocr_candidates_are_deduplicated_and_ranked():
    candidates = rank_catalog_ocr_candidates([
        ("standard", "BM0006253"),
        ("upscaled_psm7", "BM000625315"),
        ("upscaled_psm13", "BM000625315"),
        ("autocontrast_psm7", "591"),
        ("autocontrast_psm13", "BM 000625315"),
    ])

    assert candidates[0]["normalised"] == "BM000625315"
    assert candidates[0]["votes"] == 3
    assert candidates[1]["normalised"] == "BM0006253"
    assert all(item["normalised"] != "591" for item in candidates)


def test_run_ocr_ensembles_only_top_catalog_regions(tmp_path: Path, monkeypatch):
    image = tmp_path / "sheet.jpg"
    image.write_bytes(b"image")
    layout = pd.DataFrame([
        {
            "occurrenceID": "eval:1",
            "region_id": "eval:1::number-high",
            "region_label": "number",
            "region_type": "number",
            "layout_confidence": 0.9,
            "evidence_source": "sheet_component:number",
            "prompt_header": "FIELD=catalog_number",
            "image_path": str(image),
            "crop_path": str(image),
            "fixture_label_text": "",
        },
        {
            "occurrenceID": "eval:1",
            "region_id": "eval:1::number-low",
            "region_label": "number",
            "region_type": "number",
            "layout_confidence": 0.4,
            "evidence_source": "sheet_component:number",
            "prompt_header": "FIELD=catalog_number",
            "image_path": str(image),
            "crop_path": str(image),
            "fixture_label_text": "",
        },
    ])
    monkeypatch.setattr(
        "herbarium_scribe.ocr.ocr_image_tesseract",
        lambda *args, **kwargs: ("BM0006253", None, "ok"),
    )
    calls = []

    def fake_ensemble(image_path, *, standard_text, lang, config):
        calls.append((image_path, standard_text, lang, config))
        return (
            "BM000625315\nBM0006253",
            [
                {
                    "value": "BM000625315",
                    "normalised": "BM000625315",
                    "votes": 3,
                    "sources": ["upscaled_psm7"],
                    "score": 34,
                },
                {
                    "value": "BM0006253",
                    "normalised": "BM0006253",
                    "votes": 1,
                    "sources": ["standard"],
                    "score": 14,
                },
            ],
            "ok",
            4,
            0.5,
        )

    monkeypatch.setattr(
        "herbarium_scribe.ocr.ocr_catalog_number_ensemble",
        fake_ensemble,
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
                "catalog_number_ensemble": {
                    "enabled": True,
                    "max_regions_per_record": 1,
                },
            }
        },
        paths,
    )

    assert len(calls) == 1
    high = out[out["region_id"].eq("eval:1::number-high")].iloc[0]
    low = out[out["region_id"].eq("eval:1::number-low")].iloc[0]
    assert high["ocr_engine"] == "tesseract_catalog_number_ensemble"
    assert high["ocr_ensemble_candidate_count"] == 2
    assert "ranked_hypotheses=true" in high["prompt_header"]
    assert low["ocr_engine"] == "tesseract"
    assert low["ocr_text"] == "BM0006253"


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
