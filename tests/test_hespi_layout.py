from pathlib import Path

import pandas as pd
from PIL import Image

from herbarium_scribe.hespi_layout import detect_hespi_lite_layout
from herbarium_scribe.ocr import run_ocr


class Value:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def item(self):
        return self.value


class Coordinates:
    def __init__(self, values):
        self.values = [values]

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class Box:
    def __init__(self, class_id, confidence, bbox):
        self.cls = Value(class_id)
        self.conf = Value(confidence)
        self.xyxy = Coordinates(bbox)


class Result:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


class Model:
    def __init__(self, result):
        self.result = result

    def predict(self, **_kwargs):
        return [self.result]


class FakeHespi:
    sheet_component_model = Model(Result(
        {0: "primary specimen label", 1: "scale"},
        [Box(0, 0.95, [10, 10, 190, 90]), Box(1, 0.8, [0, 90, 50, 100])],
    ))
    label_field_model = Model(Result(
        {0: "genus", 1: "species"},
        [Box(0, 0.9, [5, 5, 80, 30]), Box(1, 0.85, [80, 5, 170, 30])],
    ))


def test_hespi_lite_selects_field_crops(tmp_path, monkeypatch):
    image_path = tmp_path / "sheet.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)
    paths = {
        "interim": tmp_path / "interim",
        "crops": tmp_path / "interim" / "crops",
        "processed": tmp_path / "processed",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    records = pd.DataFrame([{
        "occurrenceID": "urn:test:1",
        "catalogNumber": "TEST1",
        "fixture_label_text": "",
    }])
    manifest = pd.DataFrame([{
        "occurrenceID": "urn:test:1",
        "image_path": str(image_path),
    }])
    cfg = {
        "layout": {"strategy": "hespi_lite"},
        "outputs": {"prefix": "poc"},
    }
    monkeypatch.setattr(
        "herbarium_scribe.hespi_layout._create_hespi",
        lambda _cfg: FakeHespi(),
    )

    out = detect_hespi_lite_layout(records, manifest, cfg, paths)

    assert set(out["region_type"]) == {"genus", "species"}
    assert set(out["layout_method"]) == {"hespi_lite_label_field"}
    assert all(Path(path).exists() for path in out["crop_path"])
    diagnostics = pd.read_csv(paths["processed"] / "poc_hespi_layout_diagnostics.csv")
    assert diagnostics.loc[0, "primary_label_count"] == 1
    assert diagnostics.loc[0, "label_field_count"] == 2


def test_ocr_region_labels_are_only_added_to_prompt(tmp_path, monkeypatch):
    paths = {
        "ocr": tmp_path / "ocr",
        "processed": tmp_path / "processed",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    layout = pd.DataFrame([{
        "occurrenceID": "urn:test:1",
        "region_id": "urn:test:1::genus",
        "region_label": "genus",
        "region_type": "genus",
        "crop_path": str(tmp_path / "crop.jpg"),
        "image_path": str(tmp_path / "sheet.jpg"),
        "fixture_label_text": "",
    }])
    monkeypatch.setattr(
        "herbarium_scribe.ocr.ocr_image_tesseract",
        lambda *_args, **_kwargs: ("Rosa", None, "ok"),
    )

    out = run_ocr(
        layout,
        {"ocr": {"backend": "tesseract", "include_region_labels_in_prompt": True}},
        paths,
    )

    assert out.loc[0, "ocr_text"] == "Rosa"
    assert out.loc[0, "ocr_prompt_text"] == "[genus]\nRosa"


def test_hespi_hybrid_keeps_whole_sheet_primary_and_fields(tmp_path, monkeypatch):
    image_path = tmp_path / "sheet.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)
    paths = {
        "interim": tmp_path / "interim",
        "crops": tmp_path / "interim" / "crops",
        "processed": tmp_path / "processed",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    records = pd.DataFrame([{
        "occurrenceID": "urn:test:hybrid",
        "catalogNumber": "HYBRID1",
        "fixture_label_text": "",
    }])
    manifest = pd.DataFrame([{
        "occurrenceID": "urn:test:hybrid",
        "image_path": str(image_path),
    }])
    monkeypatch.setattr(
        "herbarium_scribe.hespi_layout._create_hespi",
        lambda _cfg: FakeHespi(),
    )

    out = detect_hespi_lite_layout(
        records,
        manifest,
        {"layout": {"strategy": "hespi_hybrid"}, "outputs": {"prefix": "hybrid"}},
        paths,
    )

    assert {"whole_sheet", "primary_label", "field:genus", "field:species"}.issubset(set(out["evidence_source"]))
    whole = out[out["evidence_source"] == "whole_sheet"].iloc[0]
    primary = out[out["evidence_source"] == "primary_label"].iloc[0]
    assert whole["ocr_tesseract_config"] == "--psm 11"
    assert primary["ocr_tesseract_config"] == "--psm 6"


def test_hespi_lean_hybrid_omits_label_field_crops(tmp_path, monkeypatch):
    image_path = tmp_path / "sheet.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)
    paths = {
        "interim": tmp_path / "interim",
        "crops": tmp_path / "interim" / "crops",
        "processed": tmp_path / "processed",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    records = pd.DataFrame([{
        "occurrenceID": "urn:test:lean",
        "catalogNumber": "LEAN1",
        "fixture_label_text": "",
    }])
    manifest = pd.DataFrame([{
        "occurrenceID": "urn:test:lean",
        "image_path": str(image_path),
    }])
    monkeypatch.setattr(
        "herbarium_scribe.hespi_layout._create_hespi",
        lambda _cfg: FakeHespi(),
    )

    out = detect_hespi_lite_layout(
        records,
        manifest,
        {
            "layout": {
                "strategy": "hespi_hybrid",
                "include_label_fields": False,
            },
            "outputs": {"prefix": "lean"},
        },
        paths,
    )

    assert set(out["evidence_source"]) == {"whole_sheet", "primary_label"}
    diagnostics = pd.read_csv(paths["processed"] / "lean_hespi_layout_diagnostics.csv")
    assert diagnostics.loc[0, "label_field_count"] == 0
    assert pd.isna(diagnostics.loc[0, "fallback_reason"])


def test_hespi_hybrid_selects_only_configured_htr_fields(tmp_path, monkeypatch):
    image_path = tmp_path / "sheet.jpg"
    Image.new("RGB", (200, 100), "white").save(image_path)
    paths = {
        "interim": tmp_path / "interim",
        "crops": tmp_path / "interim" / "crops",
        "processed": tmp_path / "processed",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    records = pd.DataFrame([{
        "occurrenceID": "urn:test:htr-fields",
        "catalogNumber": "HTR1",
        "fixture_label_text": "",
    }])
    manifest = pd.DataFrame([{
        "occurrenceID": "urn:test:htr-fields",
        "image_path": str(image_path),
    }])

    class HtrFieldHespi(FakeHespi):
        label_field_model = Model(Result(
            {0: "collector", 1: "year", 2: "genus"},
            [
                Box(0, 0.95, [5, 5, 80, 30]),
                Box(1, 0.90, [80, 5, 120, 30]),
                Box(2, 0.85, [120, 5, 180, 30]),
            ],
        ))

    monkeypatch.setattr(
        "herbarium_scribe.hespi_layout._create_hespi",
        lambda _cfg: HtrFieldHespi(),
    )

    out = detect_hespi_lite_layout(
        records,
        manifest,
        {
            "layout": {
                "strategy": "hespi_hybrid",
                "include_label_fields": True,
                "selected_field_types": ["collector", "year"],
            },
            "outputs": {"prefix": "htr_fields"},
        },
        paths,
    )

    selected_fields = out[out["evidence_source"].astype(str).str.startswith("field:")]
    assert set(selected_fields["region_type"]) == {"collector", "year"}
    all_fields = pd.read_csv(paths["processed"] / "htr_fields_hespi_label_fields.csv")
    assert set(all_fields["region_type"]) == {"collector", "year", "genus"}
    assert all_fields["selected_for_ocr"].sum() == 2
    diagnostics = pd.read_csv(paths["processed"] / "htr_fields_hespi_layout_diagnostics.csv")
    assert diagnostics.loc[0, "label_field_count"] == 3
    assert diagnostics.loc[0, "selected_label_field_count"] == 2


def test_run_ocr_adds_trocr_as_supplementary_row(tmp_path, monkeypatch):
    image_path = tmp_path / "collector.jpg"
    Image.new("RGB", (120, 30), "white").save(image_path)
    paths = {
        "ocr": tmp_path / "ocr",
        "processed": tmp_path / "processed",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    layout = pd.DataFrame([{
        "occurrenceID": "urn:test:htr",
        "region_id": "urn:test:htr::collector",
        "region_label": "collector",
        "region_type": "collector",
        "layout_confidence": 0.9,
        "evidence_source": "field:collector",
        "prompt_header": "FIELD=collector",
        "crop_path": str(image_path),
        "image_path": str(image_path),
        "fixture_label_text": "",
    }])
    monkeypatch.setattr(
        "herbarium_scribe.ocr.ocr_image_tesseract",
        lambda *_args, **_kwargs: ("G. Forrest", None, "ok"),
    )
    monkeypatch.setattr(
        "herbarium_scribe.ocr.ocr_image_hespi_trocr",
        lambda *_args, **_kwargs: ("George Forrest", "ok", 1.25),
    )

    out = run_ocr(
        layout,
        {
            "ocr": {
                "backend": "tesseract",
                "handwriting_recognition": {
                    "enabled": True,
                    "model_size": "small",
                    "region_types": ["collector"],
                },
            }
        },
        paths,
    )

    assert set(out["ocr_engine"]) == {"tesseract", "hespi_trocr_small"}
    htr = out[out["ocr_engine"].eq("hespi_trocr_small")].iloc[0]
    assert htr["ocr_text"] == "George Forrest"
    assert "FIELD=recorded_by" in htr["prompt_header"]
    assert "SOURCE=trocr_handwritten" in htr["prompt_header"]
    assert htr["htr_source_region_id"] == "urn:test:htr::collector"
