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
