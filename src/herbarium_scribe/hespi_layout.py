from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .download import safe_filename
from .logging_utils import get_logger
from .metadata import clean_str

logger = get_logger(__name__)

PRIMARY_LABEL_CLASSES = {
    "primary_specimen_label",
    "primary_label",
    "institutional_label",
}


def _normalise_label(value: Any) -> str:
    return "_".join(clean_str(value).lower().replace(":", " ").split())


def _scalar(value: Any) -> float:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _bbox(box: Any) -> list[int]:
    value = box.xyxy
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(round(float(item))) for item in value[:4]]


def _detections(result: Any) -> list[dict[str, Any]]:
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    boxes = [] if boxes is None else boxes
    rows = []
    for box in boxes:
        class_id = int(_scalar(box.cls))
        label = _normalise_label(names.get(class_id, class_id))
        rows.append({
            "label": label,
            "confidence": _scalar(box.conf),
            "bbox": _bbox(box),
        })
    return sorted(rows, key=lambda row: row["confidence"], reverse=True)


def _predict(model: Any, image_path: Path, resolution: int, confidence: float) -> tuple[list[dict[str, Any]], Any]:
    results = model.predict(
        source=[str(image_path)],
        show=False,
        save=False,
        batch=1,
        imgsz=resolution,
        conf=confidence,
        verbose=False,
    )
    result = next(iter(results))
    return _detections(result), result


def _save_annotation(result: Any, out_path: Path) -> str:
    if not hasattr(result, "plot"):
        return ""
    try:
        plotted = result.plot()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(plotted[..., ::-1]).save(out_path)
        return str(out_path)
    except Exception as exc:
        logger.warning("Could not save Hespi annotation %s: %s", out_path, exc)
        return ""


def _clamp_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return [x0, y0, x1, y1]


def _crop(image: Image.Image, bbox: list[int], out_path: Path) -> tuple[list[int], str]:
    fixed = _clamp_bbox(bbox, image.width, image.height)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(tuple(fixed)).convert("RGB").save(out_path)
    return fixed, str(out_path)


def _create_hespi(layout_cfg: dict[str, Any]) -> Any:
    from hespi.hespi import Hespi

    return Hespi(
        gpu=bool(layout_cfg.get("gpu", False)),
        htr=False,
        fuzzy=False,
        llm_model="none",
        force_download=bool(layout_cfg.get("force_download", False)),
        sheet_component_res=int(layout_cfg.get("sheet_component_resolution", 1280)),
        label_field_res=int(layout_cfg.get("label_field_resolution", 1280)),
    )


def _fallback_row(
    record: pd.Series,
    image_path: str,
    paths: dict[str, Path],
    method: str,
) -> dict[str, Any]:
    occurrence_id = clean_str(record.get("occurrenceID"))
    crop_path = ""
    bbox = [0, 0, 0, 0]
    if image_path and Path(image_path).exists():
        try:
            image = Image.open(image_path)
            bbox = [0, 0, image.width, image.height]
            target = paths["crops"] / "hespi_lite" / safe_filename(occurrence_id) / "full_image.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            image.convert("RGB").save(target)
            crop_path = str(target)
        except Exception:
            method = f"{method}_image_read_failed"
    return {
        "occurrenceID": occurrence_id,
        "region_id": f"{occurrence_id}::full_image",
        "region_label": "full_image",
        "region_type": "full_image",
        "layout_method": method,
        "layout_confidence": "",
        "bbox": json.dumps(bbox),
        "image_path": image_path,
        "crop_path": crop_path,
        "fixture_label_text": clean_str(record.get("fixture_label_text", "")),
    }


def _write_manifest(df: pd.DataFrame, name: str, cfg: dict[str, Any], paths: dict[str, Path]) -> None:
    target = paths["processed"] / name
    df.to_csv(target, index=False)
    prefix = clean_str(cfg.get("outputs", {}).get("prefix", ""))
    if prefix:
        df.to_csv(paths["processed"] / f"{prefix}_{name}", index=False)


def detect_hespi_lite_layout(
    records: pd.DataFrame,
    image_manifest: pd.DataFrame,
    cfg: dict[str, Any],
    paths: dict[str, Path],
) -> pd.DataFrame:
    layout_cfg = cfg.get("layout", {})
    sheet_resolution = int(layout_cfg.get("sheet_component_resolution", 1280))
    field_resolution = int(layout_cfg.get("label_field_resolution", 1280))
    component_confidence = float(layout_cfg.get("component_confidence", 0.25))
    field_confidence = float(layout_cfg.get("field_confidence", 0.20))
    max_primary_labels = int(layout_cfg.get("max_primary_labels", 2))
    max_fields = int(layout_cfg.get("max_fields_per_label", 20))
    configured_primary = layout_cfg.get("primary_label_classes", [])
    primary_classes = {
        _normalise_label(value) for value in configured_primary
    } or PRIMARY_LABEL_CLASSES

    manifest_by_id = {
        clean_str(row.get("occurrenceID")): clean_str(row.get("image_path"))
        for _, row in image_manifest.iterrows()
    }
    output_root = paths["interim"] / "hespi_lite"
    output_root.mkdir(parents=True, exist_ok=True)

    hespi = None
    load_error = ""
    hespi_version = ""
    hespi_device = ""
    try:
        hespi = _create_hespi(layout_cfg)
        try:
            hespi_version = version("hespi")
        except PackageNotFoundError:
            hespi_version = "unknown"
        hespi_device = clean_str(getattr(hespi, "device", ""))
        sheet_model = hespi.sheet_component_model
        field_model = hespi.label_field_model
    except Exception as exc:
        load_error = f"{type(exc).__name__}: {exc}"
        sheet_model = None
        field_model = None
        logger.warning("Hespi-lite unavailable; using full-image fallback: %s", load_error)

    layout_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for _, record in records.iterrows():
        occurrence_id = clean_str(record.get("occurrenceID"))
        image_path = manifest_by_id.get(occurrence_id, "")
        safe_id = safe_filename(clean_str(record.get("catalogNumber")) or occurrence_id)
        record_dir = output_root / safe_id
        fallback_reason = ""
        sheet_annotation = ""
        field_annotations: list[str] = []
        selected_rows: list[dict[str, Any]] = []
        n_components = 0
        n_primary = 0
        n_fields = 0

        if not image_path or not Path(image_path).exists():
            fallback_reason = "missing_image"
            selected_rows.append(_fallback_row(record, image_path, paths, "hespi_missing_image_fallback"))
        elif hespi is None or sheet_model is None or field_model is None:
            fallback_reason = f"hespi_load_failed:{load_error}"
            selected_rows.append(_fallback_row(record, image_path, paths, "hespi_unavailable_full_image_fallback"))
        else:
            try:
                source_image = Image.open(image_path).convert("RGB")
                components, sheet_result = _predict(
                    sheet_model,
                    Path(image_path),
                    sheet_resolution,
                    component_confidence,
                )
                n_components = len(components)
                sheet_annotation = _save_annotation(
                    sheet_result,
                    record_dir / "sheet_components_annotated.jpg",
                )
                primary: list[dict[str, Any]] = []
                for index, component in enumerate(components):
                    label = component["label"]
                    component_path = record_dir / "components" / f"{index:02d}_{label}.jpg"
                    fixed_bbox, crop_path = _crop(source_image, component["bbox"], component_path)
                    is_primary = label in primary_classes and len(primary) < max_primary_labels
                    component_rows.append({
                        "occurrenceID": occurrence_id,
                        "component_id": f"{occurrence_id}::component_{index}",
                        "component_type": label,
                        "confidence": component["confidence"],
                        "bbox": json.dumps(fixed_bbox),
                        "image_path": image_path,
                        "crop_path": crop_path,
                        "selected_for_field_detection": is_primary,
                        "annotation_path": sheet_annotation,
                    })
                    if is_primary:
                        primary.append({
                            **component,
                            "bbox": fixed_bbox,
                            "crop_path": crop_path,
                        })
                n_primary = len(primary)

                for primary_index, primary_item in enumerate(primary):
                    primary_path = Path(primary_item["crop_path"])
                    primary_image = Image.open(primary_path).convert("RGB")
                    fields, field_result = _predict(
                        field_model,
                        primary_path,
                        field_resolution,
                        field_confidence,
                    )
                    annotation_path = _save_annotation(
                        field_result,
                        record_dir / f"label_{primary_index:02d}_fields_annotated.jpg",
                    )
                    if annotation_path:
                        field_annotations.append(annotation_path)
                    selected_fields = sorted(
                        fields[:max_fields],
                        key=lambda item: (item["bbox"][1], item["bbox"][0]),
                    )
                    for field_index, field in enumerate(selected_fields):
                        label = field["label"]
                        field_path = record_dir / "fields" / f"{primary_index:02d}_{field_index:02d}_{label}.jpg"
                        fixed_bbox, crop_path = _crop(primary_image, field["bbox"], field_path)
                        parent_x0, parent_y0, _, _ = primary_item["bbox"]
                        global_bbox = [
                            fixed_bbox[0] + parent_x0,
                            fixed_bbox[1] + parent_y0,
                            fixed_bbox[2] + parent_x0,
                            fixed_bbox[3] + parent_y0,
                        ]
                        region_id = f"{occurrence_id}::hespi_field_{primary_index}_{field_index}"
                        field_record = {
                            "occurrenceID": occurrence_id,
                            "region_id": region_id,
                            "region_label": label,
                            "region_type": label,
                            "layout_method": "hespi_lite_label_field",
                            "layout_confidence": field["confidence"],
                            "bbox": json.dumps(fixed_bbox),
                            "bbox_coordinate_space": "primary_specimen_label",
                            "global_bbox": json.dumps(global_bbox),
                            "parent_bbox": json.dumps(primary_item["bbox"]),
                            "image_path": image_path,
                            "crop_path": crop_path,
                            "primary_label_crop_path": str(primary_path),
                            "fixture_label_text": clean_str(record.get("fixture_label_text", "")),
                        }
                        selected_rows.append(field_record)
                        field_rows.append({
                            **field_record,
                            "field_annotation_path": annotation_path,
                        })
                        n_fields += 1

                if not selected_rows and primary:
                    fallback_reason = "no_label_fields_detected"
                    for primary_index, primary_item in enumerate(primary):
                        selected_rows.append({
                            "occurrenceID": occurrence_id,
                            "region_id": f"{occurrence_id}::primary_label_{primary_index}",
                            "region_label": "primary_specimen_label",
                            "region_type": "primary_specimen_label",
                            "layout_method": "hespi_primary_label_fallback",
                            "layout_confidence": primary_item["confidence"],
                            "bbox": json.dumps(primary_item["bbox"]),
                            "image_path": image_path,
                            "crop_path": primary_item["crop_path"],
                            "fixture_label_text": clean_str(record.get("fixture_label_text", "")),
                        })
                elif not selected_rows:
                    fallback_reason = "no_primary_label_detected"
                    selected_rows.append(_fallback_row(
                        record,
                        image_path,
                        paths,
                        "hespi_no_primary_label_full_image_fallback",
                    ))
            except Exception as exc:
                fallback_reason = f"hespi_detection_failed:{type(exc).__name__}:{exc}"
                logger.warning("Hespi-lite failed for %s: %s", occurrence_id, exc)
                selected_rows.append(_fallback_row(
                    record,
                    image_path,
                    paths,
                    "hespi_detection_error_full_image_fallback",
                ))

        layout_rows.extend(selected_rows)
        diagnostic_rows.append({
            "occurrenceID": occurrence_id,
            "image_path": image_path,
            "hespi_available": hespi is not None,
            "hespi_version": hespi_version,
            "hespi_device": hespi_device,
            "sheet_component_count": n_components,
            "primary_label_count": n_primary,
            "label_field_count": n_fields,
            "ocr_region_count": len(selected_rows),
            "fallback_used": bool(fallback_reason),
            "fallback_reason": fallback_reason,
            "sheet_annotation_path": sheet_annotation,
            "field_annotation_paths": json.dumps(field_annotations),
        })

    component_columns = [
        "occurrenceID", "component_id", "component_type", "confidence", "bbox",
        "image_path", "crop_path", "selected_for_field_detection", "annotation_path",
    ]
    field_columns = [
        "occurrenceID", "region_id", "region_label", "region_type", "layout_method",
        "layout_confidence", "bbox", "bbox_coordinate_space", "global_bbox",
        "parent_bbox", "image_path", "crop_path",
        "primary_label_crop_path", "fixture_label_text", "field_annotation_path",
    ]
    diagnostic_columns = [
        "occurrenceID", "image_path", "hespi_available", "hespi_version", "hespi_device",
        "sheet_component_count",
        "primary_label_count", "label_field_count", "ocr_region_count", "fallback_used",
        "fallback_reason", "sheet_annotation_path", "field_annotation_paths",
    ]
    layout_df = pd.DataFrame(layout_rows)
    _write_manifest(pd.DataFrame(component_rows, columns=component_columns), "hespi_sheet_components.csv", cfg, paths)
    _write_manifest(pd.DataFrame(field_rows, columns=field_columns), "hespi_label_fields.csv", cfg, paths)
    _write_manifest(pd.DataFrame(diagnostic_rows, columns=diagnostic_columns), "hespi_layout_diagnostics.csv", cfg, paths)
    return layout_df
