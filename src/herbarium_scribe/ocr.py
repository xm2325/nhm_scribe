from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .logging_utils import get_logger
from .metadata import clean_str

logger = get_logger(__name__)
_TROCR_ENGINES: dict[str, Any] = {}
_TROCR_LOAD_ERRORS: dict[str, str] = {}


def _catalog_candidate_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", clean_str(value)).upper()


def _catalog_candidate_values(text: str) -> list[str]:
    values = []
    for line in str(text or "").splitlines():
        value = re.sub(r"\s+", " ", clean_str(line)).strip(" .,:;|_-")
        key = _catalog_candidate_key(value)
        if not 5 <= len(key) <= 24:
            continue
        if not any(char.isdigit() for char in key):
            continue
        if value not in values:
            values.append(value)
    return values


def rank_catalog_ocr_candidates(
    readings: list[tuple[str, str]],
    max_candidates: int = 8,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for source, text in readings:
        for value in _catalog_candidate_values(text):
            key = _catalog_candidate_key(value)
            item = grouped.setdefault(
                key,
                {
                    "value": value,
                    "normalised": key,
                    "votes": 0,
                    "sources": [],
                },
            )
            item["votes"] += 1
            if source not in item["sources"]:
                item["sources"].append(source)
            if len(value) < len(item["value"]):
                item["value"] = value
    ranked = []
    for item in grouped.values():
        key = item["normalised"]
        pattern_score = 0
        if any(char.isalpha() for char in key) and any(char.isdigit() for char in key):
            pattern_score += 2
        if 7 <= len(key) <= 16:
            pattern_score += 1
        if key[:1].isalpha():
            pattern_score += 1
        if key.isdigit():
            pattern_score -= 2
        item["score"] = item["votes"] * 10 + pattern_score
        ranked.append(item)
    ranked.sort(
        key=lambda item: (
            -int(item["score"]),
            -int(item["votes"]),
            str(item["normalised"]),
        )
    )
    return ranked[:max(1, int(max_candidates))]


def ocr_catalog_number_ensemble(
    image_path: str,
    *,
    standard_text: str,
    lang: str,
    config: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], str, int, float]:
    started = time.monotonic()
    if not image_path or not Path(image_path).exists():
        return standard_text, [], "missing_image", 0, 0.0
    if shutil.which("tesseract") is None:
        return standard_text, [], "tesseract_binary_missing", 0, time.monotonic() - started
    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
        import pytesseract
    except Exception as exc:
        return standard_text, [], f"unavailable:{type(exc).__name__}", 0, time.monotonic() - started

    upscale_factor = max(1, int(config.get("upscale_factor", 4)))
    psm_modes = [int(value) for value in config.get("psm_modes", [7, 13])]
    whitelist = clean_str(
        config.get("character_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.-")
    )
    max_candidates = int(config.get("max_candidates", 8))
    readings = [("standard", standard_text)]
    attempts = 0
    errors = []
    try:
        image = Image.open(image_path).convert("RGB")
        resized = image.resize(
            (image.width * upscale_factor, image.height * upscale_factor),
            Image.Resampling.LANCZOS,
        )
        grey = ImageOps.grayscale(resized)
        enhanced = ImageEnhance.Contrast(ImageOps.autocontrast(grey)).enhance(1.5)
        enhanced = enhanced.filter(ImageFilter.SHARPEN)
        variants = {
            "upscaled": resized,
            "autocontrast": enhanced,
        }
        for variant_name, variant in variants.items():
            for psm in psm_modes:
                attempts += 1
                tesseract_config = f"--psm {psm}"
                if whitelist:
                    tesseract_config += f" -c tessedit_char_whitelist={whitelist}"
                source = f"{variant_name}_psm{psm}"
                try:
                    text = pytesseract.image_to_string(
                        variant,
                        lang=lang,
                        config=tesseract_config,
                    )
                    readings.append((source, clean_str(text)))
                except Exception as exc:
                    errors.append(f"{source}:{type(exc).__name__}")
    except Exception as exc:
        return standard_text, [], f"error:{type(exc).__name__}", attempts, time.monotonic() - started

    candidates = rank_catalog_ocr_candidates(readings, max_candidates=max_candidates)
    text = "\n".join(str(item["value"]) for item in candidates)
    if not text:
        text = standard_text
    if candidates:
        status = "ok"
    elif errors and len(errors) == attempts:
        status = "error:" + ",".join(errors)
    elif errors:
        status = "partial_error:" + ",".join(errors)
    else:
        status = "no_candidate"
    return text, candidates, status, attempts, time.monotonic() - started


def _catalog_ensemble_region_ids(
    layout_df: pd.DataFrame,
    config: dict[str, Any],
) -> set[str]:
    region_types = {
        clean_str(value).lower()
        for value in config.get("region_types", ["number", "barcode", "database_label"])
    }
    max_regions = max(1, int(config.get("max_regions_per_record", 3)))
    if layout_df.empty or "region_id" not in layout_df.columns:
        return set()
    candidates = layout_df.copy()
    region_series = candidates.get(
        "region_type",
        pd.Series("", index=candidates.index, dtype=str),
    ).astype(str).str.lower()
    candidates = candidates[region_series.isin(region_types)].copy()
    if candidates.empty:
        return set()
    candidates["_confidence"] = pd.to_numeric(
        candidates.get("layout_confidence", pd.Series("", index=candidates.index)),
        errors="coerce",
    ).fillna(-1.0)
    candidates["_order"] = range(len(candidates))
    candidates = candidates.sort_values(
        ["occurrenceID", "_confidence", "_order"],
        ascending=[True, False, True],
        kind="stable",
    )
    selected = candidates.groupby("occurrenceID", sort=False).head(max_regions)
    return set(selected["region_id"].astype(str))


def decode_barcodes(
    image_path: str,
    max_dimension: int = 6000,
) -> tuple[list[dict[str, str]], str, float]:
    started = time.monotonic()
    if not image_path or not Path(image_path).exists():
        return [], "missing_image", 0.0
    try:
        from PIL import Image
        import zxingcpp
    except Exception as exc:
        return [], f"unavailable:{type(exc).__name__}", time.monotonic() - started
    try:
        image = Image.open(image_path).convert("RGB")
        if max_dimension > 0 and max(image.size) > max_dimension:
            image.thumbnail((max_dimension, max_dimension))
        decoded = zxingcpp.read_barcodes(
            image,
            try_rotate=True,
            try_downscale=True,
            try_invert=True,
        )
        values = []
        seen = set()
        for item in decoded:
            value = clean_str(getattr(item, "text", ""))
            if not value or value in seen:
                continue
            seen.add(value)
            values.append({
                "value": value,
                "format": clean_str(str(getattr(item, "format", ""))),
            })
        return values, "ok" if values else "no_barcode", time.monotonic() - started
    except Exception as exc:
        return [], f"error:{type(exc).__name__}", time.monotonic() - started


def resolve_ocr_backend(backend: str = "tesseract") -> str:
    backend = (backend or "tesseract").lower()
    if backend == "auto":
        try:
            import paddleocr  # noqa: F401
            return "paddle"
        except Exception as e:
            logger.warning("PaddleOCR unavailable; falling back to Tesseract: %s", e)
            return "tesseract"
    if backend == "paddle":
        try:
            import paddleocr  # noqa: F401
            return "paddle"
        except Exception as e:
            logger.warning("Requested PaddleOCR but import failed; falling back to Tesseract: %s", e)
            return "tesseract"
    return "tesseract"


def ocr_image_tesseract(
    image_path: str,
    lang: str = "eng",
    config: str = "",
) -> tuple[str, float | None, str]:
    if not image_path or not Path(image_path).exists():
        return "", None, "missing_image"
    if shutil.which("tesseract") is None:
        return "", None, "tesseract_binary_missing"
    try:
        from PIL import Image
        import pytesseract
        text = pytesseract.image_to_string(Image.open(image_path), lang=lang, config=config)
        return clean_str(text), None, "ok"
    except Exception as e:
        return "", None, f"error:{type(e).__name__}"


def ocr_image_paddle(image_path: str) -> tuple[str, float | None, str]:
    if not image_path or not Path(image_path).exists():
        return "", None, "missing_image"
    try:
        from paddleocr import PaddleOCR
        engine = PaddleOCR(use_angle_cls=True, lang="latin", show_log=False)
        result = engine.ocr(image_path, cls=True)
        texts, confs = [], []
        for block in result or []:
            for item in block or []:
                if len(item) >= 2 and isinstance(item[1], (list, tuple)):
                    texts.append(str(item[1][0]))
                    try:
                        confs.append(float(item[1][1]))
                    except Exception:
                        pass
        conf = sum(confs) / len(confs) if confs else None
        return clean_str("\n".join(texts)), conf, "ok"
    except Exception as e:
        return "", None, f"error:{type(e).__name__}"


def ocr_image_hespi_trocr(
    image_path: str,
    model_size: str = "small",
) -> tuple[str, str, float]:
    started = time.monotonic()
    if not image_path or not Path(image_path).exists():
        return "", "missing_image", 0.0
    try:
        from hespi.ocr import TrOCR
    except Exception as exc:
        return "", f"unavailable:{type(exc).__name__}", time.monotonic() - started
    try:
        if model_size in _TROCR_LOAD_ERRORS:
            return "", _TROCR_LOAD_ERRORS[model_size], time.monotonic() - started
        engine = _TROCR_ENGINES.get(model_size)
        if engine is None:
            try:
                engine = TrOCR(size=model_size)
            except Exception as exc:
                status = f"model_load_error:{type(exc).__name__}"
                _TROCR_LOAD_ERRORS[model_size] = status
                return "", status, time.monotonic() - started
            _TROCR_ENGINES[model_size] = engine
        text = clean_str(engine.get_text(Path(image_path)))
        return text, "ok" if text else "empty_output", time.monotonic() - started
    except Exception as exc:
        return "", f"error:{type(exc).__name__}", time.monotonic() - started


def _htr_region_ids(
    layout_df: pd.DataFrame,
    config: dict[str, Any],
) -> set[str]:
    region_types = {
        clean_str(value).lower()
        for value in config.get("region_types", ["collector", "year", "month", "day"])
    }
    max_regions = max(1, int(config.get("max_regions_per_record", 8)))
    if layout_df.empty or "region_id" not in layout_df.columns:
        return set()
    candidates = layout_df.copy()
    region_series = candidates.get(
        "region_type",
        pd.Series("", index=candidates.index, dtype=str),
    ).astype(str).str.lower()
    candidates = candidates[region_series.isin(region_types)].copy()
    if candidates.empty:
        return set()
    candidates["_confidence"] = pd.to_numeric(
        candidates.get("layout_confidence", pd.Series("", index=candidates.index)),
        errors="coerce",
    ).fillna(-1.0)
    candidates["_order"] = range(len(candidates))
    candidates = candidates.sort_values(
        ["occurrenceID", "_confidence", "_order"],
        ascending=[True, False, True],
        kind="stable",
    )
    selected = candidates.groupby("occurrenceID", sort=False).head(max_regions)
    return set(selected["region_id"].astype(str))


def _empty_htr_diagnostics() -> dict[str, Any]:
    return {
        "htr_model": "",
        "htr_elapsed_seconds": "",
        "htr_source_region_id": "",
    }


def run_ocr(layout_df: pd.DataFrame, cfg: dict[str, Any], paths: dict[str, Path]) -> pd.DataFrame:
    ocfg = cfg.get("ocr", {})
    backend = resolve_ocr_backend(ocfg.get("backend", "tesseract"))
    allow_fixture = bool(ocfg.get("allow_fixture_text", True))
    include_region_labels = bool(ocfg.get("include_region_labels_in_prompt", False))
    ensemble_cfg = ocfg.get("catalog_number_ensemble", {})
    ensemble_enabled = backend == "tesseract" and bool(ensemble_cfg.get("enabled", False))
    ensemble_region_ids = (
        _catalog_ensemble_region_ids(layout_df, ensemble_cfg)
        if ensemble_enabled
        else set()
    )
    rows = []
    for _, row in layout_df.iterrows():
        crop_path = clean_str(row.get("crop_path", ""))
        fixture_text = clean_str(row.get("fixture_label_text", ""))
        status = ""
        conf = None
        engine_used = backend
        used_fixture_text = False
        tesseract_config = clean_str(row.get("ocr_tesseract_config", "")) or clean_str(ocfg.get("tesseract_config", ""))
        if backend == "paddle":
            text, conf, status = ocr_image_paddle(crop_path)
            if not text:
                text, conf2, status2 = ocr_image_tesseract(
                    crop_path,
                    lang=ocfg.get("tesseract_lang", "eng"),
                    config=tesseract_config,
                )
                engine_used = "tesseract_after_paddle_fallback"
                conf = conf if conf is not None else conf2
                status = f"paddle_{status};tesseract_{status2}"
        else:
            text, conf, status = ocr_image_tesseract(
                crop_path,
                lang=ocfg.get("tesseract_lang", "eng"),
                config=tesseract_config,
            )
        ensemble_used = False
        ensemble_candidates: list[dict[str, Any]] = []
        ensemble_attempts = 0
        ensemble_elapsed = 0.0
        if clean_str(row.get("region_id")) in ensemble_region_ids:
            (
                ensemble_text,
                ensemble_candidates,
                ensemble_status,
                ensemble_attempts,
                ensemble_elapsed,
            ) = ocr_catalog_number_ensemble(
                crop_path,
                standard_text=text,
                lang=ocfg.get("tesseract_lang", "eng"),
                config=ensemble_cfg,
            )
            text = ensemble_text
            status = f"{status};ensemble_{ensemble_status}"
            engine_used = "tesseract_catalog_number_ensemble"
            ensemble_used = True
        if not text and allow_fixture and fixture_text:
            text = fixture_text
            engine_used = f"fixture_text_after_{engine_used}"
            status = f"{status};fixture_text_used"
            conf = 1.0
            used_fixture_text = True
        out_txt = paths["ocr"] / (clean_str(row.get("region_id", "region")).replace(":", "_").replace("/", "_") + ".txt")
        out_txt.write_text(text, encoding="utf-8")
        error_message = ""
        if "error:" in status or "missing" in status:
            error_message = status
        region_type = clean_str(row.get("region_type", row.get("region_label", "label")))
        prompt_header = clean_str(row.get("prompt_header", ""))
        if ensemble_used and text:
            prompt_header = (
                "FIELD=catalog_number; SOURCE=tesseract_crop_ensemble; "
                f"candidate_count={len(ensemble_candidates)}; "
                f"ambiguous={'true' if len(ensemble_candidates) > 1 else 'false'}; "
                "ranked_hypotheses=true"
            )
            prompt_text = f"[{prompt_header}]\n{text}"
        elif prompt_header and text:
            prompt_text = f"[{prompt_header}]\n{text}"
        else:
            prompt_text = f"[{region_type}]\n{text}" if include_region_labels and text else text
        rows.append({
            "occurrenceID": clean_str(row.get("occurrenceID")),
            "region_id": clean_str(row.get("region_id")),
            "region_label": clean_str(row.get("region_label", "label")),
            "region_type": region_type,
            "evidence_source": clean_str(row.get("evidence_source", region_type)),
            "prompt_header": prompt_header,
            "tesseract_config": tesseract_config,
            "ocr_engine": engine_used,
            "ocr_status": status,
            "ocr_confidence": conf if conf is not None else "",
            "ocr_text": text,
            "ocr_prompt_text": prompt_text,
            "text_length": len(text),
            "image_path": clean_str(row.get("image_path", "")),
            "crop_path": crop_path,
            "error_message": error_message,
            "used_fixture_text": used_fixture_text,
            "ocr_text_path": str(out_txt),
            "barcode_format": "",
            "barcode_count": "",
            "barcode_ambiguous": "",
            "barcode_elapsed_seconds": "",
            "ocr_ensemble_used": ensemble_used,
            "ocr_ensemble_candidate_count": len(ensemble_candidates),
            "ocr_ensemble_ambiguous": len(ensemble_candidates) > 1,
            "ocr_ensemble_candidates_json": json.dumps(ensemble_candidates, ensure_ascii=False),
            "ocr_ensemble_attempts": ensemble_attempts,
            "ocr_ensemble_elapsed_seconds": round(ensemble_elapsed, 3),
            **_empty_htr_diagnostics(),
        })
    htr_cfg = ocfg.get("handwriting_recognition", {})
    if bool(htr_cfg.get("enabled", False)):
        model_size = clean_str(htr_cfg.get("model_size", "small")) or "small"
        htr_region_ids = _htr_region_ids(layout_df, htr_cfg)
        prompt_headers = {
            "collector": "FIELD=recorded_by",
            "year": "FIELD=event_date",
            "month": "FIELD=event_date",
            "day": "FIELD=event_date",
            "locality": "FIELD=locality",
            "geolocation": "FIELD=locality",
        }
        prompt_headers.update({
            clean_str(key).lower(): clean_str(value)
            for key, value in htr_cfg.get("prompt_headers", {}).items()
        })
        selected = layout_df[
            layout_df["region_id"].astype(str).isin(htr_region_ids)
        ]
        for _, htr_row in selected.iterrows():
            occurrence_id = clean_str(htr_row.get("occurrenceID"))
            source_region_id = clean_str(htr_row.get("region_id"))
            region_type = clean_str(
                htr_row.get("region_type", htr_row.get("region_label", "field"))
            )
            crop_path = clean_str(htr_row.get("crop_path"))
            text, status, elapsed = ocr_image_hespi_trocr(
                crop_path,
                model_size=model_size,
            )
            header = (
                f"{prompt_headers.get(region_type.lower(), f'FIELD={region_type}')}; "
                f"SOURCE=trocr_handwritten; region_type={region_type}; "
                f"model=microsoft/trocr-{model_size}-handwritten"
            )
            prompt_text = f"[{header}]\n{text}" if text else ""
            region_id = f"{source_region_id}::trocr"
            out_txt = paths["ocr"] / (
                region_id.replace(":", "_").replace("/", "_") + ".txt"
            )
            out_txt.write_text(text, encoding="utf-8")
            rows.append({
                "occurrenceID": occurrence_id,
                "region_id": region_id,
                "region_label": region_type,
                "region_type": region_type,
                "evidence_source": f"htr:{region_type}",
                "prompt_header": header,
                "tesseract_config": "",
                "ocr_engine": f"hespi_trocr_{model_size}",
                "ocr_status": status,
                "ocr_confidence": "",
                "ocr_text": text,
                "ocr_prompt_text": prompt_text,
                "text_length": len(text),
                "image_path": clean_str(htr_row.get("image_path", "")),
                "crop_path": crop_path,
                "error_message": (
                    status
                    if status.startswith(
                        ("error:", "missing", "unavailable:", "model_load_error:")
                    )
                    else ""
                ),
                "used_fixture_text": False,
                "ocr_text_path": str(out_txt),
                "barcode_format": "",
                "barcode_count": "",
                "barcode_ambiguous": "",
                "barcode_elapsed_seconds": "",
                "ocr_ensemble_used": False,
                "ocr_ensemble_candidate_count": "",
                "ocr_ensemble_ambiguous": "",
                "ocr_ensemble_candidates_json": "",
                "ocr_ensemble_attempts": "",
                "ocr_ensemble_elapsed_seconds": "",
                "htr_model": f"microsoft/trocr-{model_size}-handwritten",
                "htr_elapsed_seconds": round(elapsed, 3),
                "htr_source_region_id": source_region_id,
            })
    barcode_cfg = ocfg.get("barcode_decoder", {})
    if bool(barcode_cfg.get("enabled", False)):
        max_dimension = int(barcode_cfg.get("max_dimension", 6000))
        image_rows = (
            layout_df[["occurrenceID", "image_path"]]
            .drop_duplicates("occurrenceID")
            .fillna("")
        )
        for _, image_row in image_rows.iterrows():
            occurrence_id = clean_str(image_row.get("occurrenceID"))
            image_path = clean_str(image_row.get("image_path"))
            decoded, status, elapsed = decode_barcodes(image_path, max_dimension=max_dimension)
            values = [item["value"] for item in decoded]
            formats = [item["format"] for item in decoded]
            text = "\n".join(values)
            header = (
                "FIELD=catalog_number; SOURCE=barcode_decoder; "
                f"decoded_values={len(values)}; ambiguous={'true' if len(values) > 1 else 'false'}"
            )
            prompt_text = f"[{header}]\n{text}" if text else ""
            region_id = f"{occurrence_id}::barcode_decoder"
            out_txt = paths["ocr"] / (
                clean_str(region_id).replace(":", "_").replace("/", "_") + ".txt"
            )
            out_txt.write_text(text, encoding="utf-8")
            rows.append({
                "occurrenceID": occurrence_id,
                "region_id": region_id,
                "region_label": "barcode_decoder",
                "region_type": "barcode_decoder",
                "evidence_source": "barcode_decoder",
                "prompt_header": header,
                "tesseract_config": "",
                "ocr_engine": "zxingcpp",
                "ocr_status": status,
                "ocr_confidence": 1.0 if len(values) == 1 else "",
                "ocr_text": text,
                "ocr_prompt_text": prompt_text,
                "text_length": len(text),
                "image_path": image_path,
                "crop_path": image_path,
                "error_message": status if status.startswith(("error:", "missing", "unavailable:")) else "",
                "used_fixture_text": False,
                "ocr_text_path": str(out_txt),
                "barcode_format": ";".join(formats),
                "barcode_count": len(values),
                "barcode_ambiguous": len(values) > 1,
                "barcode_elapsed_seconds": round(elapsed, 3),
                "ocr_ensemble_used": False,
                "ocr_ensemble_candidate_count": "",
                "ocr_ensemble_ambiguous": "",
                "ocr_ensemble_candidates_json": "",
                "ocr_ensemble_attempts": "",
                "ocr_ensemble_elapsed_seconds": "",
                **_empty_htr_diagnostics(),
            })
    out = pd.DataFrame(rows)
    out.to_csv(paths["processed"] / "ocr_by_region.csv", index=False)
    combined = out.groupby("occurrenceID", as_index=False).agg({
        "ocr_text": "\n".join,
        "ocr_prompt_text": "\n\n".join,
        "text_length": "sum",
    })
    combined.to_csv(paths["processed"] / "ocr_combined.csv", index=False)
    return out
