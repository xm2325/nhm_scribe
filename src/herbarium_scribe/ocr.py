from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .logging_utils import get_logger
from .metadata import clean_str

logger = get_logger(__name__)


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


def run_ocr(layout_df: pd.DataFrame, cfg: dict[str, Any], paths: dict[str, Path]) -> pd.DataFrame:
    ocfg = cfg.get("ocr", {})
    backend = resolve_ocr_backend(ocfg.get("backend", "tesseract"))
    allow_fixture = bool(ocfg.get("allow_fixture_text", True))
    include_region_labels = bool(ocfg.get("include_region_labels_in_prompt", False))
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
        if prompt_header and text:
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
