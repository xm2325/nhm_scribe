from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .logging_utils import get_logger
from .metadata import clean_str

logger = get_logger(__name__)


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


def ocr_image_tesseract(image_path: str, lang: str = "eng") -> tuple[str, float | None, str]:
    if not image_path or not Path(image_path).exists():
        return "", None, "missing_image"
    if shutil.which("tesseract") is None:
        return "", None, "tesseract_binary_missing"
    try:
        from PIL import Image
        import pytesseract
        text = pytesseract.image_to_string(Image.open(image_path), lang=lang)
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
    rows = []
    for _, row in layout_df.iterrows():
        crop_path = clean_str(row.get("crop_path", ""))
        fixture_text = clean_str(row.get("fixture_label_text", ""))
        status = ""
        conf = None
        engine_used = backend
        if backend == "paddle":
            text, conf, status = ocr_image_paddle(crop_path)
            if not text:
                text, conf2, status2 = ocr_image_tesseract(crop_path, lang=ocfg.get("tesseract_lang", "eng"))
                engine_used = "tesseract_after_paddle_fallback"
                conf = conf if conf is not None else conf2
                status = f"paddle_{status};tesseract_{status2}"
        else:
            text, conf, status = ocr_image_tesseract(crop_path, lang=ocfg.get("tesseract_lang", "eng"))
        if not text and allow_fixture and fixture_text:
            text = fixture_text
            engine_used = f"fixture_text_after_{engine_used}"
            status = f"{status};fixture_text_used"
            conf = 1.0
        out_txt = paths["ocr"] / (clean_str(row.get("region_id", "region")).replace(":", "_").replace("/", "_") + ".txt")
        out_txt.write_text(text, encoding="utf-8")
        error_message = ""
        if "error:" in status or "missing" in status:
            error_message = status
        rows.append({
            "occurrenceID": clean_str(row.get("occurrenceID")),
            "region_id": clean_str(row.get("region_id")),
            "region_label": clean_str(row.get("region_label", "label")),
            "region_type": clean_str(row.get("region_type", row.get("region_label", "label"))),
            "ocr_engine": engine_used,
            "ocr_status": status,
            "ocr_confidence": conf if conf is not None else "",
            "ocr_text": text,
            "text_length": len(text),
            "image_path": clean_str(row.get("image_path", "")),
            "crop_path": crop_path,
            "error_message": error_message,
            "ocr_text_path": str(out_txt),
        })
    out = pd.DataFrame(rows)
    out.to_csv(paths["processed"] / "ocr_by_region.csv", index=False)
    combined = out.groupby("occurrenceID", as_index=False).agg({"ocr_text": "\n".join, "text_length": "sum"})
    combined.to_csv(paths["processed"] / "ocr_combined.csv", index=False)
    return out
