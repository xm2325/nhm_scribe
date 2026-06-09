from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from .metadata import clean_str
from .schema import EXTRACTION_FIELDS, validate_record


def image_data_url(
    image_path: str | Path,
    *,
    max_dimension: int = 2200,
    jpeg_quality: int = 92,
) -> tuple[str, dict[str, Any]]:
    path = Path(image_path)
    with Image.open(path) as source:
        image = source.convert("RGB")
        original_size = image.size
        scale = min(1.0, max_dimension / max(image.size))
        if scale < 1.0:
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", {
        "source_path": str(path),
        "original_width": original_size[0],
        "original_height": original_size[1],
        "sent_width": image.width,
        "sent_height": image.height,
        "sent_bytes": len(output.getvalue()),
    }


def primary_label_vision_messages(
    crop_paths: list[str | Path],
    *,
    max_dimension: int = 2200,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": (
            "These images are automatically cropped primary labels from one herbarium specimen sheet. "
            "Read only visible text. First transcribe every visible character line by line, preserving "
            "spelling, punctuation, abbreviations, line breaks, and uncertain characters. Do not silently "
            "correct names or infer missing text. Then extract the requested herbarium fields only when the "
            "value is supported by your transcription. Return one JSON object with this shape: "
            '{"transcriptions":[{"image_index":1,"text":"all visible text","uncertain_spans":["..."]}],'
            '"fields":{"catalogNumber":{"value":"","confidence":0.0,"evidence_span":""},'
            '"scientificName":{"value":"","confidence":0.0,"evidence_span":""},'
            '"recordedBy":{"value":"","confidence":0.0,"evidence_span":""},'
            '"eventDate":{"value":"","confidence":0.0,"evidence_span":""},'
            '"country":{"value":"","confidence":0.0,"evidence_span":""},'
            '"stateProvince":{"value":"","confidence":0.0,"evidence_span":""},'
            '"decimalLatitude":{"value":"","confidence":0.0,"evidence_span":""},'
            '"decimalLongitude":{"value":"","confidence":0.0,"evidence_span":""},'
            '"typeStatus":{"value":"","confidence":0.0,"evidence_span":""}},'
            '"observations":[]}. Every non-empty evidence_span must be copied exactly from a transcription. '
            "Use empty values instead of guessing. Return JSON only."
        ),
    }]
    image_meta: list[dict[str, Any]] = []
    for index, crop_path in enumerate(crop_paths, start=1):
        url, meta = image_data_url(crop_path, max_dimension=max_dimension)
        image_meta.append({"image_index": index, **meta})
        content.extend([
            {"type": "text", "text": f"PRIMARY LABEL IMAGE {index}"},
            {"type": "image_url", "image_url": {"url": url}},
        ])
    return [
        {
            "role": "system",
            "content": (
                "You are a conservative herbarium-label transcription system. Separate visual transcription "
                "from semantic extraction. Never invent text that is not visible."
            ),
        },
        {"role": "user", "content": content},
    ], image_meta


def _strip_fence(text: str) -> str:
    value = clean_str(text)
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def parse_qwen_vision_output(raw_output: str) -> dict[str, Any] | None:
    text = _strip_fence(raw_output)
    if not text:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    transcriptions = obj.get("transcriptions", [])
    if isinstance(transcriptions, dict):
        transcriptions = [transcriptions]
    if not isinstance(transcriptions, list):
        transcriptions = []
    normalised_transcriptions = []
    for index, item in enumerate(transcriptions, start=1):
        if isinstance(item, str):
            item = {"image_index": index, "text": item, "uncertain_spans": []}
        if not isinstance(item, dict):
            continue
        uncertain = item.get("uncertain_spans", [])
        if isinstance(uncertain, str):
            uncertain = [uncertain]
        normalised_transcriptions.append({
            "image_index": item.get("image_index", index),
            "text": clean_str(item.get("text", "")),
            "uncertain_spans": [clean_str(value) for value in uncertain if clean_str(value)],
        })
    fields = obj.get("fields", {})
    if not isinstance(fields, dict):
        fields = {}
    record = validate_record(fields)
    observations = obj.get("observations", [])
    if isinstance(observations, str):
        observations = [observations]
    return {
        "transcriptions": normalised_transcriptions,
        "full_transcription": "\n\n".join(
            item["text"] for item in normalised_transcriptions if item["text"]
        ),
        "fields": {field: record[field] for field in EXTRACTION_FIELDS},
        "validation_warnings": record["warnings"],
        "observations": [clean_str(value) for value in observations if clean_str(value)],
    }
