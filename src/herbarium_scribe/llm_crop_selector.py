from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any

from .metadata import clean_str


YOLOE_CLASS_ALIASES = {
    "primary specimen label": "primary_specimen_label",
    "printed label": "printed_label",
    "annotation label": "annotation_label",
    "handwritten note": "handwritten_data",
    "database label": "database_label",
    "barcode label": "barcode",
    "catalog number": "number",
    "type specimen label": "type_label",
    "herbarium stamp": "stamp",
}

COMPONENT_ROLES = {
    "primary_specimen_label": "primary_label",
    "printed_label": "primary_label",
    "database_label": "identifier_parent",
    "barcode": "identifier_child",
    "number": "identifier_child",
    "annotation_label": "annotation",
    "handwritten_data": "annotation",
    "type_label": "annotation",
    "stamp": "annotation",
}

DEFAULT_CLASS_THRESHOLDS = {
    "primary_specimen_label": 0.08,
    "printed_label": 0.10,
    "database_label": 0.10,
    "barcode": 0.08,
    "number": 0.05,
    "annotation_label": 0.10,
    "handwritten_data": 0.08,
    "type_label": 0.08,
    "stamp": 0.08,
}

DEFAULT_ROLE_QUOTAS = {
    "primary_label": 2,
    "identifier_parent": 2,
    "identifier_child": 3,
    "annotation": 2,
}

ROLE_ORDER = (
    "primary_label",
    "identifier_child",
    "identifier_parent",
    "annotation",
)

REFERENCE_EVIDENCE_FIELDS = (
    "catalogNumber",
    "scientificName",
    "recordedBy",
    "eventDate",
    "country",
    "stateProvince",
    "decimalLatitude",
    "decimalLongitude",
    "typeStatus",
)


def parse_bbox(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        items = value
    else:
        try:
            items = json.loads(clean_str(value))
        except (TypeError, json.JSONDecodeError):
            return [0, 0, 0, 0]
    if not isinstance(items, (list, tuple)) or len(items) < 4:
        return [0, 0, 0, 0]
    return [int(round(float(item))) for item in items[:4]]


def bbox_area(bbox: list[int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def bbox_intersection(left: list[int], right: list[int]) -> int:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


def bbox_iou(left: list[int], right: list[int]) -> float:
    intersection = bbox_intersection(left, right)
    union = bbox_area(left) + bbox_area(right) - intersection
    return intersection / union if union else 0.0


def containment_ratio(left: list[int], right: list[int]) -> float:
    smaller = min(bbox_area(left), bbox_area(right))
    return bbox_intersection(left, right) / smaller if smaller else 0.0


def canonical_component_type(value: Any) -> str:
    label = re.sub(r"[_\s]+", " ", clean_str(value).lower()).strip()
    return YOLOE_CLASS_ALIASES.get(label, label.replace(" ", "_"))


def component_role(component_type: Any) -> str:
    return COMPONENT_ROLES.get(canonical_component_type(component_type), "other")


def normalized_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean_str(value).lower())


def reference_evidence_variants(field: str, value: Any) -> set[str]:
    text = clean_str(value)
    normalized = normalized_text(text)
    variants = {normalized} if normalized else set()
    if field != "eventDate":
        return variants
    match = re.fullmatch(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?", text)
    if not match:
        return variants
    year, month, day = match.groups()
    if not month:
        return variants
    month_names = (
        "", "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    )
    month_number = int(month)
    if not 1 <= month_number <= 12:
        return variants
    if day:
        variants.update({
            f"{day}{month}{year}",
            f"{day}{month_names[month_number]}{year}",
            f"{month_names[month_number]}{day}{year}",
        })
    else:
        variants.add(f"{month_names[month_number]}{year}")
    return variants


def reference_evidence_hit(field: str, value: Any, evidence: Any) -> bool:
    evidence_key = normalized_text(evidence)
    return bool(evidence_key) and any(
        variant in evidence_key
        for variant in reference_evidence_variants(field, value)
    )


def text_similarity(left: Any, right: Any) -> float:
    left_text = normalized_text(left)
    right_text = normalized_text(right)
    if not left_text or not right_text:
        return 0.0
    return SequenceMatcher(None, left_text, right_text).ratio()


def identifier_like(value: Any) -> bool:
    key = re.sub(r"[^A-Za-z0-9]+", "", clean_str(value)).upper()
    return (
        5 <= len(key) <= 24
        and any(character.isalpha() for character in key)
        and any(character.isdigit() for character in key)
    )


def candidate_utility(candidate: dict[str, Any]) -> float:
    component_type = canonical_component_type(candidate.get("component_type"))
    role = component_role(component_type)
    detector = clean_str(candidate.get("detector_family")).lower()
    confidence = float(candidate.get("detector_confidence") or 0.0)
    text = clean_str(candidate.get("raw_text"))
    decoded = clean_str(candidate.get("decoded_barcode"))
    sharpness = min(float(candidate.get("sharpness_score") or 0.0), 5000.0)
    area_fraction = float(candidate.get("area_fraction") or 0.0)

    score = confidence * 100.0
    score += min(len(text), 160) / 8.0
    score += min(sharpness / 250.0, 12.0)
    if text:
        score += 18.0
    if decoded:
        score += 70.0
    if identifier_like(decoded or text):
        score += 35.0
    if role == "identifier_child":
        score += 18.0
    elif role == "identifier_parent":
        score += 12.0
    elif role == "primary_label":
        score += 15.0
    elif role == "annotation":
        score += 8.0
    if detector == "hespi" and role in {"primary_label", "annotation"}:
        score += 12.0
    if detector == "yoloe26" and role.startswith("identifier"):
        score += 8.0
    if area_fraction > 0.60:
        score -= 35.0
    elif area_fraction > 0.40 and role != "primary_label":
        score -= 20.0
    return round(score, 6)


def evidence_gate(
    candidate: dict[str, Any],
    *,
    class_thresholds: dict[str, float] | None = None,
    vision_only_confidence: float = 0.20,
) -> tuple[bool, str]:
    component_type = canonical_component_type(candidate.get("component_type"))
    role = component_role(component_type)
    if role == "other":
        return False, "unsupported_component_type"
    thresholds = {**DEFAULT_CLASS_THRESHOLDS, **(class_thresholds or {})}
    confidence = float(candidate.get("detector_confidence") or 0.0)
    if confidence < float(thresholds.get(component_type, 0.10)):
        return False, "below_class_threshold"
    text = clean_str(candidate.get("raw_text"))
    decoded = clean_str(candidate.get("decoded_barcode"))
    if text or decoded:
        return True, "direct_text_evidence"
    if role in {"primary_label", "annotation"} and confidence >= vision_only_confidence:
        return True, "vision_only_high_confidence"
    return False, "empty_evidence"


def duplicate_reason(
    candidate: dict[str, Any],
    existing: dict[str, Any],
    *,
    iou_threshold: float = 0.45,
    containment_threshold: float = 0.85,
    text_threshold: float = 0.85,
) -> str:
    left_role = component_role(candidate.get("component_type"))
    right_role = component_role(existing.get("component_type"))
    role_pair = {left_role, right_role}
    if role_pair == {"identifier_parent", "identifier_child"}:
        return ""
    if "primary_label" in role_pair and len(role_pair) > 1:
        return ""

    left_bbox = parse_bbox(candidate.get("bbox_xyxy", candidate.get("bbox")))
    right_bbox = parse_bbox(existing.get("bbox_xyxy", existing.get("bbox")))
    overlap = bbox_iou(left_bbox, right_bbox)
    containment = containment_ratio(left_bbox, right_bbox)
    mask_overlap = max(
        _pairwise_mask_iou(candidate, existing),
        _pairwise_mask_iou(existing, candidate),
    )
    similarity = text_similarity(
        candidate.get("decoded_barcode") or candidate.get("raw_text"),
        existing.get("decoded_barcode") or existing.get("raw_text"),
    )

    if left_role == right_role and (
        overlap >= iou_threshold
        or containment >= containment_threshold
        or mask_overlap >= iou_threshold
    ):
        return "same_role_geometry_duplicate"
    if similarity >= text_threshold and (
        containment >= containment_threshold or overlap >= 0.25
    ):
        return "same_text_cross_class_duplicate"
    if overlap >= 0.75:
        return "high_iou_class_conflict"
    return ""


def _pairwise_mask_iou(candidate: dict[str, Any], existing: dict[str, Any]) -> float:
    values = candidate.get("mask_iou_by_region_id", {})
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except json.JSONDecodeError:
            values = {}
    if not isinstance(values, dict):
        return 0.0
    try:
        return float(values.get(clean_str(existing.get("region_id")), 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def hierarchical_deduplicate(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item.get("utility_score", candidate_utility(item))),
            clean_str(item.get("region_id")),
        ),
    )
    for candidate in ranked:
        duplicate = ""
        duplicate_of = ""
        for existing in kept:
            reason = duplicate_reason(candidate, existing)
            if reason:
                duplicate = reason
                duplicate_of = clean_str(existing.get("region_id"))
                break
        if duplicate:
            rejected.append({
                **candidate,
                "selection_status": "rejected_duplicate",
                "selection_reason": duplicate,
                "duplicate_of_region_id": duplicate_of,
            })
        else:
            kept.append(candidate)
    return kept, rejected


def select_diverse_crops(
    candidates: list[dict[str, Any]],
    *,
    total_limit: int = 8,
    role_quotas: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    quotas = {**DEFAULT_ROLE_QUOTAS, **(role_quotas or {})}
    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item.get("utility_score", candidate_utility(item))),
            clean_str(item.get("region_id")),
        ),
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    role_counts: dict[str, int] = {role: 0 for role in quotas}

    for role in ROLE_ORDER:
        if len(selected) >= max(0, total_limit):
            break
        for candidate in ranked:
            region_id = clean_str(candidate.get("region_id"))
            if region_id in selected_ids or component_role(candidate.get("component_type")) != role:
                continue
            if role_counts.get(role, 0) >= int(quotas.get(role, 0)):
                break
            selected.append(candidate)
            selected_ids.add(region_id)
            role_counts[role] = role_counts.get(role, 0) + 1
            break

    for candidate in ranked:
        if len(selected) >= max(0, total_limit):
            break
        region_id = clean_str(candidate.get("region_id"))
        role = component_role(candidate.get("component_type"))
        if region_id in selected_ids:
            continue
        if role_counts.get(role, 0) >= int(quotas.get(role, 0)):
            continue
        selected.append(candidate)
        selected_ids.add(region_id)
        role_counts[role] = role_counts.get(role, 0) + 1

    rejected = [
        {
            **candidate,
            "selection_status": "rejected_quota",
            "selection_reason": "role_or_total_quota",
            "duplicate_of_region_id": "",
        }
        for candidate in ranked
        if clean_str(candidate.get("region_id")) not in selected_ids
    ]
    selected_rows = [
        {
            **candidate,
            "selection_status": "selected",
            "selection_reason": "diverse_role_utility",
            "duplicate_of_region_id": "",
            "input_order": index,
        }
        for index, candidate in enumerate(selected, start=1)
    ]
    return selected_rows, rejected


def selected_duplicate_pair_count(candidates: list[dict[str, Any]]) -> int:
    total = 0
    for index, candidate in enumerate(candidates):
        for existing in candidates[:index]:
            if duplicate_reason(candidate, existing):
                total += 1
    return total
