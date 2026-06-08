from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metadata import clean_str


def normalize_eval(x: str) -> str:
    x = clean_str(x).lower()
    x = re.sub(r"[^\w\s.\-]", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def _normalise_catalog_number(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", clean_str(value)).upper()


def _parse_date_part(value: str) -> str:
    value = clean_str(value)
    value = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip(" .")
    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d.%m.%Y",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d %b %Y",
        "%d %B %Y",
        "%b %d %Y",
        "%B %d %Y",
    ):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass
    if re.fullmatch(r"\d{4}-\d{2}", value):
        return value
    if re.fullmatch(r"\d{4}", value):
        return value
    return normalize_eval(value)


def _normalise_event_date(value: str) -> str:
    value = clean_str(value)
    if not value:
        return ""
    parts = [value]
    slash_parts = [part.strip() for part in value.split("/")]
    if len(slash_parts) == 2 and all(re.search(r"\b\d{4}\b", part) for part in slash_parts):
        parts = slash_parts
    return "/".join(_parse_date_part(part) for part in parts)


def _normalise_person_name(value: str) -> str:
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", clean_str(value).lower())
    return " ".join(sorted(tokens))


def _normalise_type_status(value: str) -> str:
    text = normalize_eval(value)
    for status in (
        "holotype",
        "isotype",
        "lectotype",
        "isolectotype",
        "neotype",
        "isoneotype",
        "syntype",
        "isosyntype",
        "paratype",
        "isoparatype",
        "type",
    ):
        if re.search(rf"\b{status}\b", text):
            return status
    return text


def normalize_field_value(field: str, value: str) -> str:
    if field == "catalogNumber":
        return _normalise_catalog_number(value)
    if field == "eventDate":
        return _normalise_event_date(value)
    if field == "recordedBy":
        return _normalise_person_name(value)
    if field == "typeStatus":
        return _normalise_type_status(value)
    if field in {"decimalLatitude", "decimalLongitude"}:
        try:
            return f"{float(clean_str(value)):.8f}".rstrip("0").rstrip(".")
        except (TypeError, ValueError):
            return normalize_eval(value)
    return normalize_eval(value)


def field_exact_match(field: str, pred: str, gold: str, coordinate_tolerance: float = 1e-4) -> int:
    if field in {"decimalLatitude", "decimalLongitude"}:
        try:
            return int(abs(float(clean_str(pred)) - float(clean_str(gold))) <= coordinate_tolerance)
        except (TypeError, ValueError):
            pass
    return int(normalize_field_value(field, pred) == normalize_field_value(field, gold))


def field_token_f1(field: str, pred: str, gold: str, coordinate_tolerance: float = 1e-4) -> float:
    if field in {"catalogNumber", "decimalLatitude", "decimalLongitude", "typeStatus"}:
        return float(field_exact_match(field, pred, gold, coordinate_tolerance))
    p = normalize_field_value(field, pred).replace("/", " ").split()
    g = normalize_field_value(field, gold).replace("/", " ").split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    p_counts = {t: p.count(t) for t in set(p)}
    g_counts = {t: g.count(t) for t in set(g)}
    common = sum(min(p_counts.get(t, 0), g_counts.get(t, 0)) for t in p_counts)
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_eval(pred) == normalize_eval(gold))


def token_f1(pred: str, gold: str) -> float:
    p = normalize_eval(pred).split()
    g = normalize_eval(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    p_counts = {t: p.count(t) for t in set(p)}
    g_counts = {t: g.count(t) for t in set(g)}
    common = sum(min(p_counts.get(t, 0), g_counts.get(t, 0)) for t in p_counts)
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)


def evidence_proxy(value: str, ocr_text: str) -> float | None:
    v = normalize_eval(value)
    t = normalize_eval(ocr_text)
    if not v:
        return None
    if not t:
        return 0.0
    if v in t:
        return 1.0
    # simple token overlap proxy, not CER/WER
    vt = set(v.split())
    tt = set(t.split())
    return len(vt & tt) / len(vt) if vt else None


def truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, (int, float)):
        return bool(value)
    text = clean_str(value).lower()
    if text in {"", "0", "false", "f", "no", "n", "none", "nan"}:
        return False
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    return bool(text)


def assign_ocr_tertiles(scores: pd.Series) -> pd.Series:
    filled = scores.fillna(0.0)
    if filled.nunique() < 3:
        return pd.Series(np.where(filled >= filled.median(), "high", "low"), index=scores.index)
    try:
        return pd.qcut(filled, q=3, labels=["low", "medium", "high"], duplicates="drop").astype(str)
    except ValueError:
        ranks = filled.rank(method="first", pct=True)
        labels = np.select(
            [ranks <= 1 / 3, ranks <= 2 / 3],
            ["low", "medium"],
            default="high",
        )
        return pd.Series(labels, index=scores.index)


def evidence_support_status(
    predicted: bool,
    evidence_span: str,
    evidence_support_score: float | None,
    prediction_evidence_alignment: float | None,
    prediction_ocr_support_score: float | None,
) -> str:
    if not predicted:
        return "not_predicted"
    aligned = prediction_evidence_alignment is not None and prediction_evidence_alignment >= 0.5
    if evidence_support_score is not None and evidence_support_score >= 1.0 and aligned:
        return "direct"
    if evidence_support_score is not None and evidence_support_score >= 0.5 and aligned:
        return "partial_direct"
    if prediction_ocr_support_score is not None and prediction_ocr_support_score >= 1.0:
        return "prediction_in_ocr"
    if evidence_support_score is not None and evidence_support_score >= 1.0:
        return "contextual_inference"
    return "unsupported"


def review_decision(
    *,
    field: str,
    predicted: bool,
    support_status: str,
    confidence: float | None,
    validation_warning: bool,
    review_config: dict[str, Any] | None = None,
) -> tuple[bool, str, str]:
    if not predicted:
        return False, "", ""
    config = review_config or {}
    min_confidence = float(config.get("min_confidence", 0.75))
    high_risk_min_confidence = float(config.get("high_risk_min_confidence", 0.90))
    high_risk_fields = set(config.get(
        "high_risk_fields",
        ["country", "stateProvince", "eventDate", "recordedBy", "decimalLatitude", "decimalLongitude"],
    ))
    accepted_support = set(config.get("accepted_support_statuses", ["direct"]))
    reasons = []
    if support_status not in accepted_support:
        reasons.append(f"support:{support_status}")
    if confidence is None or pd.isna(confidence):
        reasons.append("confidence:missing")
    elif confidence < min_confidence:
        reasons.append(f"confidence_below:{min_confidence:g}")
    if field in high_risk_fields:
        if support_status != "direct":
            reasons.append("high_risk_without_direct_evidence")
        if confidence is None or pd.isna(confidence) or confidence < high_risk_min_confidence:
            reasons.append(f"high_risk_confidence_below:{high_risk_min_confidence:g}")
    if validation_warning:
        reasons.append("schema_validation_warning")
    if not reasons:
        return False, "", ""
    high_priority = (
        support_status == "unsupported"
        or (field in high_risk_fields and support_status != "direct")
        or (confidence is not None and not pd.isna(confidence) and confidence < 0.5)
    )
    return True, "high" if high_priority else "medium", ";".join(dict.fromkeys(reasons))


def evaluate_predictions(
    pred_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    ocr_df: pd.DataFrame,
    fields: list[str],
    paths: dict[str, Path] | None = None,
    review_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gold = gold_df.set_index("occurrenceID")
    ocr_text = ocr_df.groupby("occurrenceID")["ocr_text"].apply("\n".join).to_dict() if len(ocr_df) else {}
    barcode_candidates: dict[str, list[str]] = {}
    if len(ocr_df) and "ocr_engine" in ocr_df.columns:
        barcode_rows = ocr_df[ocr_df["ocr_engine"].astype(str).eq("zxingcpp")]
        for occurrence_id, group in barcode_rows.groupby("occurrenceID"):
            values = []
            for text in group["ocr_text"].astype(str):
                for value in text.splitlines():
                    value = clean_str(value)
                    if value and value not in values:
                        values.append(value)
            barcode_candidates[str(occurrence_id)] = values
    proxy_rows = []
    for occ, grow in gold.iterrows():
        vals = []
        for field in fields:
            score = evidence_proxy(grow.get(field, ""), ocr_text.get(occ, ""))
            if score is not None:
                vals.append(score)
        proxy_rows.append({"occurrenceID": occ, "ocr_evidence_proxy_score": float(np.mean(vals)) if vals else np.nan})
    proxy = pd.DataFrame(proxy_rows)
    proxy["ocr_quality_tertile"] = assign_ocr_tertiles(proxy["ocr_evidence_proxy_score"])
    proxy_map = proxy.set_index("occurrenceID").to_dict(orient="index")

    rows = []
    for _, prow in pred_df.iterrows():
        occ = prow["occurrenceID"]
        if occ not in gold.index:
            continue
        not_evaluated = truthy_flag(prow.get("not_evaluated", False))
        for field in fields:
            pred_val = clean_str(prow.get(field, ""))
            gold_val = clean_str(gold.loc[occ].get(field, ""))
            evidence_span = clean_str(prow.get(f"{field}_evidence_span", ""))
            evidence_support_score = evidence_proxy(evidence_span, ocr_text.get(occ, "")) if evidence_span else None
            prediction_evidence_alignment = (
                field_token_f1(field, pred_val, evidence_span)
                if pred_val and evidence_span
                else None
            )
            prediction_ocr_support_score = evidence_proxy(pred_val, ocr_text.get(occ, "")) if pred_val else None
            evaluable = bool(gold_val) and not not_evaluated
            predicted = bool(pred_val) and not not_evaluated
            support_status = evidence_support_status(
                predicted,
                evidence_span,
                evidence_support_score,
                prediction_evidence_alignment,
                prediction_ocr_support_score,
            )
            direct_evidence_supported = int(support_status == "direct") if predicted else np.nan
            confidence = pd.to_numeric(prow.get(f"{field}_confidence", np.nan), errors="coerce")
            validation_warning = bool(clean_str(prow.get("validation_warnings", "")))
            review_required, review_priority, review_reasons = review_decision(
                field=field,
                predicted=predicted,
                support_status=support_status,
                confidence=confidence,
                validation_warning=validation_warning,
                review_config=review_config,
            )
            record_barcode_candidates = barcode_candidates.get(str(occ), [])
            if predicted and field == "catalogNumber" and record_barcode_candidates:
                normalised_prediction = normalize_field_value(field, pred_val)
                normalised_candidates = {
                    normalize_field_value(field, value)
                    for value in record_barcode_candidates
                }
                barcode_reasons = []
                if len(record_barcode_candidates) > 1:
                    barcode_reasons.append("multiple_decoded_barcodes")
                if (
                    len(record_barcode_candidates) == 1
                    and normalised_prediction not in normalised_candidates
                ):
                    barcode_reasons.append("catalog_mismatch_single_decoded_barcode")
                if barcode_reasons:
                    review_required = True
                    review_priority = "high"
                    review_reasons = ";".join(
                        dict.fromkeys(
                            [
                                value
                                for value in [review_reasons, *barcode_reasons]
                                if value
                            ]
                        )
                    )
            rows.append({
                "occurrenceID": occ,
                "method": prow.get("method", "unknown"),
                "field": field,
                "prediction": pred_val,
                "gold": gold_val,
                "normalised_prediction": normalize_field_value(field, pred_val),
                "normalised_gold": normalize_field_value(field, gold_val),
                "evaluable": int(evaluable),
                "coverage": int(bool(pred_val)) if evaluable else np.nan,
                "exact_match": field_exact_match(field, pred_val, gold_val) if evaluable else np.nan,
                "token_f1": field_token_f1(field, pred_val, gold_val) if evaluable else np.nan,
                "evidence_span": evidence_span,
                "evidence_span_present": int(bool(evidence_span)) if predicted else np.nan,
                "evidence_span_ocr_support_score": evidence_support_score if predicted else np.nan,
                "prediction_evidence_alignment_score": prediction_evidence_alignment if predicted else np.nan,
                "prediction_ocr_support_score": prediction_ocr_support_score if predicted else np.nan,
                "evidence_support_status": support_status,
                "direct_evidence_supported": direct_evidence_supported,
                "unsupported_prediction": (1 - direct_evidence_supported) if predicted else np.nan,
                "prediction_confidence": confidence,
                "barcode_candidates": " | ".join(record_barcode_candidates) if field == "catalogNumber" else "",
                "barcode_candidate_count": len(record_barcode_candidates) if field == "catalogNumber" else np.nan,
                "review_required": int(review_required) if predicted else np.nan,
                "review_priority": review_priority,
                "review_reasons": review_reasons,
                "validation_warning": int(validation_warning),
                "parse_failure": int(truthy_flag(prow.get("parse_failure", False))),
                "not_evaluated": int(not_evaluated),
                "ocr_quality_tertile": proxy_map.get(occ, {}).get("ocr_quality_tertile", "unknown"),
                "ocr_evidence_proxy_score": proxy_map.get(occ, {}).get("ocr_evidence_proxy_score", np.nan),
            })
    detail = pd.DataFrame(rows)
    if len(detail):
        summary = detail.groupby(["method", "field"], as_index=False).agg(
            coverage=("coverage", "mean"),
            exact_match=("exact_match", "mean"),
            token_f1=("token_f1", "mean"),
            validation_warning_rate=("validation_warning", "mean"),
            parse_failure_rate=("parse_failure", "mean"),
            not_evaluated_rate=("not_evaluated", "mean"),
            evidence_span_present_rate=("evidence_span_present", "mean"),
            direct_evidence_support_rate=("direct_evidence_supported", "mean"),
            unsupported_prediction_rate=("unsupported_prediction", "mean"),
            review_required_rate=("review_required", "mean"),
            n=("evaluable", "sum"),
        )
        strat = detail.groupby(["method", "ocr_quality_tertile", "field"], as_index=False).agg(
            exact_match=("exact_match", "mean"),
            token_f1=("token_f1", "mean"),
            coverage=("coverage", "mean"),
            n_records=("occurrenceID", "nunique"),
        )
        strat = strat.rename(columns={"field": "field_name"})
    else:
        summary = pd.DataFrame()
        strat = pd.DataFrame()
    if paths:
        detail.to_csv(paths["processed"] / "eval_detail.csv", index=False)
        summary.to_csv(paths["processed"] / "eval_summary.csv", index=False)
        strat.to_csv(paths["processed"] / "eval_stratified_by_ocr_tertile.csv", index=False)
        proxy.to_csv(paths["processed"] / "ocr_proxy_field_presence.csv", index=False)
        review_queue = detail[detail["review_required"].eq(1)].copy()
        review_queue.to_csv(paths["processed"] / "review_queue.csv", index=False)
    return detail, summary, strat
