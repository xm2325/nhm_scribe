from __future__ import annotations

import re
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


def assign_ocr_tertiles(scores: pd.Series) -> pd.Series:
    filled = scores.fillna(0.0)
    if filled.nunique() < 3:
        return pd.Series(np.where(filled >= filled.median(), "high", "low"), index=scores.index)
    return pd.qcut(filled, q=3, labels=["low", "medium", "high"], duplicates="drop").astype(str)


def evaluate_predictions(pred_df: pd.DataFrame, gold_df: pd.DataFrame, ocr_df: pd.DataFrame, fields: list[str], paths: dict[str, Path] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gold = gold_df.set_index("occurrenceID")
    ocr_text = ocr_df.groupby("occurrenceID")["ocr_text"].apply("\n".join).to_dict() if len(ocr_df) else {}
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
        for field in fields:
            pred_val = clean_str(prow.get(field, ""))
            gold_val = clean_str(gold.loc[occ].get(field, ""))
            rows.append({
                "occurrenceID": occ,
                "method": prow.get("method", "unknown"),
                "field": field,
                "prediction": pred_val,
                "gold": gold_val,
                "coverage": int(bool(pred_val)),
                "exact_match": exact_match(pred_val, gold_val) if gold_val else np.nan,
                "token_f1": token_f1(pred_val, gold_val) if gold_val else np.nan,
                "validation_warning": int(bool(clean_str(prow.get("validation_warnings", "")))),
                "parse_failure": int(bool(prow.get("parse_failure", False))),
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
            n=("occurrenceID", "nunique"),
        )
        strat = detail.groupby(["method", "ocr_quality_tertile"], as_index=False).agg(
            exact_match=("exact_match", "mean"),
            token_f1=("token_f1", "mean"),
            coverage=("coverage", "mean"),
            n_records=("occurrenceID", "nunique"),
        )
    else:
        summary = pd.DataFrame()
        strat = pd.DataFrame()
    if paths:
        detail.to_csv(paths["processed"] / "eval_detail.csv", index=False)
        summary.to_csv(paths["processed"] / "eval_summary.csv", index=False)
        strat.to_csv(paths["processed"] / "eval_stratified_by_ocr_tertile.csv", index=False)
        proxy.to_csv(paths["processed"] / "ocr_proxy_field_presence.csv", index=False)
    return detail, summary, strat
