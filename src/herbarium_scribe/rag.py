from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import pandas as pd

from .metadata import clean_str

FIELD_GUIDE_DOCS = [
    "catalogNumber: specimen barcode or accession number printed on the sheet label.",
    "scientificName: Latin binomial or full taxon name, usually genus followed by species.",
    "recordedBy: collector or collecting team name.",
    "eventDate: collection date, often a year or ISO-like date.",
    "country and stateProvince: geography written on the label.",
    "decimalLatitude and decimalLongitude: signed decimal coordinates when present.",
    "typeStatus: holotype, isotype, lectotype, syntype, or paratype when present.",
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", clean_str(text).lower())


def build_rag_corpus(demo_df: pd.DataFrame | None = None, authority_rows: list[str] | None = None) -> list[dict[str, Any]]:
    docs = [{"source": "field_guide", "text": d} for d in FIELD_GUIDE_DOCS]
    if demo_df is not None and len(demo_df):
        for _, row in demo_df.iterrows():
            text = " | ".join(f"{k}: {row.get(k, '')}" for k in ["catalogNumber", "scientificName", "recordedBy", "country", "stateProvince"])
            docs.append({"source": "demo_example", "occurrenceID": row.get("occurrenceID", ""), "text": text})
    for item in authority_rows or []:
        docs.append({"source": "authority_cache", "text": item})
    return docs


def retrieve_context(query: str, corpus: list[dict[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
    q_tokens = tokenize(query)
    q = Counter(q_tokens)
    rows = []
    for doc in corpus:
        d = Counter(tokenize(doc.get("text", "")))
        common = set(q) & set(d)
        score = sum(q[t] * d[t] for t in common)
        norm = math.sqrt(sum(v * v for v in q.values())) * math.sqrt(sum(v * v for v in d.values()))
        score = score / norm if norm else 0.0
        rows.append({**doc, "score": score})
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[:top_k]


def format_context_for_prompt(items: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{i + 1}] {it.get('source')}: {it.get('text')}" for i, it in enumerate(items))
