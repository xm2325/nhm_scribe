from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

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
_CLIP_MODELS: dict[str, tuple[Any, Any]] = {}


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


def assert_no_rag_leakage(
    eval_occurrence_ids: set[str],
    reference_occurrence_ids: set[str],
) -> None:
    overlap = {clean_str(value) for value in eval_occurrence_ids} & {
        clean_str(value) for value in reference_occurrence_ids
    }
    overlap.discard("")
    if overlap:
        raise ValueError(f"RAG reference corpus contains EVAL records: {sorted(overlap)[:10]}")


def _tfidf_vectors(texts: list[str]) -> np.ndarray:
    token_rows = [tokenize(text) for text in texts]
    document_frequency = Counter()
    for tokens in token_rows:
        document_frequency.update(set(tokens))
    vocabulary = sorted(document_frequency)
    if not vocabulary:
        return np.zeros((len(texts), 0), dtype=np.float32)
    positions = {token: index for index, token in enumerate(vocabulary)}
    matrix = np.zeros((len(texts), len(vocabulary)), dtype=np.float32)
    n_documents = max(1, len(texts))
    for row_index, tokens in enumerate(token_rows):
        counts = Counter(tokens)
        length = max(1, sum(counts.values()))
        for token, count in counts.items():
            inverse_document_frequency = math.log((1 + n_documents) / (1 + document_frequency[token])) + 1
            matrix[row_index, positions[token]] = (count / length) * inverse_document_frequency
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


def tfidf_cosine_scores(query: str, texts: list[str]) -> list[float]:
    vectors = _tfidf_vectors([query, *texts])
    if vectors.shape[1] == 0:
        return [0.0] * len(texts)
    return (vectors[1:] @ vectors[0]).astype(float).tolist()


def tfidf_embeddings(texts: list[str]) -> np.ndarray:
    return _tfidf_vectors(texts)


def clip_image_embeddings(
    image_paths: list[str],
    *,
    model_name: str = "openai/clip-vit-base-patch32",
    enabled: bool = True,
) -> tuple[np.ndarray | None, str]:
    if not enabled:
        return None, "disabled"
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except Exception as exc:
        return None, f"unavailable:{type(exc).__name__}"
    valid_images = []
    try:
        for value in image_paths:
            path = Path(clean_str(value))
            if not path.exists():
                return None, f"missing_image:{path}"
            valid_images.append(Image.open(path).convert("RGB"))
        cached = _CLIP_MODELS.get(model_name)
        if cached is None:
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
            _CLIP_MODELS[model_name] = (processor, model)
        else:
            processor, model = cached
        inputs = processor(images=valid_images, return_tensors="pt", padding=True)
        with torch.no_grad():
            vectors = model.get_image_features(**inputs).cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)
        return vectors, "ok"
    except Exception as exc:
        return None, f"error:{type(exc).__name__}:{str(exc)[:200]}"


def inner_product_scores(query: np.ndarray, references: np.ndarray) -> list[float]:
    query = np.asarray(query, dtype=np.float32).reshape(1, -1)
    references = np.asarray(references, dtype=np.float32)
    try:
        import faiss

        index = faiss.IndexFlatIP(references.shape[1])
        index.add(references)
        scores, order = index.search(query, len(references))
        result = np.zeros(len(references), dtype=np.float32)
        for score, index_value in zip(scores[0], order[0]):
            result[int(index_value)] = float(score)
        return result.tolist()
    except Exception:
        return (references @ query[0]).astype(float).tolist()


def retrieve_hybrid_references(
    *,
    query_text: str,
    query_visual_embedding: np.ndarray | None,
    query_institution_code: str,
    references: list[dict[str, Any]],
    reference_visual_embeddings: np.ndarray | None = None,
    top_k: int = 3,
    visual_weight: float = 0.45,
    text_weight: float = 0.45,
    institution_bonus_weight: float = 0.10,
) -> list[dict[str, Any]]:
    texts = [clean_str(item.get("text", "")) for item in references]
    tfidf_scores = tfidf_cosine_scores(query_text, texts)
    lexical_scores = [
        float(item.get("score", 0.0))
        for item in retrieve_context(query_text, [
            {"source": "reference", "text": text, "row": index}
            for index, text in enumerate(texts)
        ], top_k=max(1, len(texts)))
    ]
    lexical_by_row = {}
    for item in retrieve_context(query_text, [
        {"source": "reference", "text": text, "row": index}
        for index, text in enumerate(texts)
    ], top_k=max(1, len(texts))):
        lexical_by_row[int(item["row"])] = float(item["score"])
    text_scores = [
        (tfidf_scores[index] + lexical_by_row.get(index, 0.0)) / 2
        for index in range(len(references))
    ]
    visual_scores = [0.0] * len(references)
    if query_visual_embedding is not None and reference_visual_embeddings is not None:
        visual_scores = inner_product_scores(query_visual_embedding, reference_visual_embeddings)
    rows = []
    for index, reference in enumerate(references):
        institution_bonus = float(
            bool(clean_str(query_institution_code))
            and clean_str(reference.get("institutionCode")) == clean_str(query_institution_code)
        )
        combined = (
            visual_weight * visual_scores[index]
            + text_weight * text_scores[index]
            + institution_bonus_weight * institution_bonus
        )
        rows.append({
            **reference,
            "visual_similarity": visual_scores[index],
            "text_similarity": text_scores[index],
            "institution_bonus": institution_bonus,
            "combined_similarity": combined,
        })
    rows.sort(key=lambda item: item["combined_similarity"], reverse=True)
    return [{**item, "rank": rank} for rank, item in enumerate(rows[:max(0, top_k)], start=1)]
