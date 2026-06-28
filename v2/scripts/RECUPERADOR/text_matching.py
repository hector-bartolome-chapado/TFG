from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any


def get_search_text(row: dict[str, Any]) -> str:
    value = row.get("retrieval_text") or row.get("text") or ""
    return str(value)


def tokenize_for_bm25(text: str) -> list[str]:
    pattern = r"\d+(?:[.,]\d+)*%?|[^\W\d_]+"
    return [match.group(0).lower() for match in re.finditer(pattern, text, flags=re.UNICODE)]


def normalize_for_ranking(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def bm25_scores(
    query: str,
    embedding_rows: list[dict[str, Any]],
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    query_tokens = tokenize_for_bm25(query)
    documents = [tokenize_for_bm25(get_search_text(row)) for row in embedding_rows]
    if not embedding_rows:
        return {}

    doc_count = len(documents)
    lengths = [len(document) for document in documents]
    avg_length = sum(lengths) / doc_count if doc_count else 0.0
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(document))

    scores: dict[str, float] = {}
    for row, document, length in zip(embedding_rows, documents, lengths):
        term_frequency = Counter(document)
        score = 0.0
        for token in query_tokens:
            frequency = term_frequency.get(token, 0)
            if frequency == 0:
                continue
            idf = math.log(1 + (doc_count - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5))
            denominator = frequency + k1 * (1 - b + b * (length / avg_length if avg_length else 0.0))
            score += idf * (frequency * (k1 + 1)) / denominator
        scores[row["chunk_id"]] = score
    return scores


def normalized_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    min_score = min(scores.values())
    if max_score == min_score:
        return {key: 1.0 if max_score > 0 else 0.0 for key in scores}
    return {key: (value - min_score) / (max_score - min_score) for key, value in scores.items()}


def reciprocal_rank_fusion(
    vector_ids: list[str],
    bm25_ids: list[str],
    rrf_k: int = 60,
) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranked_ids in (vector_ids, bm25_ids):
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return fused
