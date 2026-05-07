from __future__ import annotations

import json
import math
import pathlib
from typing import Any, Callable

from scripts.simple_extractor.config import DEFAULT_EMBED_MODEL, DEFAULT_LLAMUS_BASE_URL
from scripts.simple_extractor.embeddings import request_embedding


def load_embedding_rows(embeddings_path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with embeddings_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row.get("embedding"), list):
                raise ValueError(f"Fila sin embedding usable en {embeddings_path}.")
            rows.append(row)
    return rows


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("No se puede calcular coseno con vectores de distinta dimensiÃ³n.")

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def rank_chunks_by_similarity(
    query_embedding: list[float],
    embedding_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranked_rows: list[dict[str, Any]] = []
    for row in embedding_rows:
        ranked_rows.append(
            {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "text": row["text"],
                "score": cosine_similarity(query_embedding, row["embedding"]),
            }
        )
    return sorted(ranked_rows, key=lambda row: row["score"], reverse=True)


def retrieve_top_k(
    question: str,
    embedding_rows: list[dict[str, Any]],
    top_k: int = 5,
    model: str = DEFAULT_EMBED_MODEL,
    base_url: str = DEFAULT_LLAMUS_BASE_URL,
    api_key: str | None = None,
    embedder: Callable[[str, str, str, str | None], list[float]] | None = None,
) -> list[dict[str, Any]]:
    effective_embedder = embedder or request_embedding
    query_embedding = effective_embedder(question, model, base_url, api_key)
    ranked_rows = rank_chunks_by_similarity(query_embedding, embedding_rows)
    return ranked_rows[:top_k]
