from __future__ import annotations

import json
import math
import pathlib
import re
from typing import Any, Callable

from recuperacion.fiscal_scoring import expand_fiscal_query, fiscal_exact_score
from recuperacion.legal_scoring import is_legal_query, legal_relevance_score
from recuperacion.table_scoring import is_table_query, table_relevance_score
from recuperacion.text_matching import (
    bm25_scores,
    normalized_scores,
    reciprocal_rank_fusion,
    tokenize_for_bm25,
)
from ingesta.config import DEFAULT_EMBED_MODEL, DEFAULT_LLAMUS_BASE_URL
from ingesta.embeddings import request_embedding


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
        raise ValueError("No se puede calcular coseno con vectores de distinta dimension.")

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def build_focused_context(
    child_text: str,
    parent_text: str,
    question: str,
    max_chars: int = 1200,
) -> str:
    normalized_query_tokens = set(tokenize_for_bm25(expand_fiscal_query(question)))
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", parent_text.replace("\n", " ")) if part.strip()]
    ranked_sentences = sorted(
        sentences,
        key=lambda sentence: sum(1 for token in tokenize_for_bm25(sentence) if token in normalized_query_tokens),
        reverse=True,
    )
    selected: list[str] = []
    if child_text.strip():
        selected.append(child_text.strip())
    for sentence in ranked_sentences:
        if sentence not in selected:
            selected.append(sentence)
        if len(" ".join(selected)) >= max_chars:
            break
    focused = " ".join(selected)
    if len(focused) <= max_chars:
        return focused
    return focused[:max_chars].rsplit(" ", 1)[0].strip()


def copy_retrieval_fields(row: dict[str, Any], score: float) -> dict[str, Any]:
    ranked_row = {
        "chunk_id": row["chunk_id"],
        "doc_id": row["doc_id"],
        "text": row["text"],
        "score": score,
    }
    for field in (
        "parent_id",
        "parent_text",
        "page_start",
        "page_end",
        "chunking_strategy",
        "retrieval_text",
        "context_text",
        "source_file",
        "sheet",
        "sheet_index",
        "row_start",
        "row_end",
    ):
        if field in row:
            ranked_row[field] = row[field]
    return ranked_row


def rank_chunks_by_similarity(
    query_embedding: list[float],
    embedding_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranked_rows: list[dict[str, Any]] = []
    for row in embedding_rows:
        vector_score = cosine_similarity(query_embedding, row["embedding"])
        ranked_row = copy_retrieval_fields(row, vector_score)
        ranked_row["vector_score"] = vector_score
        ranked_rows.append(ranked_row)
    return sorted(ranked_rows, key=lambda row: row["score"], reverse=True)


def rank_chunks_hybrid(
    question: str,
    query_embedding: list[float],
    embedding_rows: list[dict[str, Any]],
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    vector_ranked = rank_chunks_by_similarity(query_embedding, embedding_rows)
    bm25_by_id = bm25_scores(question, embedding_rows)
    rows_by_id = {row["chunk_id"]: row for row in embedding_rows}
    vector_rank_by_id = {row["chunk_id"]: rank for rank, row in enumerate(vector_ranked, start=1)}
    bm25_ids = sorted(bm25_by_id, key=lambda chunk_id: bm25_by_id[chunk_id], reverse=True)
    bm25_rank_by_id = {chunk_id: rank for rank, chunk_id in enumerate(bm25_ids, start=1)}
    fused_by_id = reciprocal_rank_fusion(
        vector_ids=[row["chunk_id"] for row in vector_ranked],
        bm25_ids=bm25_ids,
        rrf_k=rrf_k,
    )

    ranked_rows: list[dict[str, Any]] = []
    for chunk_id, rrf_score in fused_by_id.items():
        source_row = rows_by_id[chunk_id]
        vector_score = vector_ranked[vector_rank_by_id[chunk_id] - 1]["vector_score"]
        ranked_row = copy_retrieval_fields(source_row, rrf_score)
        ranked_row["vector_score"] = vector_score
        ranked_row["bm25_score"] = bm25_by_id.get(chunk_id, 0.0)
        ranked_row["rrf_score"] = rrf_score
        ranked_row["vector_rank"] = vector_rank_by_id.get(chunk_id)
        ranked_row["bm25_rank"] = bm25_rank_by_id.get(chunk_id)
        ranked_rows.append(ranked_row)

    return sorted(
        ranked_rows,
        key=lambda row: (
            row["rrf_score"],
            row["bm25_score"],
            row["vector_score"],
        ),
        reverse=True,
    )


def rank_chunks_fiscal_hybrid(
    question: str,
    query_embedding: list[float],
    embedding_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expanded_question = expand_fiscal_query(question)
    vector_ranked = rank_chunks_by_similarity(query_embedding, embedding_rows)
    vector_scores = {row["chunk_id"]: row["vector_score"] for row in vector_ranked}
    bm25_by_id = bm25_scores(expanded_question, embedding_rows)
    vector_norm = normalized_scores(vector_scores)
    bm25_norm = normalized_scores(bm25_by_id)
    vector_rank_by_id = {row["chunk_id"]: rank for rank, row in enumerate(vector_ranked, start=1)}
    bm25_ids = sorted(bm25_by_id, key=lambda chunk_id: bm25_by_id[chunk_id], reverse=True)
    bm25_rank_by_id = {chunk_id: rank for rank, chunk_id in enumerate(bm25_ids, start=1)}
    has_exact_query = any(char.isdigit() for char in question) or any(
        token in set(tokenize_for_bm25(question))
        for token in {"irpf", "iva", "is", "iiee", "sociedades", "hidrocarburos", "tabaco", "electricidad"}
    )
    vector_weight = 0.35 if has_exact_query else 0.55
    bm25_weight = 0.50 if has_exact_query else 0.35
    exact_weight = 0.15 if has_exact_query else 0.10
    legal_query = is_legal_query(question)
    table_query = is_table_query(question)

    ranked_rows: list[dict[str, Any]] = []
    for row in embedding_rows:
        chunk_id = row["chunk_id"]
        lexical_boost = fiscal_exact_score(question, row)
        legal_boost = legal_relevance_score(row) if legal_query else 0.0
        table_boost = table_relevance_score(question, row) if table_query else 0.0
        score = (
            vector_weight * vector_norm.get(chunk_id, 0.0)
            + bm25_weight * bm25_norm.get(chunk_id, 0.0)
            + exact_weight * lexical_boost
            + legal_boost
            + 0.45 * table_boost
        )
        ranked_row = copy_retrieval_fields(row, score)
        ranked_row["vector_score"] = vector_scores.get(chunk_id, 0.0)
        ranked_row["bm25_score"] = bm25_by_id.get(chunk_id, 0.0)
        ranked_row["fiscal_exact_score"] = lexical_boost
        ranked_row["legal_relevance_score"] = legal_boost
        ranked_row["table_relevance_score"] = table_boost
        ranked_row["vector_rank"] = vector_rank_by_id.get(chunk_id)
        ranked_row["bm25_rank"] = bm25_rank_by_id.get(chunk_id)
        ranked_row["expanded_query"] = expanded_question
        if row.get("parent_text") and row.get("chunking_strategy") != "xlsx_rows_parent_child":
            ranked_row["context_text"] = build_focused_context(
                child_text=str(row.get("text") or ""),
                parent_text=str(row["parent_text"]),
                question=question,
            )
        ranked_rows.append(ranked_row)
    return sorted(
        ranked_rows,
        key=lambda item: (
            item["score"],
            item["legal_relevance_score"],
            item["table_relevance_score"],
            item["fiscal_exact_score"],
            item["bm25_score"],
            item["vector_score"],
        ),
        reverse=True,
    )


def retrieve_top_k(
    question: str,
    embedding_rows: list[dict[str, Any]],
    top_k: int = 5,
    model: str = DEFAULT_EMBED_MODEL,
    base_url: str = DEFAULT_LLAMUS_BASE_URL,
    api_key: str | None = None,
    embedder: Callable[[str, str, str, str | None], list[float]] | None = None,
    strategy: str = "fiscal_hybrid",
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    effective_embedder = embedder or request_embedding
    embedding_question = expand_fiscal_query(question) if strategy == "fiscal_hybrid" else question
    query_embedding = effective_embedder(embedding_question, model, base_url, api_key)
    if strategy == "dense":
        ranked_rows = rank_chunks_by_similarity(query_embedding, embedding_rows)
    elif strategy == "fiscal_hybrid":
        ranked_rows = rank_chunks_fiscal_hybrid(question, query_embedding, embedding_rows)
    elif strategy == "hybrid":
        ranked_rows = rank_chunks_hybrid(question, query_embedding, embedding_rows, rrf_k=rrf_k)
    else:
        raise ValueError(f"Estrategia de recuperacion no soportada: {strategy}")
    return ranked_rows[:top_k]

