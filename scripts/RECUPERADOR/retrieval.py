from __future__ import annotations

import json
import math
import pathlib
import re
import unicodedata
from collections import Counter
from typing import Any, Callable

from scripts.simple_extractor.config import DEFAULT_EMBED_MODEL, DEFAULT_LLAMUS_BASE_URL
from scripts.simple_extractor.embeddings import request_embedding

FISCAL_QUERY_EXPANSIONS = {
    "irpf": "Impuesto sobre la Renta de las Personas Físicas retenciones renta hogares salarios pensiones",
    "iva": "Impuesto sobre el Valor Añadido gasto sujeto IVA",
    "is": "Impuesto sobre Sociedades beneficios sociedades",
    "iiee": "Impuestos Especiales Hidrocarburos Tabaco Electricidad Alcohol",
    "ii.ee": "Impuestos Especiales Hidrocarburos Tabaco Electricidad Alcohol",
    "sociedades": "Impuesto sobre Sociedades beneficios sociedades",
    "hidrocarburos": "Impuesto sobre Hidrocarburos gasolinas gasóleos",
    "tabaco": "Impuesto sobre Labores del Tabaco labores del tabaco",
    "electricidad": "Impuesto sobre la Electricidad IVA electricidad gas natural",
}

LEGAL_QUERY_TERMS = (
    "ley",
    "real decreto",
    "decreto",
    "articulo",
    "art.",
    "norma",
    "normativa",
    "regula",
    "establece",
    "ambito de aplicacion",
    "hecho imponible",
    "texto consolidado",
)

TABLE_QUERY_TERMS = (
    "cuadro",
    "tabla",
    "fila",
    "columna",
    "presupuesto",
    "presupuestos",
    "capitulo",
    "capitulos",
    "transferencias",
    "operaciones",
    "gastos",
)


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


def get_search_text(row: dict[str, Any]) -> str:
    value = row.get("retrieval_text") or row.get("text") or ""
    return str(value)


def expand_fiscal_query(question: str) -> str:
    tokens = set(tokenize_for_bm25(question))
    expansions: list[str] = []
    for token, expansion in FISCAL_QUERY_EXPANSIONS.items():
        normalized_token = token.replace(".", "")
        if token in tokens or normalized_token in {item.replace(".", "") for item in tokens}:
            expansions.append(expansion)
    if not expansions:
        return question
    return f"{question} {' '.join(expansions)}"


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


def tokenize_for_bm25(text: str) -> list[str]:
    pattern = r"\d+(?:[.,]\d+)*%?|[^\W\d_]+"
    return [match.group(0).lower() for match in re.finditer(pattern, text, flags=re.UNICODE)]


def normalize_for_ranking(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def is_legal_query(question: str) -> bool:
    normalized = normalize_for_ranking(question)
    return any(term in normalized for term in LEGAL_QUERY_TERMS)


def is_table_query(question: str) -> bool:
    normalized = normalize_for_ranking(question)
    return any(term in normalized for term in TABLE_QUERY_TERMS)


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


def normalized_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    min_score = min(scores.values())
    if max_score == min_score:
        return {key: 1.0 if max_score > 0 else 0.0 for key in scores}
    return {key: (value - min_score) / (max_score - min_score) for key, value in scores.items()}


def fiscal_exact_score(question: str, row: dict[str, Any]) -> float:
    query_tokens = set(tokenize_for_bm25(expand_fiscal_query(question)))
    text_tokens = set(tokenize_for_bm25(get_search_text(row)))
    important_tokens = {
        token
        for token in query_tokens
        if token.isupper()
        or any(char.isdigit() for char in token)
        or token in {"irpf", "iva", "is", "iiee", "sociedades", "hidrocarburos", "tabaco", "electricidad"}
    }
    if not important_tokens:
        return 0.0
    return len(important_tokens & text_tokens) / len(important_tokens)


def legal_relevance_score(row: dict[str, Any]) -> float:
    text = normalize_for_ranking(
        " ".join(
            str(row.get(field) or "")
            for field in ("text", "retrieval_text", "parent_text", "context_text")
        )
    )
    score = 0.0
    if "indice" in text and "articulo" not in text:
        score -= 0.65
    if "articulo" in text:
        score += 0.45
    if "naturaleza del impuesto" in text:
        score += 0.35
    if "ambito espacial de aplicacion" in text or "territorialidad" in text:
        score += 0.25
    if "titulo preliminar" in text:
        score += 0.15
    if "ley 37/1992" in text:
        score += 0.05
    return score


def table_relevance_score(question: str, row: dict[str, Any]) -> float:
    if row.get("chunking_strategy") != "xlsx_rows_parent_child":
        return 0.0
    text = str(row.get("retrieval_text") or row.get("text") or "")
    query_tokens = set(tokenize_for_bm25(normalize_for_ranking(question)))
    requested_years = set(re.findall(r"\b20\d{2}\b", question))
    best_row_score = 0.0
    has_requested_year_header = False

    for line in text.splitlines():
        match = re.match(r"\s*Fila\s+\d+:\s*(.+)$", line.strip(), flags=re.IGNORECASE)
        if not match or "|" not in match.group(1):
            continue
        cells = [cell.strip() for cell in match.group(1).split("|") if cell.strip()]
        if len(cells) < 2:
            continue
        normalized_cells = [normalize_for_ranking(cell) for cell in cells]
        if requested_years and any(
            header == year or header == f"{year}-p"
            for year in requested_years
            for header in normalized_cells[1:]
        ):
            has_requested_year_header = True
        label_tokens = set(tokenize_for_bm25(normalize_for_ranking(cells[0])))
        overlap = len(query_tokens & label_tokens)
        label_score = float(overlap)
        normalized_label = normalize_for_ranking(cells[0])
        normalized_question = normalize_for_ranking(question)
        if normalized_label and normalized_label in normalized_question:
            label_score += 2.0
        if "total" in query_tokens and "total" in label_tokens:
            label_score += 1.0
        if "presupuesto" in query_tokens and "presupuesto" in label_tokens:
            label_score += 1.0
        best_row_score = max(best_row_score, label_score)

    if best_row_score <= 0:
        return 0.0
    score = min(best_row_score / 6.0, 1.0)
    if has_requested_year_header:
        score += 0.25
    return score


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
