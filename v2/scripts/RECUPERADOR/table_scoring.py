from __future__ import annotations

import re
from typing import Any

from scripts.RECUPERADOR.text_matching import normalize_for_ranking, tokenize_for_bm25


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


def is_table_query(question: str) -> bool:
    normalized = normalize_for_ranking(question)
    return any(term in normalized for term in TABLE_QUERY_TERMS)


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
