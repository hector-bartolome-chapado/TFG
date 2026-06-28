from __future__ import annotations

from typing import Any

from recuperacion.text_matching import normalize_for_ranking


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


def is_legal_query(question: str) -> bool:
    normalized = normalize_for_ranking(question)
    return any(term in normalized for term in LEGAL_QUERY_TERMS)


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

