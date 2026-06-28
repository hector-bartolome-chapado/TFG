from __future__ import annotations

from typing import Any

from scripts.RECUPERADOR.text_matching import get_search_text, tokenize_for_bm25


FISCAL_QUERY_EXPANSIONS = {
    "irpf": "Impuesto sobre la Renta de las Personas Fisicas retenciones renta hogares salarios pensiones",
    "iva": "Impuesto sobre el Valor Anadido gasto sujeto IVA",
    "is": "Impuesto sobre Sociedades beneficios sociedades",
    "iiee": "Impuestos Especiales Hidrocarburos Tabaco Electricidad Alcohol",
    "ii.ee": "Impuestos Especiales Hidrocarburos Tabaco Electricidad Alcohol",
    "sociedades": "Impuesto sobre Sociedades beneficios sociedades",
    "hidrocarburos": "Impuesto sobre Hidrocarburos gasolinas gasoleos",
    "tabaco": "Impuesto sobre Labores del Tabaco labores del tabaco",
    "electricidad": "Impuesto sobre la Electricidad IVA electricidad gas natural",
}


def expand_fiscal_query(question: str) -> str:
    tokens = set(tokenize_for_bm25(question))
    expansions: list[str] = []
    normalized_tokens = {item.replace(".", "") for item in tokens}
    for token, expansion in FISCAL_QUERY_EXPANSIONS.items():
        normalized_token = token.replace(".", "")
        if token in tokens or normalized_token in normalized_tokens:
            expansions.append(expansion)
    if not expansions:
        return question
    return f"{question} {' '.join(expansions)}"


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
