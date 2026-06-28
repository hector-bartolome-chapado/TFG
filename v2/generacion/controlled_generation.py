from __future__ import annotations

import re
import time
from typing import Any

from generacion.context_builder import build_context
from generacion.legal_answers import extract_legal_answer
from generacion.question_routing import classify_question_route
from generacion.table_answers import (
    extract_document_presence_answer,
    extract_single_row_table_answer,
    extract_table_cell_answer,
)
from generacion.text_utils import (
    STANDARD_NO_ANSWER,
    content_tokens,
    lower_first,
    normalize_for_generation,
    question_subject,
)


def restate_answer(question: str, answer: str) -> str:
    cleaned_answer = answer.strip()
    if not cleaned_answer:
        return cleaned_answer
    if cleaned_answer.startswith(("Dato principal:", "Cuadro o tabla:", "Norma:", "Documentos localizados:")):
        return cleaned_answer
    subject = question_subject(question)
    if not subject:
        return cleaned_answer
    if cleaned_answer.startswith(STANDARD_NO_ANSWER):
        return f"No hay informacion suficiente en el documento proporcionado para responder sobre {subject}."
    if normalize_for_generation(cleaned_answer).startswith(("sobre ", "respecto a ")):
        return cleaned_answer
    prefix = "Respecto a" if classify_question_route(question) == "table" else "Sobre"
    return f"{prefix} {subject}, {lower_first(cleaned_answer)}"


def build_generation_decision(question: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    route = classify_question_route(question)
    context = build_context(hits)
    query_tokens = content_tokens(question)
    context_tokens = content_tokens(context)
    overlap = len(query_tokens & context_tokens) / len(query_tokens) if query_tokens else 0.0
    top_score = float(hits[0].get("score", 0.0)) if hits else 0.0
    should_reject = False
    if not hits or not context.strip():
        should_reject = True
    elif overlap < 0.14:
        should_reject = True
    elif overlap < 0.18 and top_score < 0.25:
        should_reject = True
    return {
        "route": route,
        "should_reject": should_reject,
        "answer": restate_answer(question, STANDARD_NO_ANSWER) if should_reject else "",
        "top_score": top_score,
        "query_context_overlap": overlap,
    }


def score_sentence_for_question(sentence: str, question: str, route: str) -> float:
    sentence_tokens = content_tokens(sentence)
    query_tokens = content_tokens(question)
    if not sentence_tokens or not query_tokens:
        return 0.0
    overlap = len(sentence_tokens & query_tokens) / len(query_tokens)
    number_bonus = 0.15 if re.search(r"\d", sentence) else 0.0
    table_bonus = 0.15 if route == "table" and any(
        term in normalize_for_generation(sentence) for term in ("cuadro", "tabla", "nota informativa")
    ) else 0.0
    return overlap + number_bonus + table_bonus


def extract_relevant_sentences(question: str, context: str, route: str, limit: int) -> list[str]:
    parts = [
        part.strip(" -|\t")
        for part in re.split(r"(?<=[.!?])\s+|\n+", context)
        if part.strip(" -|\t")
    ]
    ranked = sorted(
        parts,
        key=lambda part: score_sentence_for_question(part, question, route),
        reverse=True,
    )
    selected: list[str] = []
    for part in ranked:
        if part not in selected and score_sentence_for_question(part, question, route) > 0:
            selected.append(part)
        if len(selected) >= limit:
            break
    return selected


def generate_controlled_answer(question: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    started_at = time.perf_counter()
    decision = build_generation_decision(question, hits)
    if decision["should_reject"]:
        return {
            "answer": decision["answer"],
            "latency_seconds": time.perf_counter() - started_at,
            "decision": decision,
        }

    context = build_context(hits)
    route = decision["route"]
    for extractor in (
        lambda: extract_document_presence_answer(question, hits),
        lambda: extract_table_cell_answer(question, context),
        lambda: extract_single_row_table_answer(question, context),
        lambda: extract_legal_answer(question, context) if route == "legal" else None,
    ):
        extracted = extractor()
        if extracted:
            return {
                "answer": extracted,
                "latency_seconds": time.perf_counter() - started_at,
                "decision": decision,
            }

    sentence_limit = 3 if route in {"synthesis", "legal"} else 2
    sentences = extract_relevant_sentences(question, context, route, sentence_limit)
    if not sentences:
        answer = STANDARD_NO_ANSWER
        decision = {**decision, "should_reject": True}
    elif route in {"tax_exact", "table", "chart"}:
        answer = " ".join(sentences[:2])
    elif route in {"synthesis", "legal"}:
        answer = " ".join(sentences[:3])
    else:
        answer = sentences[0]
    return {
        "answer": restate_answer(question, answer.strip()),
        "latency_seconds": time.perf_counter() - started_at,
        "decision": decision,
    }

