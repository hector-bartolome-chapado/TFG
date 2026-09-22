from __future__ import annotations

import json
import pathlib
import re
from typing import Any, Callable

from generacion.rag_service import ask_llamus, load_document_embeddings
from ingesta.config import DEFAULT_EMBED_MODEL, DEFAULT_LLAMUS_BASE_URL, DEFAULT_RAG_CHAT_MODEL


FOLLOWUP_PREFIX = re.compile(
    r"^(?:[¿¡]?\s*(?:y\b|también\b|entonces\b|en\s+(?:19|20)\d{2}\b|"
    r"qué\s+(?:hay\s+de|pasa\s+con)\b|lo\s+mismo\b|ese\b|esa\b|"
    r"estos\b|estas\b|sobre\s+eso\b))",
    re.IGNORECASE,
)
FOLLOWUP_REFERENCE = re.compile(
    r"\b(?:ese|esa|eso|este|esta|aquello|anterior|mismo|misma|dicho|dicha)\b",
    re.IGNORECASE,
)


def load_corpus(
    paths: list[pathlib.Path],
    expected_model: str = DEFAULT_EMBED_MODEL,
    expected_dimensions: int = 2560,
) -> list[dict[str, Any]]:
    if not paths:
        raise ValueError("No hay documentos indexados disponibles.")
    rows: list[dict[str, Any]] = []
    chunk_ids: set[str] = set()
    for path in paths:
        for row in load_document_embeddings(path):
            if row.get("model") != expected_model:
                raise ValueError(f"Modelo de embedding incompatible en {path.name}.")
            if len(row["embedding"]) != expected_dimensions:
                raise ValueError(f"La dimensión del embedding no coincide en {path.name}.")
            chunk_id = row.get("chunk_id")
            if not chunk_id or chunk_id in chunk_ids:
                raise ValueError(f"Identificador de fragmento duplicado o ausente en {path.name}.")
            chunk_ids.add(chunk_id)
            rows.append(row)
    if not rows:
        raise ValueError("Los índices documentales están vacíos.")
    return rows


def is_followup(question: str) -> bool:
    normalized = question.strip()
    return bool(FOLLOWUP_PREFIX.search(normalized) or FOLLOWUP_REFERENCE.search(normalized))


def resolve_question(
    question: str,
    history: list[dict[str, Any]],
    api_key: str,
    base_url: str = DEFAULT_LLAMUS_BASE_URL,
    chat_client: Callable[..., dict[str, Any]] = ask_llamus,
    force_rewrite: bool = False,
) -> dict[str, str | None]:
    question = question.strip()
    if not history or not (force_rewrite or is_followup(question)):
        return {"question": question, "clarification": None}

    recent_history = [
        {"user": str(turn.get("question", ""))[:400], "assistant": str(turn.get("answer", ""))[:350]}
        for turn in history[-4:]
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "Reescribe la última pregunta del usuario como una pregunta independiente, "
                "usando únicamente el historial para resolver referencias. No respondas a la pregunta "
                "ni inventes hechos. Si hay varios referentes posibles, pide una aclaración. "
                "Devuelve exclusivamente JSON con las claves standalone_question (cadena), "
                "needs_clarification (booleano), clarification_question (cadena)."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({"history": recent_history, "question": question}, ensure_ascii=False),
        },
    ]
    raw_answer = chat_client(
        messages,
        model=DEFAULT_RAG_CHAT_MODEL,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=40,
    )["answer"]
    match = re.search(r"\{.*\}", raw_answer, re.DOTALL)
    if not match:
        raise ValueError("El modelo no devolvió una reformulación estructurada.")
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError as error:
        raise ValueError("El modelo devolvió una reformulación inválida.") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("needs_clarification"), bool):
        raise ValueError("El modelo devolvió una reformulación incompleta.")
    if payload["needs_clarification"]:
        clarification = payload.get("clarification_question")
        if not isinstance(clarification, str) or not clarification.strip():
            raise ValueError("El modelo no precisó qué aclaración necesita.")
        return {"question": question, "clarification": clarification.strip()[:500]}
    standalone = payload.get("standalone_question")
    if not isinstance(standalone, str) or not standalone.strip():
        raise ValueError("El modelo no devolvió una pregunta independiente.")
    return {"question": standalone.strip()[:700], "clarification": None}


def source_name(hit: dict[str, Any]) -> str:
    source_file = hit.get("source_file")
    if isinstance(source_file, str) and source_file.strip():
        return source_file.replace("\\", "/").rsplit("/", 1)[-1]
    return str(hit.get("doc_id") or "Documento sin identificar").replace("_", " ")


def evidence_label(hit: dict[str, Any]) -> str:
    name = source_name(hit)
    sheet = hit.get("sheet")
    row_start = hit.get("row_start")
    row_end = hit.get("row_end")
    if sheet is not None:
        location = f"hoja {sheet}"
        if row_start is not None:
            location += f", fila {row_start}" if row_end in (None, row_start) else f", filas {row_start}–{row_end}"
        return f"{name} · {location}"
    page_start = hit.get("page_start")
    page_end = hit.get("page_end")
    if page_start is not None:
        location = f"página {page_start}" if page_end in (None, page_start) else f"páginas {page_start}–{page_end}"
        return f"{name} · {location}"
    return name
