from __future__ import annotations

import pathlib
import time
from typing import Any

import requests

from generacion.context_builder import build_context
from generacion.controlled_generation import (
    build_generation_decision,
    generate_controlled_answer,
    restate_answer,
    score_sentence_for_question,
)
from generacion.legal_answers import clean_context_artifacts, extract_legal_answer
from generacion.prompting import build_answer_prompt, build_controlled_answer_prompt
from generacion.question_routing import classify_question_route
from generacion.table_answers import (
    extract_document_presence_answer,
    extract_single_row_table_answer,
    extract_table_cell_answer,
    format_table_value,
    infer_pge_headers,
    looks_like_year_header,
    parse_table_line,
    requested_table_header,
    score_table_row_label,
)
from generacion.text_utils import (
    STANDARD_NO_ANSWER,
    content_tokens,
    lower_first,
    normalize_for_generation,
    question_subject,
)
from recuperacion.retrieval import load_embedding_rows, retrieve_top_k
from ingesta.config import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLAMUS_BASE_URL,
    DEFAULT_RAG_CHAT_MODEL,
)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = PROJECT_ROOT / "RESULTADOS EMBEDDING" / "embeddings"


def list_embedding_files(embeddings_dir: pathlib.Path = EMBEDDINGS_DIR) -> list[pathlib.Path]:
    if not embeddings_dir.exists():
        return []
    return sorted(path for path in embeddings_dir.glob("*.jsonl") if path.is_file())


def load_document_embeddings(embeddings_path: pathlib.Path) -> list[dict[str, Any]]:
    return load_embedding_rows(embeddings_path)


def run_retrieval(
    question: str,
    rows: list[dict[str, Any]],
    top_k: int,
    model: str = DEFAULT_EMBED_MODEL,
    base_url: str = DEFAULT_LLAMUS_BASE_URL,
    api_key: str | None = None,
    strategy: str = "fiscal_hybrid",
    rrf_k: int = 60,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    hits = retrieve_top_k(
        question=question,
        embedding_rows=rows,
        top_k=top_k,
        model=model,
        base_url=base_url,
        api_key=api_key,
        strategy=strategy,
        rrf_k=rrf_k,
    )
    return {"hits": hits, "latency_seconds": time.perf_counter() - started_at}


def ask_llamus(
    prompt_messages: list[dict[str, str]],
    model: str = DEFAULT_RAG_CHAT_MODEL,
    base_url: str = DEFAULT_LLAMUS_BASE_URL,
    api_key: str | None = None,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    started_at = time.perf_counter()
    response = requests.post(
        f"{base_url.rstrip('/')}/api/chat/completions",
        headers=headers,
        json={"model": model, "messages": prompt_messages, "temperature": 0.0},
        timeout=120,
    )
    ended_at = time.perf_counter()

    if response.status_code >= 400:
        detail = (response.text or "")[:500]
        raise requests.HTTPError(
            f"Error al pedir respuesta ({response.status_code}): {detail}",
            response=response,
        )

    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("La respuesta de chat no contiene choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("La respuesta de chat no contiene texto usable.")
    return {"answer": content.strip(), "latency_seconds": ended_at - started_at}

