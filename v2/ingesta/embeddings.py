from __future__ import annotations

from typing import Any, Callable

import requests

from ingesta.config import DEFAULT_EMBED_MODEL, DEFAULT_LLAMUS_BASE_URL


def request_embedding(
    text: str,
    model: str = DEFAULT_EMBED_MODEL,
    base_url: str = DEFAULT_LLAMUS_BASE_URL,
    api_key: str | None = None,
    timeout: float = 90,
) -> list[float]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(
        f"{base_url.rstrip('/')}/ollama/api/embed",
        headers=headers,
        json={"model": model, "input": text},
        timeout=timeout,
    )
    if response.status_code >= 400:
        detail = (response.text or "")[:500]
        raise requests.HTTPError(
            f"Error al pedir embedding ({response.status_code}): {detail}",
            response=response,
        )
    payload = response.json()

    if isinstance(payload.get("embedding"), list):
        return payload["embedding"]
    if isinstance(payload.get("embeddings"), list) and payload["embeddings"]:
        return payload["embeddings"][0]
    raise ValueError("La respuesta de embeddings no contiene un vector usable.")


def embed_chunks(
    chunks: list[dict[str, Any]],
    model: str = DEFAULT_EMBED_MODEL,
    base_url: str = DEFAULT_LLAMUS_BASE_URL,
    api_key: str | None = None,
    requestor: Callable[[str, str, str, str | None], list[float]] | None = None,
) -> list[dict[str, Any]]:
    requester = requestor or request_embedding
    embedded_rows: list[dict[str, Any]] = []
    for chunk in chunks:
        row = {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "model": model,
            "text": chunk["text"],
            "embedding": requester(chunk["text"], model, base_url, api_key),
        }
        for field in (
            "parent_id",
            "parent_text",
            "retrieval_text",
            "chunk_index",
            "chunk_count",
            "chunking_strategy",
            "max_child_tokens",
            "overlap_tokens",
        ):
            if field in chunk:
                row[field] = chunk[field]
        embedded_rows.append(row)
    return embedded_rows

