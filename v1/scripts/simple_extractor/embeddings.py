from __future__ import annotations

from typing import Any, Callable

import requests

from scripts.simple_extractor.config import DEFAULT_EMBED_MODEL, DEFAULT_LLAMUS_BASE_URL


# Pide un embedding a llamus para un único texto.
#
# Entra:
# - text: chunk que queremos vectorizar.
# - model: nombre del modelo de embeddings.
# - base_url: servidor llamus.
# - api_key: token del servidor.
# Sale:
# - una lista de números de coma flotante.
# Por qué existe:
# - esta es la integración mínima con el servidor real del TFG.
def request_embedding(
    text: str,
    model: str = DEFAULT_EMBED_MODEL,
    base_url: str = DEFAULT_LLAMUS_BASE_URL,
    api_key: str | None = None,
) -> list[float]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(
        f"{base_url.rstrip('/')}/ollama/api/embed",
        headers=headers,
        json={"model": model, "input": text},
        timeout=90,
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


# Recorre todos los chunks y les añade su embedding.
#
# Entra:
# - chunks: lista de chunks ya construidos.
# - model, base_url, api_key: parámetros del backend.
# - requestor: función opcional para tests o mocks.
# Sale:
# - una lista de registros con chunk y embedding.
# Por qué existe:
# - separa claramente la fase de chunking de la fase de vectorización.
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
        embedded_rows.append(
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "model": model,
                "text": chunk["text"],
                "embedding": requester(chunk["text"], model, base_url, api_key),
            }
        )
    return embedded_rows
