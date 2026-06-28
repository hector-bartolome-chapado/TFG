from __future__ import annotations

import pathlib
import time
from typing import Any

import requests

from scripts.RECUPERADOR.retrieval import load_embedding_rows, retrieve_top_k
from scripts.simple_extractor.config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLAMUS_BASE_URL,
    RAG_SYSTEM_PROMPT,
)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = PROJECT_ROOT / "RESULTADOS EMBEDDING" / "embeddings"


# Descubre los ficheros de embeddings disponibles para poblar el selector de documento.
#
# Entra:
# - embeddings_dir: carpeta donde viven los `.jsonl` del proyecto.
# Sale:
# - lista ordenada de rutas a embeddings disponibles.
# Por qué existe:
# - la interfaz debe poder cambiar de documento sin tocar el código ni escribir rutas a mano.
def list_embedding_files(embeddings_dir: pathlib.Path = EMBEDDINGS_DIR) -> list[pathlib.Path]:
    if not embeddings_dir.exists():
        return []
    return sorted(path for path in embeddings_dir.glob("*.jsonl") if path.is_file())


# Carga en memoria los embeddings de un documento concreto.
#
# Entra:
# - embeddings_path: ruta al `.jsonl` del documento.
# Sale:
# - lista de filas con chunk, texto y embedding.
# Por qué existe:
# - separa la selección de documento de la carga real de datos.
def load_document_embeddings(embeddings_path: pathlib.Path) -> list[dict[str, Any]]:
    return load_embedding_rows(embeddings_path)


# Ejecuta retrieval y devuelve tanto los hits como el tiempo empleado.
#
# Entra:
# - question: consulta del usuario.
# - rows: embeddings del documento cargado.
# - top_k: número de chunks a recuperar.
# - model/base_url/api_key: backend de embeddings para la pregunta.
# Sale:
# - un diccionario con hits y latencia.
# Por qué existe:
# - la interfaz necesita enseñar retrieval y timing sin repetir lógica.
def run_retrieval(
    question: str,
    rows: list[dict[str, Any]],
    top_k: int,
    model: str = DEFAULT_EMBED_MODEL,
    base_url: str = DEFAULT_LLAMUS_BASE_URL,
    api_key: str | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    hits = retrieve_top_k(
        question=question,
        embedding_rows=rows,
        top_k=top_k,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )
    return {"hits": hits, "latency_seconds": time.perf_counter() - started_at}


# Construye el contexto exacto que se enviará al modelo a partir de los hits recuperados.
#
# Entra:
# - hits: chunks recuperados por retrieval.
# Sale:
# - un string con cada chunk identificado y separado.
# Por qué existe:
# - así puedes ver en la interfaz qué contexto recibe el modelo y depurar alucinaciones.
def build_context(hits: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for index, hit in enumerate(hits, start=1):
        parts.append(f"[Chunk {index} | {hit['chunk_id']} | score={hit['score']:.4f}]\n{hit['text']}")
    return "\n\n".join(parts)


# Construye el prompt de respuesta anclado al contexto recuperado.
#
# Entra:
# - question: pregunta del usuario.
# - context: texto consolidado de retrieval.
# Sale:
# - una lista de mensajes para `chat/completions`.
# Por qué existe:
# - extractor e interfaz deben compartir una sola política de respuesta.
def build_answer_prompt(question: str, context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"Pregunta:\n{question}\n\nContexto recuperado:\n{context}"},
    ]


# Llama al endpoint de chat de `llamus` y devuelve la respuesta del modelo.
#
# Entra:
# - prompt_messages: mensajes listos para `chat/completions`.
# - model/base_url/api_key: backend de generación.
# Sale:
# - un diccionario con la respuesta y la latencia.
# Por qué existe:
# - la interfaz debe cerrar el loop RAG y mostrar la respuesta final sin duplicar prompts.
def ask_llamus(
    prompt_messages: list[dict[str, str]],
    model: str = DEFAULT_CHAT_MODEL,
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
