from __future__ import annotations

import os
import pathlib


DEFAULT_LLAMUS_BASE_URL = "https://llamus.cs.us.es"
DEFAULT_EMBED_MODEL = "mxbai-embed-large:v1"
DEFAULT_CHAT_MODEL = "llama3.1:8b"
RAG_SYSTEM_PROMPT = (
    "Eres un asistente RAG. Responde solo con la informaciÃ³n del contexto proporcionado. "
    "Si la respuesta no estÃ¡ en el contexto, dilo explÃ­citamente. No inventes cifras ni hechos."
)


def load_local_api_key(project_root: pathlib.Path) -> str | None:
    candidate = project_root / ".llamus_api_key"
    if not candidate.exists():
        return None
    value = candidate.read_text(encoding="utf-8").strip()
    return value or None


def get_api_key(project_root: pathlib.Path) -> str | None:
    return load_local_api_key(project_root) or os.environ.get("LLAMUS_API_KEY")
