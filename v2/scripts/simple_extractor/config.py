from __future__ import annotations

import os
import pathlib


DEFAULT_LLAMUS_BASE_URL = "https://llamus.cs.us.es"
DEFAULT_EMBED_MODEL = "qwen3-embedding:4b"
DEFAULT_QA_GENERATION_MODEL = "qwen3:8b"
DEFAULT_GOLD_VALIDATOR_MODEL = "qwen3:8b"
DEFAULT_RAG_CHAT_MODEL = "gemma3:12b"
RAG_SYSTEM_PROMPT = (
    "Eres un asistente RAG. Responde solo con la información del contexto proporcionado. "
    "Si la respuesta no está en el contexto, dilo explícitamente. No inventes cifras ni hechos."
)


def load_local_api_key(project_root: pathlib.Path) -> str | None:
    candidate = project_root / ".llamus_api_key"
    if not candidate.exists():
        return None
    value = candidate.read_text(encoding="utf-8").strip()
    return value or None


def get_api_key(project_root: pathlib.Path) -> str | None:
    return load_local_api_key(project_root) or os.environ.get("LLAMUS_API_KEY")
