from __future__ import annotations

import os
import pathlib


DEFAULT_LLAMUS_BASE_URL = "https://llamus.cs.us.es"
DEFAULT_EMBED_MODEL = "mxbai-embed-large:v1"
DEFAULT_CHAT_MODEL = "llama3.1:8b"
RAG_SYSTEM_PROMPT = (
    "Eres un asistente RAG. Responde solo con la información del contexto proporcionado. "
    "Si la respuesta no está en el contexto, dilo explícitamente. No inventes cifras ni hechos."
)


# Lee la API key desde el fichero local del proyecto si existe.
#
# Entra:
# - project_root: raíz del proyecto TFG.
# Sale:
# - la clave leída o `None` si no hay fichero.
# Por qué existe:
# - extractor, retrieval e interfaz deben resolver la autenticación de la misma forma.
def load_local_api_key(project_root: pathlib.Path) -> str | None:
    candidate = project_root / ".llamus_api_key"
    if not candidate.exists():
        return None
    value = candidate.read_text(encoding="utf-8").strip()
    return value or None


# Resuelve la API key combinando fichero local y variable de entorno.
#
# Entra:
# - project_root: raíz del proyecto TFG.
# Sale:
# - la clave o `None`.
# Por qué existe:
# - centraliza la política de autenticación en un único sitio.
def get_api_key(project_root: pathlib.Path) -> str | None:
    return load_local_api_key(project_root) or os.environ.get("LLAMUS_API_KEY")
