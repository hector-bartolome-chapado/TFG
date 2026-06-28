from __future__ import annotations

import re
from typing import Any


# Parte un texto largo en trozos más pequeños intentando respetar frases completas.
#
# Entra:
# - text: bloque de texto original.
# - max_chars: tamaño máximo de cada trozo.
# Sale:
# - una lista de fragmentos con longitud razonable.
# Por qué existe:
# - en el v1 un párrafo puede ser más largo que el límite de chunking y, si no lo partimos, el embedding puede fallar.
def split_long_text(text: str, max_chars: int) -> list[str]:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return [stripped]

    sentences = re.split(r"(?<=[.!?])\s+", stripped)
    parts: list[str] = []
    pending = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = sentence if not pending else pending + " " + sentence
        if len(candidate) <= max_chars:
            pending = candidate
            continue

        if pending:
            parts.append(pending)
            pending = ""

        if len(sentence) <= max_chars:
            pending = sentence
            continue

        start = 0
        while start < len(sentence):
            parts.append(sentence[start : start + max_chars].strip())
            start += max_chars

    if pending:
        parts.append(pending)
    return [part for part in parts if part]


# Agrupa bloques consecutivos en chunks simples hasta un tamaño máximo.
#
# Entra:
# - blocks: bloques ya creados a partir de párrafos.
# - max_chars: número máximo de caracteres por chunk.
# Sale:
# - chunks con rango de páginas y texto concatenado.
# Por qué existe:
# - es la primera versión del chunking: juntar párrafos cercanos sin reglas especiales.
def build_chunks(blocks: list[dict[str, Any]], max_chars: int = 400) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None

    def flush_pending() -> None:
        nonlocal pending
        if pending is not None:
            chunks.append(pending)
            pending = None

    for block in blocks:
        text_parts = split_long_text(block["text"], max_chars)
        for part_index, text_part in enumerate(text_parts, start=1):
            chunk_id = f"{block['block_id']}::chunk" if len(text_parts) == 1 else f"{block['block_id']}::chunk::{part_index}"
            if pending is None:
                pending = {
                    "chunk_id": chunk_id,
                    "doc_id": block["doc_id"],
                    "page_start": block["page"],
                    "page_end": block["page"],
                    "text": text_part,
                }
                continue

            candidate_text = pending["text"] + "\n\n" + text_part
            if len(candidate_text) <= max_chars:
                pending["text"] = candidate_text
                pending["page_end"] = block["page"]
            else:
                flush_pending()
                pending = {
                    "chunk_id": chunk_id,
                    "doc_id": block["doc_id"],
                    "page_start": block["page"],
                    "page_end": block["page"],
                    "text": text_part,
                }

    flush_pending()
    return chunks
