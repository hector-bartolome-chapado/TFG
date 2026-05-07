from __future__ import annotations

import re
from typing import Any


def split_into_paragraphs(page_text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", page_text) if part.strip()]


# Convierte páginas en bloques mínimos que luego se podrán agrupar en chunks.
#
# Entra:
# - doc_id: id del documento.
# - pages: páginas con texto limpio.
# Sale:
# - una lista de bloques con id, página y texto.
# Por qué existe:
# - separar "extraer texto" de "crear bloques" te deja ver dos fases distintas del pipeline.
def build_blocks(doc_id: str, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for page in pages:
        page_number = int(page["page"])
        for block_index, paragraph in enumerate(split_into_paragraphs(page["text"]), start=1):
            blocks.append(
                {
                    "block_id": f"{doc_id}::p{page_number}::b{block_index}",
                    "doc_id": doc_id,
                    "page": page_number,
                    "text": paragraph,
                }
            )
    return blocks

