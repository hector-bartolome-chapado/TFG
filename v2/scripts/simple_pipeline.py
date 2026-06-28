from __future__ import annotations

import json
import pathlib
from typing import Any

from scripts.simple_extractor.blocks import build_blocks
from scripts.simple_extractor.chunks import build_chunks
from scripts.simple_extractor.config import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLAMUS_BASE_URL,
    get_api_key,
)
from scripts.simple_extractor.embeddings import embed_chunks
from scripts.simple_extractor.extract import extract_pages


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_document(
    pdf_path: pathlib.Path,
    output_root: pathlib.Path,
    max_chars: int = 400,
    max_child_tokens: int | None = 220,
    overlap_tokens: int = 40,
    embed: bool = False,
    model: str = DEFAULT_EMBED_MODEL,
    base_url: str = DEFAULT_LLAMUS_BASE_URL,
    api_key: str | None = None,
) -> dict[str, Any]:
    project_root = pathlib.Path(__file__).resolve().parents[1]
    doc_id = pdf_path.stem
    pages = extract_pages(pdf_path)
    blocks = build_blocks(doc_id, pages)
    chunks = build_chunks(
        blocks,
        max_chars=max_chars if max_child_tokens is None else None,
        max_child_tokens=max_child_tokens,
        overlap_tokens=overlap_tokens,
    )

    blocks_path = output_root / "blocks" / f"{doc_id}.jsonl"
    chunks_path = output_root / "chunks" / f"{doc_id}.jsonl"
    write_jsonl(blocks_path, blocks)
    write_jsonl(chunks_path, chunks)

    result: dict[str, Any] = {
        "doc_id": doc_id,
        "page_count": len(pages),
        "block_count": len(blocks),
        "chunk_count": len(chunks),
        "chunking_strategy": "legacy_chars" if max_child_tokens is None else "parent_child_tokens",
        "blocks_path": str(blocks_path),
        "chunks_path": str(chunks_path),
    }

    if not embed:
        return result

    effective_api_key = api_key or get_api_key(project_root)
    if not effective_api_key:
        raise ValueError("No hay API key para embeddings. Usa TFG/.llamus_api_key o LLAMUS_API_KEY.")

    embeddings_path = output_root / "embeddings" / f"{doc_id}.jsonl"
    embedded_chunks = embed_chunks(
        chunks=chunks,
        model=model,
        base_url=base_url,
        api_key=effective_api_key,
    )
    write_jsonl(embeddings_path, embedded_chunks)
    result["embeddings_path"] = str(embeddings_path)
    return result
