from __future__ import annotations

import re
from typing import Any


DEFAULT_CHILD_TOKENS = 220
DEFAULT_OVERLAP_TOKENS = 40


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


def split_text_by_tokens(text: str, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
    tokens = text.strip().split()
    if not tokens:
        return []
    if max_tokens <= 0:
        raise ValueError("max_tokens debe ser mayor que cero.")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens no puede ser negativo.")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens debe ser menor que max_tokens.")
    if len(tokens) <= max_tokens:
        return [" ".join(tokens)]

    parts: list[str] = []
    step = max_tokens - overlap_tokens
    start = 0
    while start < len(tokens):
        part_tokens = tokens[start : start + max_tokens]
        if part_tokens:
            parts.append(" ".join(part_tokens))
        if start + max_tokens >= len(tokens):
            break
        start += step
    return parts


def build_legacy_chunks(blocks: list[dict[str, Any]], max_chars: int = 400) -> list[dict[str, Any]]:
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


def build_parent_child_chunks(
    blocks: list[dict[str, Any]],
    max_child_tokens: int = DEFAULT_CHILD_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for block in blocks:
        parent_text = block["text"].strip()
        if not parent_text:
            continue

        parent_id = f"{block['block_id']}::parent"
        child_texts = split_text_by_tokens(
            parent_text,
            max_tokens=max_child_tokens,
            overlap_tokens=overlap_tokens,
        )

        for child_index, child_text in enumerate(child_texts, start=1):
            chunks.append(
                {
                    "chunk_id": f"{parent_id}::child::{child_index}",
                    "parent_id": parent_id,
                    "doc_id": block["doc_id"],
                    "page_start": block["page"],
                    "page_end": block["page"],
                    "text": child_text,
                    "retrieval_text": child_text,
                    "parent_text": parent_text,
                    "chunk_index": child_index,
                    "chunk_count": len(child_texts),
                    "chunking_strategy": "parent_child_tokens",
                    "max_child_tokens": max_child_tokens,
                    "overlap_tokens": overlap_tokens,
                }
            )
    return chunks


def build_chunks(
    blocks: list[dict[str, Any]],
    max_chars: int | None = None,
    max_child_tokens: int | None = None,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[dict[str, Any]]:
    if max_child_tokens is not None:
        return build_parent_child_chunks(
            blocks,
            max_child_tokens=max_child_tokens,
            overlap_tokens=overlap_tokens,
        )
    if max_chars is not None:
        return build_legacy_chunks(blocks, max_chars=max_chars)
    return build_parent_child_chunks(blocks)

