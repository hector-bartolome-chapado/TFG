from __future__ import annotations

from typing import Any


def build_context(hits: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    seen_context_ids: set[str] = set()
    for index, hit in enumerate(hits, start=1):
        context_id = hit.get("parent_id") or hit["chunk_id"]
        if context_id in seen_context_ids:
            continue
        seen_context_ids.add(context_id)

        context_text = hit.get("context_text") or hit.get("parent_text") or hit["text"]
        label = f"Chunk {index} | {hit['chunk_id']}"
        if hit.get("parent_id"):
            label += f" | parent={hit['parent_id']}"
        parts.append(f"[{label} | score={hit['score']:.4f}]\n{context_text}")
    return "\n\n".join(parts)

