from __future__ import annotations

import argparse
import json
import pathlib
import re
import time
from typing import Any

from pypdf import PdfReader

from scripts.simple_extractor.chunks import build_parent_child_chunks
from scripts.simple_extractor.config import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLAMUS_BASE_URL,
    get_api_key,
)
from scripts.simple_extractor.embeddings import request_embedding
from scripts.simple_pipeline import write_jsonl


def read_existing_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: pathlib.Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_pdf_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_blocks(pdf_path: pathlib.Path, doc_id: str) -> list[dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    blocks: list[dict[str, Any]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_pdf_text(page.extract_text() or "")
        if not text:
            continue
        blocks.append(
            {
                "block_id": f"{doc_id}::p{page_number}::b1",
                "doc_id": doc_id,
                "page": page_number,
                "source_file": str(pdf_path),
                "text": text,
            }
        )
    return blocks


def embed_chunks_incremental(
    chunks: list[dict[str, Any]],
    embeddings_path: pathlib.Path,
    model: str,
    base_url: str,
    api_key: str,
) -> list[dict[str, Any]]:
    existing_rows = read_existing_jsonl(embeddings_path)
    existing_ids = {row["chunk_id"] for row in existing_rows}
    total = len(chunks)

    for index, chunk in enumerate(chunks, start=1):
        if chunk["chunk_id"] in existing_ids:
            print(f"[{index}/{total}] skip {chunk['chunk_id']}", flush=True)
            continue
        print(f"[{index}/{total}] embedding {chunk['chunk_id']}", flush=True)
        row = dict(chunk)
        row["model"] = model
        row["embedding"] = request_embedding(
            chunk["retrieval_text"],
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        append_jsonl(embeddings_path, row)
        existing_rows.append(row)
        existing_ids.add(chunk["chunk_id"])

    return existing_rows


def process_pdf(
    pdf_path: pathlib.Path,
    output_root: pathlib.Path,
    max_child_tokens: int,
    overlap_tokens: int,
    embed: bool,
    model: str,
    base_url: str,
    api_key: str | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    doc_id = pdf_path.stem
    blocks = extract_blocks(pdf_path, doc_id)
    chunks = build_parent_child_chunks(
        blocks,
        max_child_tokens=max_child_tokens,
        overlap_tokens=overlap_tokens,
    )

    blocks_path = output_root / "blocks" / f"{doc_id}.jsonl"
    chunks_path = output_root / "chunks" / f"{doc_id}.jsonl"
    embeddings_path = output_root / "embeddings" / f"{doc_id}.jsonl"
    write_jsonl(blocks_path, blocks)
    write_jsonl(chunks_path, chunks)

    result: dict[str, Any] = {
        "doc_id": doc_id,
        "source_file": str(pdf_path),
        "extractor": "pypdf_direct_text",
        "page_count": len(PdfReader(str(pdf_path)).pages),
        "block_count": len(blocks),
        "chunk_count": len(chunks),
        "chunking_strategy": "parent_child_tokens",
        "max_child_tokens": max_child_tokens,
        "overlap_tokens": overlap_tokens,
        "blocks_path": str(blocks_path),
        "chunks_path": str(chunks_path),
    }

    if embed:
        if not api_key:
            raise ValueError("No hay API key para embeddings. Usa TFG/.llamus_api_key o LLAMUS_API_KEY.")
        embedded = embed_chunks_incremental(
            chunks=chunks,
            embeddings_path=embeddings_path,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        result["embeddings_path"] = str(embeddings_path)
        result["embedding_model"] = model
        result["embedding_count"] = len(embedded)
        result["embedding_dimension"] = len(embedded[0]["embedding"]) if embedded else 0

    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Procesa PDFs por extracción directa y genera JSONL/embeddings.")
    parser.add_argument("paths", nargs="+", type=pathlib.Path)
    parser.add_argument("--output-root", type=pathlib.Path, default=pathlib.Path("RESULTADOS EMBEDDING"))
    parser.add_argument("--max-child-tokens", type=int, default=220)
    parser.add_argument("--overlap-tokens", type=int, default=40)
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_LLAMUS_BASE_URL)
    args = parser.parse_args()

    project_root = pathlib.Path(__file__).resolve().parents[2]
    api_key = get_api_key(project_root)
    summaries = []
    for path in args.paths:
        summaries.append(
            process_pdf(
                pdf_path=path,
                output_root=args.output_root,
                max_child_tokens=args.max_child_tokens,
                overlap_tokens=args.overlap_tokens,
                embed=args.embed,
                model=args.model,
                base_url=args.base_url,
                api_key=api_key,
            )
        )

    summary_path = args.output_root / "ingesta_eval" / "pdf_direct_embeddings_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "documents": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
