from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import uuid
from typing import Any, Callable

import pandas as pd
import requests

from scripts.clean_aeat_pdf import DEFAULT_SKIP_PATTERNS, clean_page_text, detect_page_metadata

try:
    from docling.document_converter import DocumentConverter  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    DocumentConverter = None

import fitz


DEFAULT_LLAMUS_BASE_URL = "https://llamus.cs.us.es"
DEFAULT_LLAMUS_MODEL = "llama3.1:8b"
DEFAULT_EMBED_MODEL = "mxbai-embed-large:v1"
DEFAULT_MIN_ENRICH_CHARS = 40
LLM_METADATA_KEYS = (
    "summary",
    "answerable_questions",
    "keywords",
    "entities",
    "taxes_mentioned",
    "years_mentioned",
    "amounts_mentioned",
    "organization",
    "is_numeric_evidence",
    "is_normative_evidence",
    "relationships",
    "assumptions_or_conditions",
    "confidence_note",
    "analytic_tags",
)


def slugify(value: str) -> str:
    value = re.sub(r"\s+", "_", value.strip())
    value = re.sub(r"[^0-9A-Za-zÁÉÍÓÚáéíóúÑñ_:-]", "", value)
    return value or "block"


def numeric_density(text: str) -> float:
    if not text:
        return 0.0
    digit_count = sum(char.isdigit() for char in text)
    return digit_count / max(len(text), 1)


def is_table_like_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    dotted_leader_count = sum(1 for line in lines if re.search(r"\.{4,}", line))
    numeric_short_lines = sum(1 for line in lines if re.fullmatch(r"\d{1,3}([.,]\d+)?", line))
    tabular_rows = sum(1 for line in lines if len(re.split(r"\s{2,}|\t", line)) >= 2)
    mixed_fact_rows = sum(
        1
        for line in lines
        if re.search(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]", line)
        and len(re.findall(r"\d{2,4}", line)) >= 1
        and re.search(r"\d[^\d]*$", line)
    )

    return (
        dotted_leader_count >= 2
        or numeric_short_lines >= 3
        or mixed_fact_rows >= 3
        or (tabular_rows >= 2 and numeric_density(text) >= 0.08)
    )


def block_from_text(
    doc_id: str,
    source_family: str,
    page: int,
    section_path: list[str],
    text: str,
    block_index: int,
    page_kind: str = "content",
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = text.strip()
    num_density = numeric_density(text)
    table_like = is_table_like_text(text)
    normalized_upper = text.upper()

    if page_kind == "index":
        block_type = "index"
    elif normalized_upper.startswith("NOTA INFORMATIVA"):
        block_type = "note"
    elif table_like:
        block_type = "table"
    elif not table_like and len(text) <= 120 and text.count("\n") <= 2 and not re.search(r"[.:;]\s", text):
        block_type = "title"
    else:
        block_type = "content"

    base_metadata = {
        "table_like": table_like,
        "page_kind": page_kind,
    }
    if extra_metadata:
        base_metadata.update(extra_metadata)

    return {
        "block_id": f"{doc_id}::p{page}::b{block_index}",
        "doc_id": doc_id,
        "source_family": source_family,
        "page_start": page,
        "page_end": page,
        "section_path": section_path,
        "block_type": block_type,
        "text": text,
        "table_data": None,
        "numeric_density": num_density,
        "base_metadata": base_metadata,
    }


def split_page_into_blocks(page_text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\n\s*\n", page_text) if part.strip()]
    return parts


def parse_pdf_pages(
    pdf_path: pathlib.Path,
    skip_patterns: list[str] | None = None,
    prefer_docling: bool = False,
) -> list[dict[str, Any]]:
    if prefer_docling and DocumentConverter is not None:
        try:
            converter = DocumentConverter()
            result = converter.convert(str(pdf_path))
            markdown = result.document.export_to_markdown()
            return [{"page": 1, "text": clean_page_text(markdown, skip_patterns), "from_docling": True}]
        except Exception:
            pass

    pages: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document, start=1):
            cleaned = clean_page_text(page.get_text("text"), skip_patterns=skip_patterns or [])
            if cleaned:
                pages.append({"page": page_index, "text": cleaned, "from_docling": False})
    return pages


def extract_pdf_blocks(
    pdf_path: pathlib.Path,
    doc_id: str,
    source_family: str,
    skip_patterns: list[str] | None = None,
    prefer_docling: bool = False,
) -> list[dict[str, Any]]:
    pages = parse_pdf_pages(pdf_path, skip_patterns=skip_patterns, prefer_docling=prefer_docling)
    blocks: list[dict[str, Any]] = []

    for page_info in pages:
        text = page_info["text"]
        page = page_info["page"]
        metadata = detect_page_metadata(text)
        section_root = metadata.get("section_title") or f"page_{page}"
        page_blocks = split_page_into_blocks(text)
        for block_index, part in enumerate(page_blocks):
            effective_section = section_root
            if block_index == 0 and metadata.get("section_title"):
                effective_section = metadata["section_title"]
            block = block_from_text(
                doc_id=doc_id,
                source_family=source_family,
                page=page,
                section_path=[effective_section],
                text=part,
                block_index=block_index,
                page_kind=metadata.get("page_kind", "content"),
                extra_metadata={"source_parser": "docling" if page_info["from_docling"] else "pymupdf"},
            )
            blocks.append(block)

    return blocks


def dataframe_to_text(df: pd.DataFrame) -> str:
    header = "\t".join(str(col) for col in df.columns)
    lines = [header]
    for row in df.itertuples(index=False):
        lines.append("\t".join("" if pd.isna(value) else str(value) for value in row))
    return "\n".join(lines)


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = df.fillna("")
    return df


def extract_xlsx_blocks(
    xlsx_paths: list[pathlib.Path],
    doc_id: str,
    source_family: str,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for xlsx_path in xlsx_paths:
        with pd.ExcelFile(xlsx_path) as workbook:
            for sheet_name in workbook.sheet_names:
                df = sanitize_dataframe(workbook.parse(sheet_name=sheet_name))
                if df.empty:
                    continue

                records = json.loads(df.to_json(orient="records", force_ascii=False))
                text = dataframe_to_text(df)
                block_id = f"{doc_id}::xlsx::{slugify(xlsx_path.stem)}::{slugify(sheet_name)}"
                blocks.append(
                    {
                        "block_id": block_id,
                        "doc_id": doc_id,
                        "source_family": source_family,
                        "page_start": None,
                        "page_end": None,
                        "section_path": [sheet_name],
                        "block_type": "table",
                        "text": text,
                        "table_data": records,
                        "sheet_name": sheet_name,
                        "numeric_density": numeric_density(text),
                        "base_metadata": {
                            "table_like": True,
                            "source_parser": "xlsx",
                            "source_file": xlsx_path.name,
                        },
                    }
                )
    return blocks


def build_document_payload(doc_id: str, source_pdf: pathlib.Path, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    page_numbers = [block["page_start"] for block in blocks if isinstance(block.get("page_start"), int)]
    return {
        "doc_id": doc_id,
        "source_pdf": str(source_pdf),
        "block_count": len(blocks),
        "page_count": len(set(page_numbers)),
        "blocks": blocks,
    }


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_markdown_from_blocks(doc_id: str, blocks: list[dict[str, Any]]) -> str:
    lines = [f"# {doc_id}", ""]
    for block in blocks:
        section = " / ".join(block["section_path"]) if block.get("section_path") else "sin_sección"
        heading = f"## {section}"
        if block.get("page_start"):
            heading = f"{heading} (página {block['page_start']})"
        lines.append(heading)
        lines.append("")
        lines.append(f"- Tipo: {block['block_type']}")
        lines.append("")
        lines.append(block["text"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_chunks(blocks: list[dict[str, Any]], max_chars: int = 1800) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None

    def flush_pending() -> None:
        nonlocal pending
        if pending is not None:
            chunks.append(pending)
            pending = None

    for block in blocks:
        if block["block_type"] in {"table", "index", "title"}:
            flush_pending()
            chunks.append(
                {
                    "chunk_id": f"{block['block_id']}::chunk",
                    "doc_id": block["doc_id"],
                    "source_family": block["source_family"],
                    "section_path": block["section_path"],
                    "block_type": block["block_type"],
                    "text": block["text"],
                    "table_data": block.get("table_data"),
                    "page_start": block["page_start"],
                    "page_end": block["page_end"],
                    "base_metadata": dict(block.get("base_metadata", {})),
                    "numeric_density": block.get("numeric_density", 0.0),
                }
            )
            continue

        if pending is None:
            pending = {
                "chunk_id": f"{block['block_id']}::chunk",
                "doc_id": block["doc_id"],
                "source_family": block["source_family"],
                "section_path": block["section_path"],
                "block_type": block["block_type"],
                "text": block["text"],
                "table_data": None,
                "page_start": block["page_start"],
                "page_end": block["page_end"],
                "base_metadata": dict(block.get("base_metadata", {})),
                "numeric_density": block.get("numeric_density", 0.0),
            }
            continue

        same_section = pending["section_path"] == block["section_path"]
        same_type = pending["block_type"] == block["block_type"]
        candidate_text = pending["text"] + "\n\n" + block["text"]
        if same_section and same_type and len(candidate_text) <= max_chars:
            pending["text"] = candidate_text
            pending["page_end"] = block["page_end"]
            pending["numeric_density"] = max(pending["numeric_density"], block.get("numeric_density", 0.0))
        else:
            flush_pending()
            pending = {
                "chunk_id": f"{block['block_id']}::chunk",
                "doc_id": block["doc_id"],
                "source_family": block["source_family"],
                "section_path": block["section_path"],
                "block_type": block["block_type"],
                "text": block["text"],
                "table_data": None,
                "page_start": block["page_start"],
                "page_end": block["page_end"],
                "base_metadata": dict(block.get("base_metadata", {})),
                "numeric_density": block.get("numeric_density", 0.0),
            }

    flush_pending()
    return chunks


def has_financial_signal(text: str) -> bool:
    upper_text = text.upper()
    keyword_count = sum(
        1
        for keyword in (
            "IRPF",
            "IVA",
            "SOCIEDADES",
            "RECAUD",
            "PRESUPUEST",
            "INGRESOS",
            "GASTOS",
            "IMPUEST",
            "DEFICIT",
            "DEUDA",
            "EJECUCI",
        )
        if keyword in upper_text
    )
    has_year = bool(re.search(r"\b(19\d{2}|20\d{2})\b", text))
    has_amount = bool(re.search(r"\b\d[\d.,]*\s*(?:€|euros|%|millones|miles de millones)?\b", text, re.IGNORECASE))
    return keyword_count >= 2 or (keyword_count >= 1 and (has_year or has_amount))


def should_enrich_chunk(chunk: dict[str, Any], min_chars: int = DEFAULT_MIN_ENRICH_CHARS) -> bool:
    text = (chunk.get("text") or "").strip()
    if not text:
        return False
    if chunk.get("block_type") in {"index", "title"}:
        return False
    if chunk.get("block_type") in {"table", "note"}:
        return True
    if len(text) < min_chars:
        return False
    if chunk.get("source_family") == "BOE":
        return True
    if chunk.get("numeric_density", 0.0) >= 0.08:
        return True
    if chunk.get("block_type") == "content" and has_financial_signal(text):
        return True
    return False


def build_llamus_prompt(chunk: dict[str, Any]) -> str:
    section = " / ".join(chunk.get("section_path") or [])
    return (
        "Analiza el siguiente chunk documental del corpus fiscal-financiero español. "
        "Devuelve exclusivamente un objeto JSON válido con estas claves exactas: "
        "summary, answerable_questions, keywords, entities, taxes_mentioned, years_mentioned, "
        "amounts_mentioned, organization, is_numeric_evidence, is_normative_evidence, relationships, "
        "assumptions_or_conditions, confidence_note, analytic_tags.\n\n"
        f"doc_id: {chunk['doc_id']}\n"
        f"source_family: {chunk['source_family']}\n"
        f"section_path: {section}\n"
        f"block_type: {chunk['block_type']}\n"
        f"text:\n{chunk['text']}"
    )


def build_llamus_repair_prompt(chunk: dict[str, Any], previous_response: str) -> str:
    section = " / ".join(chunk.get("section_path") or [])
    return (
        "Reformula la salida anterior y devuelve SOLO un objeto JSON válido, sin explicación, sin markdown y sin texto extra. "
        "Las claves exactas deben ser: "
        "summary, answerable_questions, keywords, entities, taxes_mentioned, years_mentioned, "
        "amounts_mentioned, organization, is_numeric_evidence, is_normative_evidence, relationships, "
        "assumptions_or_conditions, confidence_note, analytic_tags.\n\n"
        f"doc_id: {chunk['doc_id']}\n"
        f"source_family: {chunk['source_family']}\n"
        f"section_path: {section}\n"
        f"block_type: {chunk['block_type']}\n"
        f"text:\n{chunk['text']}\n\n"
        f"salida_anterior:\n{previous_response}"
    )


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1)
    else:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            stripped = stripped[start : end + 1]
    return json.loads(stripped)


def normalize_llm_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = {key: payload.get(key) for key in LLM_METADATA_KEYS}
    normalized["summary"] = normalized.get("summary") or ""
    normalized["answerable_questions"] = normalized.get("answerable_questions") or []
    normalized["keywords"] = normalized.get("keywords") or []
    normalized["entities"] = normalized.get("entities") or []
    normalized["taxes_mentioned"] = normalized.get("taxes_mentioned") or []
    normalized["years_mentioned"] = normalized.get("years_mentioned") or []
    normalized["amounts_mentioned"] = normalized.get("amounts_mentioned") or []
    normalized["organization"] = normalized.get("organization") or ""
    normalized["is_numeric_evidence"] = bool(normalized.get("is_numeric_evidence"))
    normalized["is_normative_evidence"] = bool(normalized.get("is_normative_evidence"))
    normalized["relationships"] = normalized.get("relationships") or []
    normalized["assumptions_or_conditions"] = normalized.get("assumptions_or_conditions") or []
    normalized["confidence_note"] = normalized.get("confidence_note") or ""
    normalized["analytic_tags"] = normalized.get("analytic_tags") or []
    return normalized


def build_fallback_metadata(chunk: dict[str, Any], reason: str, raw_response: str = "") -> dict[str, Any]:
    tags = [chunk.get("source_family", "UNKNOWN"), chunk.get("block_type", "unknown")]
    text = chunk.get("text", "")
    taxes = sorted(set(re.findall(r"\b(IRPF|IVA|Sociedades|Impuesto sobre Sociedades|Impuestos Especiales)\b", text, re.IGNORECASE)))
    years = [int(year) for year in re.findall(r"\b(19\d{2}|20\d{2})\b", text)]
    amounts = re.findall(r"\b\d[\d.,]*\s*(?:€|euros|%|millones|miles de millones)?\b", text, re.IGNORECASE)
    summary = text[:280].strip()
    if len(text) > 280:
        summary += "..."
    metadata = {
        "summary": summary,
        "answerable_questions": [],
        "keywords": [],
        "entities": [],
        "taxes_mentioned": taxes,
        "years_mentioned": years[:10],
        "amounts_mentioned": amounts[:10],
        "organization": chunk.get("source_family", ""),
        "is_numeric_evidence": bool(chunk.get("numeric_density", 0.0) >= 0.08 or chunk.get("block_type") == "table"),
        "is_normative_evidence": chunk.get("source_family") == "BOE",
        "relationships": [],
        "assumptions_or_conditions": [],
        "confidence_note": f"metadata de respaldo generada localmente: {reason}",
        "analytic_tags": tags,
    }
    if raw_response:
        metadata["raw_response_excerpt"] = raw_response[:500]
    return metadata


def post_llamus_chat(prompt: str, model: str, base_url: str, api_key: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/api/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=90,
    )
    response.raise_for_status()
    body = response.json()
    return body["choices"][0]["message"]["content"]


def request_llamus_metadata(
    chunk: dict[str, Any],
    model: str,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    content = post_llamus_chat(build_llamus_prompt(chunk), model, base_url, api_key)
    try:
        return normalize_llm_metadata(extract_json_object(content))
    except (json.JSONDecodeError, TypeError, ValueError) as first_error:
        repair_content = post_llamus_chat(build_llamus_repair_prompt(chunk, content), model, base_url, api_key)
        try:
            return normalize_llm_metadata(extract_json_object(repair_content))
        except (json.JSONDecodeError, TypeError, ValueError) as second_error:
            return build_fallback_metadata(
                chunk,
                reason=f"fallo de parseo LLM ({first_error.__class__.__name__}, {second_error.__class__.__name__})",
                raw_response=repair_content or content,
            )


def enrich_chunks_with_llamus(
    chunks: list[dict[str, Any]],
    model: str,
    base_url: str,
    api_key: str,
    requestor: Callable[[dict[str, Any], str, str, str], dict[str, Any]] | None = None,
    enrich_limit: int | None = None,
) -> list[dict[str, Any]]:
    requester = requestor or request_llamus_metadata
    enriched: list[dict[str, Any]] = []
    enriched_count = 0
    for chunk in chunks:
        item = dict(chunk)
        can_enrich = should_enrich_chunk(chunk) and (enrich_limit is None or enriched_count < enrich_limit)
        if can_enrich:
            item["llm_metadata"] = requester(chunk, model, base_url, api_key)
            enriched_count += 1
        else:
            item["llm_metadata"] = None
        enriched.append(item)
    return enriched


def infer_source_family(pdf_path: pathlib.Path) -> str:
    name = pdf_path.name.lower()
    if "aeat" in name or "iart" in name or "imr_" in name:
        return "AEAT"
    if "c.g.e" in name or "infanual" in name:
        return "IGAE"
    if "presupuesto" in name or "pge" in name:
        return "PGE"
    return "UNKNOWN"


def find_related_xlsx(pdf_path: pathlib.Path) -> list[pathlib.Path]:
    stem = pdf_path.stem.lower()
    parent = pdf_path.parent
    candidates = []
    for path in parent.glob("*.xlsx"):
        if any(token in path.stem.lower() for token in stem.split()):
            candidates.append(path)
    if "presupuestos generales del estado consolidados 2023" in stem:
        candidates = [parent / "Copia de 01 Presupuestos Generales del Estado Consolidados2023.xlsx"]
    elif "presupuestos generales del estado consolidados 2024" in stem:
        candidates = [parent / "Copia de 01 Presupuestos Generales del Estado Consolidados2024.xlsx"]
    elif "presupuestos del estado" == stem:
        candidates = [parent / "Copia de 02 Presupuestos del Estado.xlsx"]
    elif "presupuesto de los organismos autónomos" in stem:
        candidates = [parent / "Copia de 04 Presupuesto de los Organismos Autónomos.xlsx"]
    return [path for path in candidates if path.exists()]


def process_document(
    pdf_path: pathlib.Path,
    output_root: pathlib.Path,
    xlsx_paths: list[pathlib.Path] | None = None,
    enrich: bool = False,
    llm_model: str = DEFAULT_LLAMUS_MODEL,
    llmus_base_url: str = DEFAULT_LLAMUS_BASE_URL,
    api_key: str | None = None,
    prefer_docling: bool = False,
    enrich_limit: int | None = None,
) -> dict[str, Any]:
    doc_id = pdf_path.stem
    source_family = infer_source_family(pdf_path)
    xlsx_paths = xlsx_paths or find_related_xlsx(pdf_path)

    pdf_blocks = extract_pdf_blocks(
        pdf_path=pdf_path,
        doc_id=doc_id,
        source_family=source_family,
        skip_patterns=DEFAULT_SKIP_PATTERNS,
        prefer_docling=prefer_docling,
    )
    xlsx_blocks = extract_xlsx_blocks(xlsx_paths=xlsx_paths, doc_id=doc_id, source_family=source_family)
    blocks = pdf_blocks + xlsx_blocks
    chunks = build_chunks(blocks)

    output_root.mkdir(parents=True, exist_ok=True)
    markdown_dir = output_root / "markdown"
    json_dir = output_root / "json"
    blocks_dir = output_root / "blocks"
    chunks_dir = output_root / "chunks"
    enriched_dir = output_root / "chunks_enriched"

    markdown_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    blocks_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = markdown_dir / f"{doc_id}.md"
    json_path = json_dir / f"{doc_id}.json"
    blocks_path = blocks_dir / f"{doc_id}.jsonl"
    chunks_path = chunks_dir / f"{doc_id}.jsonl"

    markdown_path.write_text(build_markdown_from_blocks(doc_id, blocks), encoding="utf-8")
    json_path.write_text(
        json.dumps(build_document_payload(doc_id, pdf_path, blocks), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(blocks_path, blocks)
    write_jsonl(chunks_path, chunks)

    result = {
        "doc_id": doc_id,
        "source_family": source_family,
        "block_count": len(blocks),
        "chunk_count": len(chunks),
        "markdown_path": str(markdown_path),
        "json_path": str(json_path),
        "blocks_path": str(blocks_path),
        "chunks_path": str(chunks_path),
    }

    if enrich:
        api_key = api_key or os.environ.get("LLAMUS_API_KEY")
        if not api_key:
            raise ValueError("LLAMUS_API_KEY no disponible para enriquecimiento.")
        enriched_dir.mkdir(parents=True, exist_ok=True)
        enriched_path = enriched_dir / f"{doc_id}.jsonl"
        enriched = enrich_chunks_with_llamus(
            chunks=chunks,
            model=llm_model,
            base_url=llmus_base_url,
            api_key=api_key,
            enrich_limit=enrich_limit,
        )
        write_jsonl(enriched_path, enriched)
        result["chunks_enriched_path"] = str(enriched_path)
    return result


def parse_args() -> argparse.Namespace:
    project_root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Pipeline de corpus para limpieza, chunking y enriquecimiento.")
    parser.add_argument("--pdf", type=pathlib.Path, required=True)
    parser.add_argument("--xlsx", type=pathlib.Path, nargs="*", default=None)
    parser.add_argument("--output-root", type=pathlib.Path, default=project_root / "PDF LIMPIO")
    parser.add_argument("--enrich", action="store_true")
    parser.add_argument("--enrich-limit", type=int, default=None)
    parser.add_argument("--llm-model", default=DEFAULT_LLAMUS_MODEL)
    parser.add_argument("--llamus-base-url", default=DEFAULT_LLAMUS_BASE_URL)
    parser.add_argument("--prefer-docling", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = process_document(
        pdf_path=args.pdf,
        output_root=args.output_root,
        xlsx_paths=args.xlsx,
        enrich=args.enrich,
        llm_model=args.llm_model,
        llmus_base_url=args.llamus_base_url,
        prefer_docling=args.prefer_docling,
        enrich_limit=args.enrich_limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
