from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Iterable

import fitz


DEFAULT_SKIP_PATTERNS = [
    r"^AGENCIA TRIBUTARIA - INFORME ANUAL 2024$",
    r"^INFORME ANUAL DE RECAUDACION TRIBUTARIA 2024 Pagina \d+$",
    r"^INFORME ANUAL DE RECAUDACI.N TRIBUTARIA 2024 P.gina \d+$",
    r"^INFORME ANUAL DE RECAUDACI.N TRIBUTARIA 2024\. P.gina \d+$",
    r"^Pagina \d+$",
]


def clean_page_text(text: str, skip_patterns: Iterable[str] | None = None) -> str:
    patterns = [re.compile(pattern) for pattern in (skip_patterns or [])]
    cleaned_lines: list[str] = []
    previous_blank = False

    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue

        previous_blank = False

        if any(pattern.match(line) for pattern in patterns):
            continue

        cleaned_lines.append(line)

    while cleaned_lines and cleaned_lines[0] == "":
        cleaned_lines.pop(0)
    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()

    return "\n".join(cleaned_lines)


def detect_page_metadata(text: str) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first_line = lines[0] if lines else ""

    dotted_leader_count = sum(1 for line in lines if re.search(r"\.{4,}", line))
    numeric_short_lines = sum(1 for line in lines if re.fullmatch(r"\d{1,3}", line))
    table_like = dotted_leader_count >= 2 or numeric_short_lines >= 3

    if re.match(r"^(ÍNDICE|INDICE)$", first_line, re.IGNORECASE):
        return {
            "page_kind": "index",
            "section_title": "ÍNDICE" if "Í" in first_line.upper() else "INDICE",
            "table_like": True,
        }

    if first_line and len(first_line) <= 120:
        section_title = first_line
    else:
        section_title = None

    return {
        "page_kind": "content",
        "section_title": section_title,
        "table_like": table_like,
    }


def extract_pdf_pages(pdf_path: pathlib.Path, skip_patterns: Iterable[str] | None = None) -> list[dict]:
    pages: list[dict] = []

    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document, start=1):
            text = page.get_text("text")
            cleaned = clean_page_text(text, skip_patterns=skip_patterns)
            if not cleaned:
                continue

            metadata = detect_page_metadata(cleaned)

            pages.append(
                {
                    "page": page_index,
                    "text": cleaned,
                    "page_kind": metadata["page_kind"],
                    "section_title": metadata["section_title"],
                    "table_like": metadata["table_like"],
                }
            )

    return pages


def build_markdown(doc_id: str, pages: list[dict]) -> str:
    lines = [f"# {doc_id}", ""]

    for page in pages:
        heading = f"## Pagina {page['page']}"
        if page.get("section_title"):
            heading = f"{heading} - {page['section_title']}"

        lines.append(heading)
        lines.append("")
        lines.append(
            f"- Tipo: {page.get('page_kind', 'content')} | Tabla/listado denso: {'si' if page.get('table_like') else 'no'}"
        )
        lines.append("")
        lines.append(page["text"])
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_payload(doc_id: str, source_pdf: pathlib.Path, pages: list[dict]) -> dict:
    return {
        "doc_id": doc_id,
        "source_pdf": str(source_pdf),
        "page_count": len(pages),
        "pages": pages,
    }


def extract_pdf_to_outputs(
    pdf_path: pathlib.Path,
    markdown_path: pathlib.Path,
    json_path: pathlib.Path,
    skip_patterns: Iterable[str] | None = None,
) -> dict:
    pdf_path = pathlib.Path(pdf_path)
    markdown_path = pathlib.Path(markdown_path)
    json_path = pathlib.Path(json_path)

    pages = extract_pdf_pages(pdf_path, skip_patterns=skip_patterns)
    doc_id = pdf_path.stem

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    markdown_path.write_text(build_markdown(doc_id=doc_id, pages=pages), encoding="utf-8")
    payload = build_payload(doc_id=doc_id, source_pdf=pdf_path, pages=pages)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "doc_id": doc_id,
        "page_count": len(pages),
        "markdown_path": str(markdown_path),
        "json_path": str(json_path),
    }


def parse_args() -> argparse.Namespace:
    project_root = pathlib.Path(__file__).resolve().parents[1]
    default_pdf = project_root / "PDF SIN LIMPIAR" / "AEAT_informe_anual_2024.pdf"
    default_markdown = project_root / "PDF LIMPIO" / "markdown" / "AEAT_informe_anual_2024.md"
    default_json = project_root / "PDF LIMPIO" / "json" / "AEAT_informe_anual_2024.json"

    parser = argparse.ArgumentParser(
        description="Extrae un PDF de AEAT a markdown y json con una limpieza basica."
    )
    parser.add_argument("--pdf", type=pathlib.Path, default=default_pdf)
    parser.add_argument("--markdown", type=pathlib.Path, default=default_markdown)
    parser.add_argument("--json", type=pathlib.Path, default=default_json)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = extract_pdf_to_outputs(
        pdf_path=args.pdf,
        markdown_path=args.markdown,
        json_path=args.json,
        skip_patterns=DEFAULT_SKIP_PATTERNS,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
