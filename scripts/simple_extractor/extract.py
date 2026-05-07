from __future__ import annotations

import pathlib
from typing import Any

import fitz


def clean_page_text(text: str) -> str:
    cleaned_lines: list[str] = []
    previous_blank = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if cleaned_lines and not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue
        cleaned_lines.append(line)
        previous_blank = False
    return "\n".join(cleaned_lines).strip()


def extract_pages(pdf_path: pathlib.Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = clean_page_text(page.get_text("text"))
            pages.append({"page": page_number, "text": text})
    return pages





