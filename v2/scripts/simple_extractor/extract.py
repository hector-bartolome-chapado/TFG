from __future__ import annotations

import pathlib
import tempfile
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from pypdf import PdfReader, PdfWriter


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


def build_docling_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def strip_docling_markdown_noise(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "<!-- image -->":
            continue
        lines.append(raw_line)
    return "\n".join(lines)


def write_single_page_pdf(reader: PdfReader, page_index: int, output_path: pathlib.Path) -> None:
    writer = PdfWriter()
    writer.add_page(reader.pages[page_index])
    with output_path.open("wb") as handle:
        writer.write(handle)


def extract_pages(pdf_path: pathlib.Path) -> list[dict[str, Any]]:
    converter = build_docling_converter()
    reader = PdfReader(str(pdf_path))
    pages: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        for page_index in range(len(reader.pages)):
            page_number = page_index + 1
            page_pdf_path = tmp_path / f"page-{page_number}.pdf"
            write_single_page_pdf(reader, page_index, page_pdf_path)
            conversion = converter.convert(page_pdf_path)
            text = conversion.document.export_to_markdown()
            text = clean_page_text(strip_docling_markdown_noise(text))
            pages.append({"page": page_number, "text": text})

    return pages




