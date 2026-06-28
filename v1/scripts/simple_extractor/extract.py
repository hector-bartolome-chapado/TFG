from __future__ import annotations

import pathlib
from typing import Any

import fitz


# Limpia el texto bruto de una página PDF para dejar solo saltos y líneas útiles.
#
# Entra:
# - text: texto bruto extraído por PyMuPDF.
# Sale:
# - un texto más legible, sin espacios sobrantes ni varias líneas vacías seguidas.
# Por qué existe:
# - el texto bruto de PDF suele venir con mucho ruido visual que dificulta entender el flujo.
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


# Extrae el texto de un PDF página a página usando un parser de texto simple.
#
# Entra:
# - pdf_path: ruta al PDF.
# Sale:
# - una lista de páginas con número y texto limpio.
# Por qué existe:
# - esta es la capa más simple del pipeline: leer texto sin OCR, sin layout complejo y sin heurísticas.
def extract_pages(pdf_path: pathlib.Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = clean_page_text(page.get_text("text"))
            pages.append({"page": page_number, "text": text})
    return pages





