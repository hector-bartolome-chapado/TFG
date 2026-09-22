from __future__ import annotations

import re
from typing import Any

from generacion.text_utils import content_tokens, normalize_for_generation


def parse_table_line(line: str) -> tuple[int | None, list[str]] | None:
    match = re.match(r"\s*Fila\s+(\d+):\s*(.+)$", line.strip(), flags=re.IGNORECASE)
    if not match or "|" not in match.group(2):
        return None
    cells = [cell.strip() for cell in match.group(2).split("|")]
    cells = [cell for cell in cells if cell]
    if len(cells) < 2:
        return None
    return int(match.group(1)), cells


def looks_like_year_header(cells: list[str]) -> bool:
    year_like = sum(1 for cell in cells[1:] if re.fullmatch(r"\d{4}(?:-P)?", cell.strip()))
    return year_like >= 2


def requested_table_header(question: str, headers: list[str]) -> tuple[int, str] | None:
    normalized_question = normalize_for_generation(question)
    requested_years = re.findall(r"\b(20\d{2})\b", question)
    if not requested_years:
        return None
    requested_year = requested_years[-1]
    wants_prorogued = any(term in normalized_question for term in ("prorrog", "-p", " p ", "presupuesto prorrogado"))

    candidates: list[tuple[int, str]] = []
    for index, header in enumerate(headers):
        normalized_header = normalize_for_generation(header)
        if normalized_header == requested_year:
            candidates.append((index, header))
        elif normalized_header == f"{requested_year}-p":
            candidates.append((index, header))

    if not candidates:
        return None
    if wants_prorogued:
        for candidate in candidates:
            if candidate[1].endswith("-P"):
                return candidate
    for candidate in candidates:
        if not candidate[1].endswith("-P"):
            return candidate
    return candidates[0]


def infer_pge_headers(question: str, context: str, value_count: int) -> list[str]:
    if value_count != 10:
        return []
    normalized_context = normalize_for_generation(context)
    normalized_question = normalize_for_generation(question)
    if "2025" in normalized_question or "2024" in normalized_question:
        return ["2017", "2018", "2018-P", "2019-P", "2021", "2022", "2023", "2023-P", "2024-P", "2025-P"]
    if "consolidados2023" in normalized_context or "2015-2023" in normalized_context:
        return ["2015", "2016", "2017", "2018", "2018-P", "2019-P", "2021", "2022", "2023", "2023-P"]
    return ["2017", "2018", "2018-P", "2019-P", "2021", "2022", "2023", "2023-P", "2024-P", "2025-P"]


def score_table_row_label(question: str, label: str) -> float:
    query_tokens = content_tokens(question)
    label_tokens = content_tokens(label)
    if not label_tokens:
        return 0.0
    overlap = len(query_tokens & label_tokens)
    normalized_label = normalize_for_generation(label)
    normalized_question = normalize_for_generation(question)
    exact_bonus = 0.0
    if normalized_label in normalized_question:
        exact_bonus += 2.0
    if "total" in normalized_question and "total" in normalized_label:
        exact_bonus += 1.0
    if "presupuesto" in normalized_question and "presupuesto" in normalized_label:
        exact_bonus += 1.0
    return overlap + exact_bonus


def format_table_value(value: str) -> str:
    cleaned = value.strip()
    try:
        number = float(cleaned.replace(",", "."))
    except ValueError:
        return cleaned
    formatted = f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if formatted.endswith(",00"):
        formatted = formatted[:-3]
    return formatted


def extract_table_cell_answer(question: str, context: str) -> str | None:
    current_headers: list[str] | None = None
    best: dict[str, Any] | None = None
    current_source = ""

    for raw_line in context.splitlines():
        line = raw_line.strip()
        if line.startswith("Documento:") or line.startswith("Hoja:") or line.startswith("Rango de filas:"):
            current_source = f"{current_source} {line}".strip()
        parsed = parse_table_line(line)
        if not parsed:
            continue
        row_number, cells = parsed
        if looks_like_year_header(cells):
            current_headers = cells[1:]
            continue
        effective_headers = current_headers
        if not effective_headers:
            effective_headers = infer_pge_headers(question, context, len(cells) - 1)
        if not effective_headers:
            continue

        requested = requested_table_header(question, effective_headers)
        if not requested:
            continue
        column_index, header = requested
        value_index = column_index + 1
        if value_index >= len(cells):
            continue

        label = cells[0]
        row_score = score_table_row_label(question, label)
        if row_score <= 0:
            continue
        candidate = {
            "score": row_score,
            "row_number": row_number,
            "label": label,
            "header": header,
            "value": cells[value_index],
            "source": current_source.strip(),
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    if not best:
        return None

    value = format_table_value(str(best["value"]))
    source_parts = [part for part in str(best["source"]).split(" Documento:") if part.strip()]
    source = source_parts[-1].strip() if source_parts else str(best["source"]).strip()
    return (
        f"Dato principal: {best['label']} en {best['header']}: {value} millones de euros. "
        f"Fuente: fila {best['row_number']}. {source}"
    )


def extract_single_row_table_answer(question: str, context: str) -> str | None:
    normalized_question = normalize_for_generation(question)
    month_terms = (
        "enero",
        "febrero",
        "marzo",
        "abril",
        "mayo",
        "junio",
        "julio",
        "agosto",
        "septiembre",
        "octubre",
        "noviembre",
        "diciembre",
    )
    requested_month = next((month for month in month_terms if month in normalized_question), None)
    requested_years = re.findall(r"\b(20\d{2})\b", question)
    if not requested_month or not requested_years:
        return None
    requested_year = requested_years[-1]

    current_source = ""
    for raw_line in context.splitlines():
        line = raw_line.strip()
        if line.startswith("Documento:") or line.startswith("Hoja:") or line.startswith("Rango de filas:"):
            current_source = f"{current_source} {line}".strip()
        match = re.match(r"\s*Fila\s+(\d+):\s*(.+)$", line, flags=re.IGNORECASE)
        if not match:
            continue
        value = match.group(2).strip()
        normalized_value = normalize_for_generation(value)
        if requested_month in normalized_value and requested_year in value:
            return f"Dato principal: {value}. Fuente: fila {match.group(1)}. {current_source}"
    return None


def extract_row_code_concept_answer(question: str, hits: list[dict[str, Any]]) -> str | None:
    normalized_question = normalize_for_generation(question)
    if not any(term in normalized_question for term in (
        "concepto", "denominacion", "programa", "que figura", "que corresponde", "que es",
    )):
        return None
    code_match = re.search(r"\bC\d{2}\.I\d{2}\b", question, flags=re.IGNORECASE)
    if not code_match:
        return None
    requested_code = code_match.group().upper()

    for hit in hits:
        if hit.get("sheet") is None:
            continue
        context = str(hit.get("context_text") or hit.get("parent_text") or hit.get("text") or "")
        for line in context.splitlines():
            parsed = parse_table_line(line)
            if not parsed:
                continue
            row_number, cells = parsed
            for cell in cells:
                concept_match = re.match(
                    rf"^{re.escape(requested_code)}\s+(.+)$", cell, flags=re.IGNORECASE,
                )
                if not concept_match:
                    continue
                concept = concept_match.group(1).strip()
                source_file = str(hit.get("source_file") or hit.get("doc_id") or "documento")
                source_name = source_file.replace("\\", "/").rsplit("/", 1)[-1]
                return (
                    f"Dato principal: {requested_code} corresponde a «{concept}». "
                    f"Fuente: {source_name}, hoja {hit['sheet']}, fila {row_number}."
                )
    return None


def extract_document_presence_answer(question: str, hits: list[dict[str, Any]]) -> str | None:
    normalized_question = normalize_for_generation(question)
    if not any(pattern in normalized_question for pattern in ("en que documentos", "que documentos", "donde aparecen")):
        return None

    doc_ids: list[str] = []
    for hit in hits:
        doc_id = str(hit.get("doc_id") or "").strip()
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)
    if not doc_ids:
        return None

    formatted_docs = "; ".join(doc_ids[:6])
    return f"Documentos localizados: {formatted_docs}. Fuente: documentos recuperados en el top-{min(len(hits), 6)} del RAG."

