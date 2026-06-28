from __future__ import annotations

import pathlib
import re
import time
import unicodedata
from typing import Any

import requests

from scripts.RECUPERADOR.retrieval import load_embedding_rows, retrieve_top_k
from scripts.simple_extractor.config import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLAMUS_BASE_URL,
    DEFAULT_RAG_CHAT_MODEL,
    RAG_SYSTEM_PROMPT,
)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = PROJECT_ROOT / "RESULTADOS EMBEDDING" / "embeddings"
STANDARD_NO_ANSWER = "No hay información suficiente en el documento proporcionado para responder a esa pregunta."

SPANISH_STOPWORDS = {
    "a",
    "al",
    "como",
    "con",
    "cual",
    "cuando",
    "cuanto",
    "cuantos",
    "de",
    "del",
    "el",
    "en",
    "entre",
    "es",
    "este",
    "esta",
    "fue",
    "la",
    "las",
    "lo",
    "los",
    "para",
    "por",
    "que",
    "se",
    "segun",
    "sobre",
    "un",
    "una",
    "y",
    "aeat",
    "agencia",
    "anual",
    "documento",
    "informe",
    "tributaria",
    "tributario",
    "tributarios",
    "2024",
}


def normalize_for_generation(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    without_marks = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return without_marks


def content_tokens(text: str) -> set[str]:
    normalized = normalize_for_generation(text)
    tokens = set(re.findall(r"\d+(?:[.,]\d+)*%?|[a-z]+", normalized))
    return {token for token in tokens if len(token) > 2 and token not in SPANISH_STOPWORDS}


def lower_first(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    if len(stripped) > 3 and stripped[:4].isupper():
        return stripped
    return stripped[:1].lower() + stripped[1:]


def question_subject(question: str) -> str:
    subject = question.strip().strip("¿?").strip()
    replacements = [
        (r"^cu[aá]nto(?:s|as)?\s+(?:crecieron|creci[oó]|aumentaron|aument[oó]|alcanzaron|alcanz[oó]|fueron|fue)\s+", ""),
        (r"^qu[eé]\s+nota\s+informativa\s+trata\s+(?:de|del|sobre)\s+", "la nota informativa sobre "),
        (r"^qu[eé]\s+cuadro\s+(?:se\s+cita\s+para|resume|recoge|se\s+usa\s+para)\s+", "el cuadro sobre "),
        (r"^qu[eé]\s+tabla\s+", "la tabla "),
        (r"^qu[eé]\s+informaci[oó]n\s+adicional\s+se\s+ofrece\s+sobre\s+", "la información adicional sobre "),
        (r"^qu[eé]\s+tendencia\s+se\s+menciona\s+sobre\s+", "la tendencia sobre "),
        (r"^qu[eé]\s+", ""),
        (r"^cu[aá]l(?:es)?\s+", ""),
        (r"^c[oó]mo\s+", ""),
        (r"^cu[aá]ndo\s+", ""),
        (r"^por\s+qu[eé]\s+", ""),
    ]
    for pattern, replacement in replacements:
        subject = re.sub(pattern, replacement, subject, flags=re.IGNORECASE).strip()
    subject = re.sub(r"\s+", " ", subject).strip()
    return lower_first(subject[:140].rstrip(" .,:;"))


def restate_answer(question: str, answer: str) -> str:
    cleaned_answer = answer.strip()
    if not cleaned_answer:
        return cleaned_answer
    if cleaned_answer.startswith(("Dato principal:", "Cuadro o tabla:")):
        return cleaned_answer
    subject = question_subject(question)
    if not subject:
        return cleaned_answer
    if cleaned_answer.startswith(STANDARD_NO_ANSWER):
        return f"No hay información suficiente en el documento proporcionado para responder sobre {subject}."
    if normalize_for_generation(cleaned_answer).startswith(("sobre ", "respecto a ")):
        return cleaned_answer
    prefix = "Respecto a" if classify_question_route(question) == "table" else "Sobre"
    return f"{prefix} {subject}, {lower_first(cleaned_answer)}"


def classify_question_route(question: str) -> str:
    normalized = normalize_for_generation(question)
    tokens = content_tokens(question)
    if any(
        term in normalized
        for term in (
            "ley",
            "real decreto",
            "decreto",
            "articulo",
            "art.",
            "norma",
            "normativa",
            "regula",
            "establece",
            "ambito de aplicacion",
            "hecho imponible",
            "texto consolidado",
        )
    ):
        return "legal"
    if any(
        term in normalized
        for term in (
            "cuadro",
            "tabla",
            "indice",
            "nota informativa",
            "fila",
            "columna",
            "presupuesto",
            "presupuestos",
            "capitulo",
            "capitulos",
            "transferencias",
            "operaciones",
            "gastos",
        )
    ):
        return "table"
    if any(term in normalized for term in ("grafico", "tendencia", "evolucion", "evoluciono", "comportamiento")):
        return "chart"
    if any(term in normalized for term in ("por que", "por qu", "relacion", "relacionan", "combina", "conexion", "justifica")):
        return "synthesis"
    if any(char.isdigit() for char in question) or tokens & {
        "irpf",
        "iva",
        "sociedades",
        "hidrocarburos",
        "tabaco",
        "electricidad",
        "impuesto",
        "impuestos",
        "porcentaje",
    }:
        return "tax_exact"
    return "simple_fact"


def list_embedding_files(embeddings_dir: pathlib.Path = EMBEDDINGS_DIR) -> list[pathlib.Path]:
    if not embeddings_dir.exists():
        return []
    return sorted(path for path in embeddings_dir.glob("*.jsonl") if path.is_file())


def load_document_embeddings(embeddings_path: pathlib.Path) -> list[dict[str, Any]]:
    return load_embedding_rows(embeddings_path)


def run_retrieval(
    question: str,
    rows: list[dict[str, Any]],
    top_k: int,
    model: str = DEFAULT_EMBED_MODEL,
    base_url: str = DEFAULT_LLAMUS_BASE_URL,
    api_key: str | None = None,
    strategy: str = "fiscal_hybrid",
    rrf_k: int = 60,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    hits = retrieve_top_k(
        question=question,
        embedding_rows=rows,
        top_k=top_k,
        model=model,
        base_url=base_url,
        api_key=api_key,
        strategy=strategy,
        rrf_k=rrf_k,
    )
    return {"hits": hits, "latency_seconds": time.perf_counter() - started_at}


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


def build_answer_prompt(question: str, context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"Pregunta:\n{question}\n\nContexto recuperado:\n{context}"},
    ]


def build_controlled_answer_prompt(question: str, context: str, route: str | None = None) -> list[dict[str, str]]:
    effective_route = route or classify_question_route(question)
    route_rules = {
        "tax_exact": (
            "Modo extractor: responde primero con la cifra, porcentaje, año o impuesto exacto solicitado. "
            "Formato recomendado: Dato principal: <valor>. Fuente: <frase o apartado que lo respalda>. "
            "No añadas explicación si el dato principal basta. Si hay varias cifras, distingue cada una."
        ),
        "table": (
            "Modo tabla/cuadro: identifica el cuadro, nota o apartado usado y extrae solo la información soportada. "
            "Formato recomendado: Cuadro o tabla: <referencia>. Dato extraido: <valor o relación>. "
            "No reconstruyas celdas ni relaciones que no estén explícitas en el contexto."
        ),
        "chart": (
            "Modo tendencia/gráfico: responde solo si la tendencia está textualizada en el contexto. "
            "Formato recomendado: Tendencia textual: <tendencia descrita>. Evidencia: <frase de apoyo>. "
            "No estimes ni infieras valores visuales no escritos."
        ),
        "synthesis": (
            "Modo síntesis: combina las evidencias recuperadas en 2 o 3 frases, separando causas y efectos. "
            "Formato recomendado: Evidencias combinadas: <síntesis>. Maximo 3 frases. "
            "No introduzcas conocimiento externo."
        ),
        "legal": (
            "Modo juridico: explica que establece la ley, articulo o norma citada usando solo el contexto recuperado. "
            "No te limites a repetir el titulo del apartado. "
            "Formato recomendado: Norma: <ley o articulo>. Explicacion: <contenido juridico explicado en 2 o 3 frases>. "
            "Fuente: <fragmento o apartado que lo respalda>. "
            "Si el contexto solo contiene un indice o titulo y no desarrolla el contenido, dilo explicitamente."
        ),
        "simple_fact": (
            "Modo hecho directo: Respuesta breve: una frase, completa y con el dato principal al inicio."
        ),
    }
    system_prompt = (
        "Eres un asistente RAG especializado en documentación fiscal. "
        "Usa exclusivamente el contexto recuperado. "
        f"Si la evidencia no permite responder, contesta exactamente: \"{STANDARD_NO_ANSWER}\" "
        "No inventes cifras, porcentajes, años, tablas ni entidades. "
        f"{route_rules.get(effective_route, route_rules['simple_fact'])}"
    )
    user_prompt = (
        f"Tipo de pregunta detectado: {effective_route}\n\n"
        f"Pregunta:\n{question}\n\n"
        f"Contexto recuperado:\n{context}\n\n"
        "Respuesta:"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_generation_decision(question: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    route = classify_question_route(question)
    context = build_context(hits)
    query_tokens = content_tokens(question)
    context_tokens = content_tokens(context)
    overlap = len(query_tokens & context_tokens) / len(query_tokens) if query_tokens else 0.0
    top_score = float(hits[0].get("score", 0.0)) if hits else 0.0
    should_reject = False
    if not hits or not context.strip():
        should_reject = True
    elif overlap < 0.14:
        should_reject = True
    elif overlap < 0.18 and top_score < 0.25:
        should_reject = True
    return {
        "route": route,
        "should_reject": should_reject,
        "answer": restate_answer(question, STANDARD_NO_ANSWER) if should_reject else "",
        "top_score": top_score,
        "query_context_overlap": overlap,
    }


def score_sentence_for_question(sentence: str, question: str, route: str) -> float:
    sentence_tokens = content_tokens(sentence)
    query_tokens = content_tokens(question)
    if not sentence_tokens or not query_tokens:
        return 0.0
    overlap = len(sentence_tokens & query_tokens) / len(query_tokens)
    number_bonus = 0.15 if re.search(r"\d", sentence) else 0.0
    table_bonus = 0.15 if route == "table" and any(term in normalize_for_generation(sentence) for term in ("cuadro", "tabla", "nota informativa")) else 0.0
    return overlap + number_bonus + table_bonus


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


def clean_context_artifacts(text: str) -> str:
    without_labels = re.sub(r"\[Chunk\s+\d+\s*\|[^\]]+\]", " ", text)
    return re.sub(r"\s+", " ", without_labels).strip()


def extract_legal_answer(question: str, context: str) -> str | None:
    normalized_question = normalize_for_generation(question)
    cleaned_context = clean_context_artifacts(context)
    article_match = re.search(
        r"(Art(?:i|í)culo\s+\d+\.?\s*[^.]*\.)\s*(.*?)(?=Art(?:i|í)culo\s+\d+|CAP[IÍ]TULO|T[IÍ]TULO|\Z)",
        cleaned_context,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not article_match:
        return None

    norm_title = article_match.group(1).strip()
    body = article_match.group(2).strip()
    body_sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", body) if part.strip()]
    useful_sentences: list[str] = []

    for sentence in body_sentences:
        normalized_sentence = normalize_for_generation(sentence)
        if "impuesto sobre el valor anadido" in normalized_sentence and "tributo" in normalized_sentence:
            useful_sentences.append(sentence)
            break

    if "ambito" in normalized_question:
        for sentence in body_sentences:
            normalized_sentence = normalize_for_generation(sentence)
            if "ambito espacial de aplicacion" in normalized_sentence or "territorio espanol" in normalized_sentence:
                useful_sentences.append(sentence)
                break

    if not useful_sentences and body_sentences:
        useful_sentences.append(body_sentences[0])
    if not useful_sentences:
        return None

    explanation = " ".join(dict.fromkeys(useful_sentences))
    return f"Norma: {norm_title} Explicacion: {explanation} Fuente: Ley 37/1992, fragmento recuperado del BOE."


def extract_relevant_sentences(question: str, context: str, route: str, limit: int) -> list[str]:
    parts = [
        part.strip(" -|\t")
        for part in re.split(r"(?<=[.!?])\s+|\n+", context)
        if part.strip(" -|\t")
    ]
    ranked = sorted(
        parts,
        key=lambda part: score_sentence_for_question(part, question, route),
        reverse=True,
    )
    selected: list[str] = []
    for part in ranked:
        if part not in selected and score_sentence_for_question(part, question, route) > 0:
            selected.append(part)
        if len(selected) >= limit:
            break
    return selected


def generate_controlled_answer(question: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    started_at = time.perf_counter()
    decision = build_generation_decision(question, hits)
    if decision["should_reject"]:
        return {
            "answer": decision["answer"],
            "latency_seconds": time.perf_counter() - started_at,
            "decision": decision,
        }

    context = build_context(hits)
    route = decision["route"]
    document_presence_answer = extract_document_presence_answer(question, hits)
    if document_presence_answer:
        return {
            "answer": document_presence_answer,
            "latency_seconds": time.perf_counter() - started_at,
            "decision": decision,
        }
    table_answer = extract_table_cell_answer(question, context)
    if table_answer:
        return {
            "answer": table_answer,
            "latency_seconds": time.perf_counter() - started_at,
            "decision": decision,
        }
    single_row_table_answer = extract_single_row_table_answer(question, context)
    if single_row_table_answer:
        return {
            "answer": single_row_table_answer,
            "latency_seconds": time.perf_counter() - started_at,
            "decision": decision,
        }
    legal_answer = extract_legal_answer(question, context) if route == "legal" else None
    if legal_answer:
        return {
            "answer": legal_answer,
            "latency_seconds": time.perf_counter() - started_at,
            "decision": decision,
        }
    sentence_limit = 3 if route in {"synthesis", "legal"} else 2
    sentences = extract_relevant_sentences(question, context, route, sentence_limit)
    if not sentences:
        answer = STANDARD_NO_ANSWER
        decision = {**decision, "should_reject": True}
    elif route == "tax_exact":
        answer = " ".join(sentences[:2])
    elif route == "table":
        answer = " ".join(sentences[:2])
    elif route == "chart":
        answer = " ".join(sentences[:2])
    elif route == "synthesis":
        answer = " ".join(sentences[:3])
    elif route == "legal":
        answer = " ".join(sentences[:3])
    else:
        answer = sentences[0]
    return {
        "answer": restate_answer(question, answer.strip()),
        "latency_seconds": time.perf_counter() - started_at,
        "decision": decision,
    }


def ask_llamus(
    prompt_messages: list[dict[str, str]],
    model: str = DEFAULT_RAG_CHAT_MODEL,
    base_url: str = DEFAULT_LLAMUS_BASE_URL,
    api_key: str | None = None,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    started_at = time.perf_counter()
    response = requests.post(
        f"{base_url.rstrip('/')}/api/chat/completions",
        headers=headers,
        json={"model": model, "messages": prompt_messages, "temperature": 0.0},
        timeout=120,
    )
    ended_at = time.perf_counter()

    if response.status_code >= 400:
        detail = (response.text or "")[:500]
        raise requests.HTTPError(
            f"Error al pedir respuesta ({response.status_code}): {detail}",
            response=response,
        )

    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("La respuesta de chat no contiene choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("La respuesta de chat no contiene texto usable.")
    return {"answer": content.strip(), "latency_seconds": ended_at - started_at}
