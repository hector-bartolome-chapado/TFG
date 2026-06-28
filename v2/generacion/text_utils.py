from __future__ import annotations

import re
import unicodedata


STANDARD_NO_ANSWER = "No hay informacion suficiente en el documento proporcionado para responder a esa pregunta."

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
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


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
        (r"^qu[eé]\s+informaci[oó]n\s+adicional\s+se\s+ofrece\s+sobre\s+", "la informacion adicional sobre "),
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

