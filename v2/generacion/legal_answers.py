from __future__ import annotations

import re

from generacion.text_utils import normalize_for_generation


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

