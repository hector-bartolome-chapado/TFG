from __future__ import annotations

from generacion.text_utils import content_tokens, normalize_for_generation


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

