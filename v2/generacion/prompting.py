from __future__ import annotations

from ingesta.config import RAG_SYSTEM_PROMPT
from generacion.question_routing import classify_question_route
from generacion.text_utils import STANDARD_NO_ANSWER


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
            "No añadas explicacion si el dato principal basta. Si hay varias cifras, distingue cada una."
        ),
        "table": (
            "Modo tabla/cuadro: identifica el cuadro, nota o apartado usado y extrae solo la informacion soportada. "
            "Formato recomendado: Cuadro o tabla: <referencia>. Dato extraido: <valor o relacion>. "
            "No reconstruyas celdas ni relaciones que no esten explicitas en el contexto."
        ),
        "chart": (
            "Modo tendencia/grafico: responde solo si la tendencia esta textualizada en el contexto. "
            "Formato recomendado: Tendencia textual: <tendencia descrita>. Evidencia: <frase de apoyo>. "
            "No estimes ni infieras valores visuales no escritos."
        ),
        "synthesis": (
            "Modo sintesis: combina las evidencias recuperadas en 2 o 3 frases, separando causas y efectos. "
            "Formato recomendado: Evidencias combinadas: <sintesis>. Maximo 3 frases. "
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
        "Eres un asistente RAG especializado en documentacion fiscal. "
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

