from __future__ import annotations

import html
import logging
import pathlib
import sys
from typing import Any

import requests
import streamlit as st

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generacion.rag_service import generate_controlled_answer, list_embedding_files, run_retrieval
from ingesta.config import DEFAULT_LLAMUS_BASE_URL, get_api_key
from ingesta.embeddings import request_embedding
from interfaz.public_service import evidence_label, load_corpus, resolve_question, source_name


EXAMPLE_QUESTIONS = (
    "¿Cuáles fueron los ingresos tributarios en 2024?",
    "¿Qué establece el artículo 1 de la Ley 37/1992 sobre el IVA?",
    "¿Qué concepto figura en la fila C09.I01 del Excel de PGE 2024?",
)

PAGE_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Libre+Baskerville:wght@400;700&display=swap');
:root { --ink: #182b3d; --muted: #536578; --paper: #f6f3ec; --line: #d7dfdf; --accent: #137973; }
html, body, [data-testid="stAppViewContainer"] { background: var(--paper); color: var(--ink); font-family: 'DM Sans', sans-serif; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1480px; padding-top: 2.2rem; padding-bottom: 4rem; }
h1, h2, h3 { font-family: 'Libre Baskerville', Georgia, serif; color: var(--ink); letter-spacing: -.025em; }
.masthead { border-top: 4px solid var(--ink); border-bottom: 1px solid var(--line); padding: 1.5rem 0 1.25rem; margin-bottom: 1.8rem; }
.eyebrow { color: var(--accent); font-size: .76rem; letter-spacing: .17em; text-transform: uppercase; font-weight: 700; }
.masthead h1 { font-size: clamp(2rem, 4vw, 3.25rem); margin: .35rem 0 .55rem; }
.masthead p { color: var(--muted); font-size: 1.03rem; max-width: 760px; line-height: 1.55; margin: 0; }
.section-title { font-size: .82rem; letter-spacing: .15em; text-transform: uppercase; color: var(--muted); font-weight: 700; border-bottom: 1px solid var(--line); padding: .65rem 0 .8rem; margin-bottom: 1rem; }
.dossier-kicker { color: var(--accent); font-weight: 700; letter-spacing: .12em; font-size: .78rem; text-transform: uppercase; }
.source-note { border-left: 3px solid var(--accent); padding-left: .8rem; margin: .75rem 0; font-size: .88rem; color: var(--muted); }
.quiet-note { font-size: .83rem; color: var(--muted); line-height: 1.5; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--line); background: #fffefa; }
[data-testid="stChatMessage"] { border: 1px solid var(--line); background: #fffefa; border-radius: 8px; margin-bottom: .75rem; }
[data-testid="stChatMessage"] p { color: var(--ink) !important; line-height: 1.62; }
[data-testid="stChatMessage"] [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
[data-testid="stAppViewContainer"] [data-testid="stMarkdown"] p { color: var(--ink); }
[data-testid="stHeader"] button, [data-testid="stHeader"] svg { color: var(--ink) !important; }
[data-testid="stToolbarActions"] button, [data-testid="stToolbarActions"] span, [data-testid="stToolbarActions"] svg { color: var(--ink) !important; fill: var(--ink) !important; }
.stButton > button { border-radius: 5px; font-weight: 600; border: 1px solid #b6caca !important; background: #fffefa !important; color: var(--ink) !important; }
.stButton > button p { color: var(--ink) !important; }
.stButton > button:hover { border-color: var(--accent) !important; color: var(--accent) !important; }
.stButton > button:hover p { color: var(--accent) !important; }
.stButton > button:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
.stButton > button[kind="primary"] { background: var(--accent) !important; border-color: var(--accent) !important; color: #fff !important; }
.stButton > button[kind="primary"] p { color: #fff !important; }
[data-testid="stChatInput"] { background: #fffefa !important; border: 1px solid #b6caca; }
[data-testid="stChatInput"] > div, [data-testid="stChatInput"] > div > div { background: #fffefa !important; }
[data-testid="stChatInput"] textarea { color: var(--ink) !important; background: #fffefa !important; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted) !important; opacity: 1; }
[data-testid="stChatInput"] button, [data-testid="stChatInput"] svg { color: var(--accent) !important; fill: var(--accent) !important; }
[data-testid="stChatInputSubmitButton"] { background: #e7f4f1 !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] p { color: var(--ink) !important; }
[data-testid="stRadio"] label, [data-testid="stRadio"] p, [data-testid="stRadio"] span { color: var(--ink) !important; }
[data-testid="stRadio"] p { white-space: normal !important; overflow-wrap: anywhere; }
[data-testid="stRadio"] input { accent-color: var(--accent); }
.evidence-excerpt { color: var(--ink); background: #f3f7f5; border: 1px solid var(--line); border-left: 3px solid var(--accent); padding: 1rem; border-radius: 4px; line-height: 1.6; white-space: pre-wrap; overflow-wrap: anywhere; }
.evidence-excerpt.spreadsheet { font-family: ui-monospace, Consolas, monospace; font-size: .86rem; }
@media (max-width: 760px) {
  .block-container { padding: 1.1rem .9rem 3rem; }
  .masthead h1 { font-size: 2rem; }
  .stButton > button { height: auto !important; min-height: 2.75rem; white-space: normal !important; }
  .stButton > button p { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; line-height: 1.35; }
}
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; } }
</style>
"""


@st.cache_resource(show_spinner=False)
def load_cached_corpus(paths: tuple[str, ...]) -> list[dict[str, Any]]:
    return load_corpus([pathlib.Path(path) for path in paths])


def select_evidence(turn_index: int, hit_index: int) -> None:
    st.session_state.selected_evidence = (turn_index, hit_index)
    st.session_state.evidence_selector = hit_index


def select_evidence_from_dossier() -> None:
    turn_index = st.session_state.selected_evidence[0]
    st.session_state.selected_evidence = (turn_index, st.session_state.evidence_selector)


def render_excerpt(value: Any, *, spreadsheet: bool = False) -> None:
    style = "evidence-excerpt spreadsheet" if spreadsheet else "evidence-excerpt"
    st.markdown(f'<div class="{style}">{html.escape(str(value))}</div>', unsafe_allow_html=True)


def reset_conversation() -> None:
    st.session_state.turns = []
    st.session_state.selected_evidence = None
    st.session_state.evidence_selector = 0
    st.session_state.pending_clarification = False
    st.session_state.failed_query = None


def failure_message(phase: str, error: Exception) -> str:
    if isinstance(error, requests.exceptions.Timeout):
        reason = "Llamus no respondió dentro del tiempo de espera."
    elif isinstance(error, requests.exceptions.HTTPError):
        status = error.response.status_code if error.response is not None else "desconocido"
        reason = f"Llamus devolvió un error HTTP {status}."
    elif isinstance(error, requests.exceptions.ConnectionError):
        reason = "No se pudo conectar con Llamus."
    else:
        reason = "No se pudo completar esta fase."
    return f"Fallo en {phase}: {reason} Puedes reintentar la consulta."


def run_public_query(question: str, rows: list[dict[str, Any]], api_key: str) -> None:
    history = st.session_state.turns
    try:
        with st.spinner("Interpretando la pregunta…"):
            resolved = resolve_question(
                question, history, api_key=api_key,
                force_rewrite=st.session_state.pending_clarification,
            )
    except Exception as error:
        logging.exception("Error en la interpretación de la repregunta")
        st.session_state.failed_query = {"question": question, "message": failure_message("interpretación", error)}
        return

    if resolved["clarification"]:
        history.append({"question": question, "answer": resolved["clarification"], "hits": [], "clarification": True})
        st.session_state.pending_clarification = True
        st.session_state.failed_query = None
        return

    standalone_question = str(resolved["question"])
    try:
        with st.spinner("Buscando evidencia en el corpus…"):
            result = run_retrieval(
                question=standalone_question, rows=rows, top_k=3,
                base_url=DEFAULT_LLAMUS_BASE_URL, api_key=api_key,
                embedder=lambda text, model, base_url, key: request_embedding(
                    text, model, base_url, key, timeout_seconds=35,
                ),
            )
    except Exception as error:
        logging.exception("Error en la recuperación pública")
        st.session_state.failed_query = {"question": question, "message": failure_message("recuperación", error)}
        return

    try:
        with st.spinner("Redactando una respuesta apoyada en la evidencia…"):
            generated = generate_controlled_answer(standalone_question, result["hits"])
    except Exception as error:
        logging.exception("Error en la generación pública")
        st.session_state.failed_query = {"question": question, "message": failure_message("generación", error)}
        return

    history.append({
        "question": question, "resolved_question": standalone_question,
        "answer": generated["answer"], "hits": result["hits"],
        "retrieval_seconds": result["latency_seconds"],
    })
    st.session_state.selected_evidence = (len(history) - 1, 0) if result["hits"] else None
    st.session_state.pending_clarification = False
    st.session_state.failed_query = None


def render_evidence_dossier() -> None:
    st.markdown('<div class="section-title">Ficha de evidencia</div>', unsafe_allow_html=True)
    selection = st.session_state.selected_evidence
    turns = st.session_state.turns
    if selection is None or selection[0] >= len(turns) or selection[1] >= len(turns[selection[0]]["hits"]):
        with st.container(border=True):
            st.markdown("### Aún no hay una fuente seleccionada")
            st.write("Las referencias aparecerán aquí tras obtener una respuesta. Podrás elegir cualquiera de ellas para examinar el fragmento recuperado.")
        return

    turn_index, hit_index = selection
    hits = turns[turn_index]["hits"]
    if st.session_state.get("evidence_selector") != hit_index:
        st.session_state.evidence_selector = hit_index
    st.radio(
        "Referencias de esta respuesta",
        options=range(len(hits)),
        format_func=lambda index: f"[{index + 1}] {evidence_label(hits[index])}",
        key="evidence_selector",
        on_change=select_evidence_from_dossier,
    )
    hit = hits[hit_index]
    label = evidence_label(hit)
    location = label.split(" · ", 1)[-1] if " · " in label else "Localización no disponible"
    with st.container(border=True):
        st.markdown(f'<div class="dossier-kicker">Referencia [{hit_index + 1}] · Consulta {turn_index + 1}</div>', unsafe_allow_html=True)
        st.markdown(f"### {source_name(hit)}")
        st.write(location)
        st.markdown("**Fragmento localizado**")
        is_spreadsheet = hit.get("sheet") is not None
        render_excerpt(hit.get("text") or "No hay extracto disponible.", spreadsheet=is_spreadsheet)
        parent_text = hit.get("parent_text")
        if parent_text:
            with st.expander("Ver contexto padre"):
                render_excerpt(parent_text, spreadsheet=is_spreadsheet)
        focused_text = hit.get("context_text")
        if focused_text and focused_text != parent_text and focused_text != hit.get("text"):
            with st.expander("Ver ventana de contexto utilizada"):
                render_excerpt(focused_text, spreadsheet=is_spreadsheet)
        with st.expander("Identificadores técnicos"):
            st.code(f"doc_id: {hit.get('doc_id')}\nchunk_id: {hit.get('chunk_id')}\nparent_id: {hit.get('parent_id')}", language="text")
    st.markdown(
        '<div class="source-note">Evidencia recuperada por el sistema. La ficha no atribuye frases concretas de la respuesta a una fuente verificada individualmente.</div>',
        unsafe_allow_html=True,
    )


def render_conversation(api_key: str | None) -> str | None:
    st.markdown('<div class="section-title">Conversación</div>', unsafe_allow_html=True)
    turns = st.session_state.turns
    if not turns:
        st.markdown("### Empieza por una pregunta")
        st.write("Consulta los informes, la normativa y las hojas de cálculo en lenguaje natural. El sistema selecciona los documentos pertinentes.")
        for index, example in enumerate(EXAMPLE_QUESTIONS):
            if st.button(example, key=f"example_{index}", use_container_width=True, disabled=not api_key):
                return example

    for turn_index, turn in enumerate(turns):
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            if turn.get("resolved_question") and turn["resolved_question"] != turn["question"]:
                st.caption(f"Consulta interpretada: {turn['resolved_question']}")
            st.write(turn["answer"])
            if turn["hits"]:
                st.markdown('<div class="source-note">Fuentes recuperadas · selecciona una referencia para inspeccionarla</div>', unsafe_allow_html=True)
                for hit_index, hit in enumerate(turn["hits"]):
                    st.button(
                        f"[{hit_index + 1}] {evidence_label(hit)}",
                        key=f"reference_{turn_index}_{hit_index}",
                        on_click=select_evidence, args=(turn_index, hit_index),
                        type="primary" if st.session_state.selected_evidence == (turn_index, hit_index) else "secondary",
                        use_container_width=True,
                    )
            elif turn.get("clarification"):
                st.caption("Necesito esa precisión antes de buscar evidencia.")

    failed_query = st.session_state.failed_query
    if failed_query:
        st.error(failed_query["message"])
        if st.button("Reintentar la consulta", key="retry_query"):
            return str(failed_query["question"])

    return st.chat_input("Pregunta sobre cualquier documento indexado…", max_chars=600, disabled=not api_key)


def main() -> None:
    st.set_page_config(page_title="Mesa de evidencias · TFG", page_icon="📚", layout="wide")
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)
    for key, default in (
        ("turns", []), ("selected_evidence", None),
        ("pending_clarification", False), ("failed_query", None),
        ("evidence_selector", 0),
    ):
        if key not in st.session_state:
            st.session_state[key] = default

    st.markdown(
        '<div class="masthead"><div class="eyebrow">Trabajo Fin de Grado · Consulta documental</div>'
        '<h1>Mesa de evidencias</h1><p>Pregunta al corpus completo. Cada respuesta conserva las referencias '
        'recuperadas para que puedas revisar el documento, su localización y el contexto original.</p></div>',
        unsafe_allow_html=True,
    )
    files = list_embedding_files()
    if not files:
        st.error("No hay documentos indexados disponibles.")
        return
    try:
        with st.spinner("Abriendo el corpus documental…"):
            rows = load_cached_corpus(tuple(str(path) for path in files))
    except Exception:
        logging.exception("No se pudo cargar el corpus público")
        st.error("No se pudo cargar el corpus indexado. Contacta con el responsable de la demostración.")
        return

    api_key = get_api_key(PROJECT_ROOT)
    info, action = st.columns([5, 1])
    with info:
        st.caption(f"{len(files)} documentos · {len(rows):,} fragmentos indexados · PDF y Excel".replace(",", "."))
    with action:
        st.button("Nueva conversación", key="new_conversation", on_click=reset_conversation, use_container_width=True)
    if not api_key:
        st.warning("La conexión con Llamus no está configurada. Las fichas pueden explorarse, pero no es posible realizar nuevas consultas.")

    conversation, dossier = st.columns([1.8, 1], gap="large")
    with conversation:
        submitted = render_conversation(api_key)
    with dossier:
        render_evidence_dossier()

    st.markdown(
        '<p class="quiet-note">Este chat explora el corpus ampliado; la evaluación académica de 80 preguntas se realizó con condiciones controladas distintas. '
        'Comprueba siempre las fuentes antes de tomar decisiones.</p>',
        unsafe_allow_html=True,
    )
    if submitted and submitted.strip() and api_key:
        run_public_query(submitted.strip(), rows, api_key)
        st.rerun()


if __name__ == "__main__":
    main()
