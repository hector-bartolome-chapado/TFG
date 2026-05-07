from __future__ import annotations

import pathlib
import sys

import streamlit as st

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.simple_extractor.config import DEFAULT_CHAT_MODEL, DEFAULT_LLAMUS_BASE_URL, get_api_key
from interfaz.rag_service import (
    build_answer_prompt,
    build_context,
    list_embedding_files,
    load_document_embeddings,
    run_retrieval,
    ask_llamus,
)


# Configura la página y el estilo base del laboratorio RAG.
#
# Entra:
# - nada.
# Sale:
# - inicializa la app `Streamlit`.
# Por qué existe:
# - concentra la configuración visual y evita mezclarla con la lógica del laboratorio.
def configure_page() -> None:
    st.set_page_config(page_title="Laboratorio RAG", layout="wide")
    st.title("Laboratorio RAG v1")
    st.caption("Pregunta → retrieval → contexto → respuesta final")


# Dibuja la barra lateral con documento, top_k y configuración básica.
#
# Entra:
# - embedding_files: lista de documentos disponibles.
# Sale:
# - un diccionario con la selección del usuario.
# Por qué existe:
# - la barra lateral fija el estado del laboratorio sin cargar el panel principal.
def render_sidebar(embedding_files: list[pathlib.Path]) -> dict[str, object]:
    st.sidebar.header("Configuración")
    if st.sidebar.button("Refrescar documentos"):
        st.rerun()

    selected_path = None
    if embedding_files:
        selected_name = st.sidebar.selectbox(
            "Documento",
            [path.name for path in embedding_files],
            index=0,
        )
        selected_path = next(path for path in embedding_files if path.name == selected_name)
    else:
        st.sidebar.warning("No hay embeddings en RESULTADOS EMBEDDING/embeddings.")

    top_k = st.sidebar.slider("top_k", min_value=1, max_value=10, value=5)
    model = st.sidebar.text_input("Modelo generador", value=DEFAULT_CHAT_MODEL)
    base_url = st.sidebar.text_input("Servidor llamus", value=DEFAULT_LLAMUS_BASE_URL)
    return {
        "selected_path": selected_path,
        "top_k": top_k,
        "model": model,
        "base_url": base_url,
    }


# Muestra cada hit recuperado de forma legible en el panel de retrieval.
#
# Entra:
# - hits: resultados de retrieval ordenados por score.
# Sale:
# - pinta expanders con score y contenido.
# Por qué existe:
# - retrieval es el primer punto de fallo del RAG y debe quedar visible por separado.
def render_hits(hits: list[dict[str, object]]) -> None:
    st.subheader("Retrieval")
    for index, hit in enumerate(hits, start=1):
        title = f"{index}. {hit['chunk_id']} · score={hit['score']:.4f}"
        with st.expander(title, expanded=index == 1):
            st.code(hit["text"], language="text")


# Muestra el contexto exacto que se va a enviar al modelo.
#
# Entra:
# - context: string consolidado con todos los hits.
# Sale:
# - renderiza el bloque de contexto.
# Por qué existe:
# - separa errores de retrieval de errores de generación.
def render_context(context: str) -> None:
    st.subheader("Contexto enviado")
    st.code(context, language="text")


# Muestra la respuesta final y un bloque breve de depuración.
#
# Entra:
# - answer: texto generado.
# - retrieval_latency: tiempo de retrieval.
# - generation_latency: tiempo de generación.
# - prompt_messages: mensajes enviados al modelo.
# - doc_name: documento seleccionado.
# - row_count: número total de chunks cargados.
# Sale:
# - renderiza respuesta y debug.
# Por qué existe:
# - deja trazabilidad suficiente sin convertir la interfaz en una consola.
def render_answer_and_debug(
    answer: str,
    retrieval_latency: float,
    generation_latency: float,
    prompt_messages: list[dict[str, str]],
    doc_name: str,
    row_count: int,
) -> None:
    st.subheader("Respuesta final")
    st.write(answer)

    st.subheader("Debug")
    left, right = st.columns(2)
    left.metric("Latencia retrieval", f"{retrieval_latency:.2f}s")
    right.metric("Latencia generación", f"{generation_latency:.2f}s")
    st.write(f"Documento activo: `{doc_name}`")
    st.write(f"Chunks cargados: `{row_count}`")
    with st.expander("Prompt enviado", expanded=False):
        st.json(prompt_messages)


# Orquesta la pantalla principal del laboratorio.
#
# Entra:
# - nada; usa la selección y la pregunta del usuario.
# Sale:
# - ejecuta retrieval y generación y pinta el resultado.
# Por qué existe:
# - mantiene el flujo principal en un sitio fácil de seguir.
def main() -> None:
    configure_page()
    embedding_files = list_embedding_files()
    sidebar_state = render_sidebar(embedding_files)

    selected_path = sidebar_state["selected_path"]
    if selected_path is None:
        st.info("Genera primero un `.jsonl` de embeddings para usar la interfaz.")
        return

    rows = load_document_embeddings(selected_path)
    api_key = get_api_key(PROJECT_ROOT)
    if not api_key:
        st.error("No hay API key. Usa `TFG/.llamus_api_key` o `LLAMUS_API_KEY`.")
        return

    question = st.text_area("Pregunta", placeholder="Escribe aquí una pregunta sobre el documento...")
    if not st.button("Buscar y responder", type="primary"):
        return

    if not question.strip():
        st.warning("Escribe una pregunta antes de lanzar el laboratorio.")
        return

    try:
        retrieval_result = run_retrieval(
            question=question,
            rows=rows,
            top_k=int(sidebar_state["top_k"]),
            base_url=str(sidebar_state["base_url"]),
            api_key=api_key,
        )
        hits = retrieval_result["hits"]
        render_hits(hits)

        context = build_context(hits)
        render_context(context)

        prompt_messages = build_answer_prompt(question, context)
        generation_result = ask_llamus(
            prompt_messages=prompt_messages,
            model=str(sidebar_state["model"]),
            base_url=str(sidebar_state["base_url"]),
            api_key=api_key,
        )
        render_answer_and_debug(
            answer=generation_result["answer"],
            retrieval_latency=float(retrieval_result["latency_seconds"]),
            generation_latency=float(generation_result["latency_seconds"]),
            prompt_messages=prompt_messages,
            doc_name=selected_path.name,
            row_count=len(rows),
        )
    except Exception as exc:
        st.exception(exc)


if __name__ == "__main__":
    main()
