from __future__ import annotations

import logging
import pathlib
import sys

import streamlit as st

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generacion.rag_service import (
    build_context,
    generate_controlled_answer,
    list_embedding_files,
    load_document_embeddings,
    run_retrieval,
)
from ingesta.config import DEFAULT_LLAMUS_BASE_URL, get_api_key


@st.cache_data(show_spinner=False, max_entries=2)
def load_rows(path: str) -> list[dict[str, object]]:
    return load_document_embeddings(pathlib.Path(path))


def main() -> None:
    st.set_page_config(page_title="RAG documental · TFG", layout="wide")
    st.title("Consulta documental con IA")
    st.caption("Proyecto de TFG · Respuestas basadas en documentación pública con evidencia consultable")

    files = list_embedding_files()
    if not files:
        st.error("No hay documentos indexados disponibles.")
        return

    api_key = get_api_key(PROJECT_ROOT)
    if not api_key:
        st.error("El servicio de consulta no está configurado. Contacta con el responsable de la demostración.")
        return

    with st.sidebar:
        st.header("Documento")
        selected_path = st.selectbox("Fuente documental", files, format_func=lambda path: path.stem)
        st.caption("El sistema consulta el documento seleccionado y muestra los fragmentos recuperados.")

    question = st.text_area("Tu pregunta", max_chars=600, placeholder="¿Qué indica el documento sobre...?")
    if not st.button("Buscar respuesta", type="primary"):
        return
    if not question.strip():
        st.warning("Escribe una pregunta para comenzar.")
        return

    try:
        with st.spinner("Buscando evidencia en el documento..."):
            rows = load_rows(str(selected_path))
            result = run_retrieval(
                question=question.strip(),
                rows=rows,
                top_k=3,
                base_url=DEFAULT_LLAMUS_BASE_URL,
                api_key=api_key,
            )
            hits = result["hits"]
            answer = generate_controlled_answer(question.strip(), hits)["answer"]
            context = build_context(hits)
    except Exception:
        logging.exception("Error en la consulta pública del RAG")
        st.error("No se pudo completar la consulta. Inténtalo de nuevo más tarde.")
        return

    st.subheader("Respuesta")
    st.write(answer)
    st.caption("Comprueba siempre la evidencia antes de utilizar la respuesta en una decisión real.")

    with st.expander("Ver evidencia recuperada"):
        if not hits:
            st.write("No se recuperaron fragmentos.")
        for index, hit in enumerate(hits, start=1):
            page = hit.get("page_start")
            source = f" · página {page}" if page is not None else ""
            st.markdown(f"**{index}. {hit['doc_id']}{source}**")
            st.write(hit.get("context_text") or hit.get("parent_text") or hit["text"])

    with st.expander("Ver contexto completo usado para la respuesta"):
        st.text(context)


if __name__ == "__main__":
    main()
