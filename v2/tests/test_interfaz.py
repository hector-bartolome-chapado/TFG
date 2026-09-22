import pathlib
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generacion.rag_service import (
    ask_llamus,
    build_answer_prompt,
    build_controlled_answer_prompt,
    build_context,
    build_generation_decision,
    classify_question_route,
    extract_legal_answer,
    extract_table_cell_answer,
    generate_controlled_answer,
    list_embedding_files,
    load_document_embeddings,
    run_retrieval,
)
from ingesta.simple_pipeline import write_jsonl
from ingesta.config import DEFAULT_RAG_CHAT_MODEL, RAG_SYSTEM_PROMPT


class InterfazServiceTests(unittest.TestCase):
    def test_config_uses_recommended_model_for_rag_chat(self):
        self.assertEqual(DEFAULT_RAG_CHAT_MODEL, "gemma3:12b")

    def test_list_embedding_files_returns_sorted_jsonl_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = pathlib.Path(tmp_dir)
            (base / "b.jsonl").write_text("", encoding="utf-8")
            (base / "a.jsonl").write_text("", encoding="utf-8")
            (base / "note.txt").write_text("", encoding="utf-8")
            files = list_embedding_files(base)
            self.assertEqual([path.name for path in files], ["a.jsonl", "b.jsonl"])

    def test_load_document_embeddings_reads_existing_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = pathlib.Path(tmp_dir) / "doc.jsonl"
            write_jsonl(path, [{"chunk_id": "c1", "doc_id": "doc", "text": "uno", "embedding": [1.0, 0.0]}])
            rows = load_document_embeddings(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["chunk_id"], "c1")

    def test_build_context_concatenates_hits_with_identifiers(self):
        hits = [
            {"chunk_id": "c1", "score": 0.9, "text": "texto uno"},
            {"chunk_id": "c2", "score": 0.8, "text": "texto dos"},
        ]
        context = build_context(hits)
        self.assertIn("Chunk 1", context)
        self.assertIn("c1", context)
        self.assertIn("texto dos", context)

    def test_build_context_uses_parent_text_when_available(self):
        hits = [
            {
                "chunk_id": "child-1",
                "parent_id": "parent-1",
                "score": 0.9,
                "text": "texto hijo",
                "parent_text": "texto padre completo",
            }
        ]
        context = build_context(hits)
        self.assertIn("parent-1", context)
        self.assertIn("texto padre completo", context)
        self.assertNotIn("\ntexto hijo", context)

    def test_build_context_prefers_focused_context_text(self):
        hits = [
            {
                "chunk_id": "child-1",
                "parent_id": "parent-1",
                "score": 0.9,
                "text": "texto hijo",
                "parent_text": "texto padre completo demasiado largo",
                "context_text": "ventana enfocada",
            }
        ]
        context = build_context(hits)
        self.assertIn("ventana enfocada", context)
        self.assertNotIn("texto padre completo demasiado largo", context)

    def test_build_answer_prompt_anchors_to_context(self):
        messages = build_answer_prompt("¿Qué dice?", "texto uno")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], RAG_SYSTEM_PROMPT)
        self.assertIn("Pregunta:", messages[1]["content"])
        self.assertIn("texto uno", messages[1]["content"])

    def test_classify_question_route_detects_generation_modes(self):
        self.assertEqual(classify_question_route("?Cu?nto creci? el IRPF en 2024?"), "tax_exact")
        self.assertEqual(classify_question_route("¿Qué indica la Ley 37/1992 sobre el ámbito de aplicación del IVA?"), "legal")
        self.assertEqual(classify_question_route("?Qu? cuadro recoge el IVA?"), "table")
        self.assertEqual(classify_question_route("Cual fue el total presupuesto en 2025?"), "table")
        self.assertEqual(classify_question_route("?C?mo evolucionaron los precios?"), "chart")
        self.assertEqual(classify_question_route("?Por qu? subieron los ingresos?"), "synthesis")

    def test_build_controlled_answer_prompt_adds_route_specific_rules(self):
        messages = build_controlled_answer_prompt("?Cu?nto creci? el IRPF?", "IRPF 7,6%", route="tax_exact")
        system_text = messages[0]["content"]
        self.assertIn("modo extractor", system_text.lower())
        self.assertIn("No hay informaci", system_text)
        self.assertIn("IRPF 7,6%", messages[1]["content"])

    def test_build_controlled_answer_prompt_adds_compact_output_contracts(self):
        route_expectations = {
            "tax_exact": ("Dato principal:", "Fuente:"),
            "table": ("Cuadro o tabla:", "Dato extraido:"),
            "chart": ("Tendencia textual:", "No estimes"),
            "synthesis": ("Evidencias combinadas:", "Maximo 3 frases"),
            "legal": ("Modo juridico:", "No te limites a repetir el titulo"),
            "simple_fact": ("Respuesta breve:", "una frase"),
        }
        for route, expected_terms in route_expectations.items():
            with self.subTest(route=route):
                messages = build_controlled_answer_prompt("?Pregunta?", "Contexto", route=route)
                system_text = messages[0]["content"]
                for term in expected_terms:
                    self.assertIn(term, system_text)

    def test_generation_decision_rejects_low_evidence_context(self):
        hits = [
            {
                "chunk_id": "c1",
                "score": 0.01,
                "text": "Informe anual de recaudaci?n tributaria.",
            }
        ]
        decision = build_generation_decision("?Qu? empresa privada audit? los embeddings fiscales?", hits)
        self.assertEqual(decision["route"], "simple_fact")
        self.assertTrue(decision["should_reject"])
        self.assertIn("No hay informaci", decision["answer"])

    def test_generate_controlled_answer_extracts_relevant_sentence(self):
        hits = [
            {
                "chunk_id": "c1",
                "score": 0.8,
                "text": "Los ingresos por el IRPF crecieron en 2024 un 7,6%. El IVA creci? un 7,9%.",
            }
        ]
        result = generate_controlled_answer("?Cu?nto crecieron los ingresos por IRPF en 2024?", hits)
        self.assertFalse(result["decision"]["should_reject"])
        self.assertIn("7,6", result["answer"])
        self.assertIn("IRPF", result["answer"])

    def test_generate_controlled_answer_restates_question_subject_for_tables(self):
        hits = [
            {
                "chunk_id": "c1",
                "score": 0.8,
                "text": "NOTA INFORMATIVA 7: ELABORACIÓN DE UN DEFLACTOR PARA LA RECAUDACIÓN DEL IVA.",
            }
        ]
        result = generate_controlled_answer("¿Qué nota informativa trata del deflactor para la recaudación del IVA?", hits)
        self.assertFalse(result["decision"]["should_reject"])
        self.assertIn("deflactor", result["answer"].lower())
        self.assertIn("NOTA INFORMATIVA 7", result["answer"])

    def test_extract_table_cell_answer_uses_header_and_requested_row(self):
        context = (
            "Documento: Copia de 01 Presupuestos Generales del Estado Consolidados.xlsx\n"
            "Hoja: 11\n"
            "Rango de filas: 4-20\n"
            "Fila 6: Capitulos | 2023 | 2024-P | 2025-P\n"
            "Fila 10: Transferencias corrientes | 329602.88314 | 328794.73672 | 328794.73672\n"
            "Fila 20: TOTAL PRESUPUESTO | 583543.3071 | 578183.02509 | 578183.02509\n"
        )
        answer = extract_table_cell_answer("Cual fue el total presupuesto en 2025?", context)
        self.assertIsNotNone(answer)
        self.assertIn("TOTAL PRESUPUESTO en 2025-P", answer)
        self.assertIn("578.183,03", answer)
        self.assertIn("fila 20", answer)

    def test_extract_table_cell_answer_infers_pge_header_when_chunk_lacks_header(self):
        context = (
            "[Chunk 1 | Copia_de_01_Presupuestos_Generales_del_Estado_Consolidados::sheet02::rows16-24::b3 | score=0.9500]\n"
            "Documento: Copia de 01 Presupuestos Generales del Estado Consolidados.xlsx\n"
            "Hoja: 11\n"
            "Rango de filas: 16-24\n"
            "Fila 20: TOTAL PRESUPUESTO | 443133.3295 | 449785.08397 | 449738.82646 | 449738.82646 | "
            "550483.87477 | 527107.76322 | 583543.3071 | 578183.02509 | 578183.02509 | 578183.0250900001\n"
        )
        answer = extract_table_cell_answer("Cual fue el total presupuesto en 2025?", context)
        self.assertIsNotNone(answer)
        self.assertIn("TOTAL PRESUPUESTO en 2025-P", answer)
        self.assertIn("578.183,03", answer)

    def test_generate_controlled_answer_extracts_excel_cell_before_sentence_selection(self):
        hits = [
            {
                "chunk_id": "xlsx-1",
                "score": 0.8,
                "text": (
                    "Documento: Copia de 01 Presupuestos Generales del Estado Consolidados.xlsx\n"
                    "Hoja: 11\n"
                    "Rango de filas: 4-20\n"
                    "Fila 6: Capitulos | 2023 | 2024-P | 2025-P\n"
                    "Fila 7: Gastos de personal | 27481.71541 | 27473.45093 | 27476.45816\n"
                    "Fila 20: TOTAL PRESUPUESTO | 583543.3071 | 578183.02509 | 578183.02509\n"
                ),
            }
        ]
        result = generate_controlled_answer("Cual fue el total presupuesto en 2025?", hits)
        self.assertFalse(result["decision"]["should_reject"])
        self.assertIn("Dato principal:", result["answer"])
        self.assertIn("578.183,03", result["answer"])

    def test_extract_legal_answer_explains_article_content(self):
        context = (
            "TITULO PRELIMINAR Naturaleza y ambito de aplicacion "
            "Articulo 1. Naturaleza del impuesto. El Impuesto sobre el Valor Anadido "
            "es un tributo de naturaleza indirecta que recae sobre el consumo y grava "
            "las entregas de bienes y prestaciones de servicios efectuadas por empresarios o profesionales. "
            "El ambito espacial de aplicacion del impuesto es el territorio espanol."
        )
        answer = extract_legal_answer(
            "Que indica la Ley 37/1992 sobre el ambito de aplicacion del IVA?",
            context,
        )
        self.assertIsNotNone(answer)
        self.assertIn("Norma: Articulo 1", answer)
        self.assertIn("tributo de naturaleza indirecta", answer)
        self.assertIn("territorio espanol", answer)

    def test_generate_controlled_answer_restates_no_answer_topic(self):
        hits = [
            {
                "chunk_id": "c1",
                "score": 0.01,
                "text": "Informe anual de recaudaci?n tributaria.",
            }
        ]
        result = generate_controlled_answer("¿Qué empresa privada auditó los embeddings fiscales?", hits)
        self.assertTrue(result["decision"]["should_reject"])
        self.assertIn("embeddings fiscales", result["answer"])

    def test_run_retrieval_returns_hits_and_latency(self):
        rows = [{"chunk_id": "c1", "doc_id": "doc", "text": "texto", "embedding": [1.0, 0.0]}]

        with mock.patch("generacion.rag_service.retrieve_top_k", return_value=[{"chunk_id": "c1", "score": 0.9, "text": "texto"}]) as retrieve_mock:
            result = run_retrieval(
                "pregunta",
                rows,
                top_k=1,
                api_key="secret",
                strategy="fiscal_hybrid",
            )
        self.assertEqual(result["hits"][0]["chunk_id"], "c1")
        self.assertGreaterEqual(result["latency_seconds"], 0.0)
        self.assertEqual(retrieve_mock.call_args.kwargs["strategy"], "fiscal_hybrid")

    def test_run_retrieval_forwards_custom_embedder(self):
        rows = [{"chunk_id": "c1", "doc_id": "doc", "text": "texto", "embedding": [1.0, 0.0]}]
        embedder = lambda text, model, base_url, api_key: [1.0, 0.0]

        with mock.patch("generacion.rag_service.retrieve_top_k", return_value=[]) as retrieve_mock:
            run_retrieval("pregunta", rows, top_k=1, embedder=embedder)

        self.assertIs(retrieve_mock.call_args.kwargs["embedder"], embedder)

    def test_ask_llamus_reads_openai_style_response(self):
        class FakeResponse:
            def __init__(self):
                self.status_code = 200
                self.text = "{}"

            def json(self):
                return {"choices": [{"message": {"content": "respuesta final"}}]}

        with mock.patch("generacion.rag_service.requests.post", return_value=FakeResponse()):
            result = ask_llamus(
                prompt_messages=[{"role": "system", "content": RAG_SYSTEM_PROMPT}],
                api_key="secret",
            )
        self.assertEqual(result["answer"], "respuesta final")
        self.assertGreaterEqual(result["latency_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
