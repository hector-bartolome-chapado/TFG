import pathlib
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingesta.simple_pipeline import write_jsonl
from interfaz.public_app import failure_message
from interfaz.public_service import evidence_label, load_corpus, resolve_question
from recuperacion.retrieval import rank_chunks_fiscal_hybrid


class PublicServiceTests(unittest.TestCase):
    def test_load_corpus_combines_documents_with_one_embedding_space(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [pathlib.Path(directory) / name for name in ("aeat.jsonl", "pge.jsonl")]
            for index, path in enumerate(paths):
                write_jsonl(path, [{
                    "chunk_id": f"chunk-{index}", "doc_id": path.stem,
                    "text": "contenido", "model": "qwen3-embedding:4b",
                    "embedding": [0.5, 0.5],
                }])
            rows = load_corpus(paths, expected_model="qwen3-embedding:4b", expected_dimensions=2)
        self.assertEqual({row["doc_id"] for row in rows}, {"aeat", "pge"})

    def test_load_corpus_rejects_mixed_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "bad.jsonl"
            write_jsonl(path, [{
                "chunk_id": "bad", "doc_id": "bad", "text": "contenido",
                "model": "qwen3-embedding:4b", "embedding": [1.0],
            }])
            with self.assertRaisesRegex(ValueError, "dimensi"):
                load_corpus([path], expected_model="qwen3-embedding:4b", expected_dimensions=2)

    def test_excel_location_survives_retrieval(self):
        row = {
            "chunk_id": "c1", "doc_id": "pge", "text": "TOTAL PRESUPUESTO 2025",
            "retrieval_text": "TOTAL PRESUPUESTO 2025", "parent_id": "p1",
            "parent_text": "Fila 20: TOTAL PRESUPUESTO 2025-P 578183",
            "chunking_strategy": "xlsx_rows_parent_child", "embedding": [1.0, 0.0],
            "source_file": r"..\documentos\Presupuestos.xlsx", "sheet": "Estadística",
            "sheet_index": 1, "row_start": 20, "row_end": 21,
        }
        hit = rank_chunks_fiscal_hybrid("total presupuesto 2025", [1.0, 0.0], [row])[0]
        self.assertEqual(hit["source_file"], row["source_file"])
        self.assertEqual(hit["sheet"], "Estadística")
        self.assertEqual(hit["row_start"], 20)
        self.assertIn("filas 20–21", evidence_label(hit))

    def test_pdf_evidence_label_uses_page(self):
        hit = {"doc_id": "AEAT_informe_anual_2024", "page_start": 9, "page_end": 9}
        self.assertIn("página 9", evidence_label(hit))

    def test_independent_question_does_not_call_chat_model(self):
        client = mock.Mock()
        result = resolve_question("¿Cuáles fueron los ingresos tributarios en 2024?", [
            {"question": "¿Qué es el IVA?", "answer": "Un impuesto."},
        ], api_key="secret", chat_client=client)
        self.assertEqual(result["question"], "¿Cuáles fueron los ingresos tributarios en 2024?")
        client.assert_not_called()

    def test_followup_is_rewritten_with_recent_history(self):
        client = mock.Mock(return_value={"answer": '{"standalone_question":"¿Cuánto recaudó el IRPF en 2023?","needs_clarification":false,"clarification_question":""}'})
        result = resolve_question("¿Y en 2023?", [
            {"question": "¿Cuánto recaudó el IRPF en 2024?", "answer": "123 millones."},
        ], api_key="secret", chat_client=client)
        self.assertEqual(result["question"], "¿Cuánto recaudó el IRPF en 2023?")
        self.assertIn("IRPF", client.call_args.args[0][1]["content"])

    def test_ambiguous_followup_requests_clarification(self):
        client = mock.Mock(return_value={"answer": '{"standalone_question":"","needs_clarification":true,"clarification_question":"¿A qué impuesto te refieres?"}'})
        result = resolve_question("¿Y ese?", [
            {"question": "¿Qué son el IVA y el IRPF?", "answer": "Dos impuestos."},
        ], api_key="secret", chat_client=client)
        self.assertEqual(result["clarification"], "¿A qué impuesto te refieres?")

    def test_interface_reference_selection_and_new_conversation(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PROJECT_ROOT / "interfaz" / "public_app.py"), default_timeout=120).run()
        app.session_state.turns = [{
            "question": "Prueba", "answer": "Respuesta", "hits": [
                {"chunk_id": "a", "doc_id": "aeat", "text": "Informe", "page_start": 9},
                {"chunk_id": "b", "doc_id": "pge", "text": "Fila 20", "source_file": "Presupuesto.xlsx", "sheet": "Estadística", "row_start": 20},
            ],
        }]
        app.session_state.selected_evidence = (0, 0)
        app.run()
        self.assertFalse(app.exception)
        self.assertIn("hoja Estadística", str(app))
        app.button(key="reference_0_1").click().run()
        self.assertEqual(app.session_state.selected_evidence, (0, 1))
        app.radio(key="evidence_selector").set_value(0).run()
        self.assertEqual(app.session_state.selected_evidence, (0, 0))
        app.button(key="new_conversation").click().run()
        self.assertEqual(app.session_state.turns, [])
        self.assertIsNone(app.session_state.selected_evidence)

    def test_failure_message_identifies_phase_without_server_details(self):
        import requests

        message = failure_message("recuperación", requests.exceptions.Timeout("private-url"))
        self.assertIn("recuperación", message)
        self.assertIn("reintentar", message)
        self.assertNotIn("private-url", message)


if __name__ == "__main__":
    unittest.main()
