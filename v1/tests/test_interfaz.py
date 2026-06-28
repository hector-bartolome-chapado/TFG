import pathlib
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interfaz.rag_service import (
    ask_llamus,
    build_answer_prompt,
    build_context,
    list_embedding_files,
    load_document_embeddings,
    run_retrieval,
)
from scripts.simple_pipeline import write_jsonl
from scripts.simple_extractor.config import RAG_SYSTEM_PROMPT


class InterfazServiceTests(unittest.TestCase):
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

    def test_build_answer_prompt_anchors_to_context(self):
        messages = build_answer_prompt("¿Qué dice?", "texto uno")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], RAG_SYSTEM_PROMPT)
        self.assertIn("Pregunta:", messages[1]["content"])
        self.assertIn("texto uno", messages[1]["content"])

    def test_run_retrieval_returns_hits_and_latency(self):
        rows = [{"chunk_id": "c1", "doc_id": "doc", "text": "texto", "embedding": [1.0, 0.0]}]

        with mock.patch("interfaz.rag_service.retrieve_top_k", return_value=[{"chunk_id": "c1", "score": 0.9, "text": "texto"}]):
            result = run_retrieval("pregunta", rows, top_k=1, api_key="secret")
        self.assertEqual(result["hits"][0]["chunk_id"], "c1")
        self.assertGreaterEqual(result["latency_seconds"], 0.0)

    def test_ask_llamus_reads_openai_style_response(self):
        class FakeResponse:
            def __init__(self):
                self.status_code = 200
                self.text = "{}"

            def json(self):
                return {"choices": [{"message": {"content": "respuesta final"}}]}

        with mock.patch("interfaz.rag_service.requests.post", return_value=FakeResponse()):
            result = ask_llamus(
                prompt_messages=[{"role": "system", "content": RAG_SYSTEM_PROMPT}],
                api_key="secret",
            )
        self.assertEqual(result["answer"], "respuesta final")
        self.assertGreaterEqual(result["latency_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
