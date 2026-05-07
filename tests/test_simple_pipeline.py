import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

import fitz


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.simple_pipeline import process_document, write_jsonl
from scripts.simple_extractor.blocks import split_into_paragraphs
from scripts.simple_extractor.chunks import build_chunks, split_long_text
from scripts.simple_extractor.embeddings import embed_chunks, request_embedding
from scripts.RECUPERADOR.retrieval import cosine_similarity, load_embedding_rows, retrieve_top_k

PROJECTOR_EXPORT_PATH = PROJECT_ROOT / "VISUAL DE EMBEDDING" / "export_to_projector.py"
projector_spec = importlib.util.spec_from_file_location("projector_export", PROJECTOR_EXPORT_PATH)
projector_module = importlib.util.module_from_spec(projector_spec)
assert projector_spec is not None and projector_spec.loader is not None
projector_spec.loader.exec_module(projector_module)

build_embedding_matrix = projector_module.build_embedding_matrix
build_metadata_rows = projector_module.build_metadata_rows
build_page_label = projector_module.build_page_label
export_projector_files = projector_module.export_projector_files


class SimplePipelineTests(unittest.TestCase):
    def test_split_into_paragraphs_separates_on_blank_lines(self):
        page_text = "Primer párrafo.\n\nSegundo párrafo.\n\nTercer párrafo."
        self.assertEqual(
            split_into_paragraphs(page_text),
            ["Primer párrafo.", "Segundo párrafo.", "Tercer párrafo."],
        )

    def test_build_chunks_respects_max_chars(self):
        blocks = [
            {"block_id": "doc::p1::b1", "doc_id": "doc", "page": 1, "text": "A" * 30},
            {"block_id": "doc::p1::b2", "doc_id": "doc", "page": 1, "text": "B" * 30},
            {"block_id": "doc::p2::b1", "doc_id": "doc", "page": 2, "text": "C" * 30},
        ]
        chunks = build_chunks(blocks, max_chars=70)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["page_start"], 1)
        self.assertEqual(chunks[0]["page_end"], 1)
        self.assertEqual(chunks[1]["page_start"], 2)

    def test_split_long_text_breaks_oversized_block(self):
        text = "Frase uno. " + ("A" * 120) + " Fin."
        parts = split_long_text(text, max_chars=60)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(part) <= 60 for part in parts))

    def test_write_jsonl_serializes_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = pathlib.Path(tmp_dir) / "rows.jsonl"
            write_jsonl(path, [{"a": 1}, {"b": 2}])
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(lines[0]), {"a": 1})
            self.assertEqual(json.loads(lines[1]), {"b": 2})

    def test_embed_chunks_uses_requestor(self):
        chunks = [
            {"chunk_id": "chunk-1", "doc_id": "doc", "page_start": 1, "page_end": 1, "text": "Texto 1"},
            {"chunk_id": "chunk-2", "doc_id": "doc", "page_start": 2, "page_end": 3, "text": "Texto 2"},
        ]

        def fake_requestor(text, model, base_url, api_key):
            self.assertEqual(model, "mxbai-embed-large:v1")
            self.assertEqual(base_url, "https://llamus.cs.us.es")
            self.assertEqual(api_key, "secret")
            return [float(len(text))]

        embedded = embed_chunks(chunks, api_key="secret", requestor=fake_requestor)
        self.assertEqual(embedded[0]["embedding"], [7.0])
        self.assertEqual(embedded[1]["embedding"], [7.0])
        self.assertEqual(embedded[0]["page_start"], 1)
        self.assertEqual(embedded[1]["page_end"], 3)

    def test_request_embedding_accepts_embedding_or_embeddings_response(self):
        class FakeResponse:
            def __init__(self, payload, status_code=200):
                self.payload = payload
                self.status_code = status_code
                self.text = json.dumps(payload)

            def json(self):
                return self.payload

        from unittest import mock

        with mock.patch(
            "scripts.simple_extractor.embeddings.requests.post",
            return_value=FakeResponse({"embedding": [0.1, 0.2]}),
        ):
            self.assertEqual(request_embedding("hola"), [0.1, 0.2])

        with mock.patch(
            "scripts.simple_extractor.embeddings.requests.post",
            return_value=FakeResponse({"embeddings": [[0.3, 0.4]]}),
        ):
            self.assertEqual(request_embedding("hola"), [0.3, 0.4])

    def test_process_document_writes_blocks_and_chunks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            pdf_path = tmp_path / "sample.pdf"

            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), "Primer párrafo.\n\nSegundo párrafo.")
            doc.save(pdf_path)
            doc.close()

            result = process_document(pdf_path=pdf_path, output_root=tmp_path / "out", max_chars=100)

            self.assertEqual(result["page_count"], 1)
            self.assertGreaterEqual(result["block_count"], 1)
            self.assertGreaterEqual(result["chunk_count"], 1)
            self.assertTrue(pathlib.Path(result["blocks_path"]).exists())
            self.assertTrue(pathlib.Path(result["chunks_path"]).exists())
            self.assertNotIn("markdown_path", result)

    def test_load_embedding_rows_reads_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = pathlib.Path(tmp_dir) / "embeddings.jsonl"
            write_jsonl(
                path,
                [
                    {"chunk_id": "c1", "doc_id": "doc", "text": "uno", "embedding": [1.0, 0.0]},
                    {"chunk_id": "c2", "doc_id": "doc", "text": "dos", "embedding": [0.0, 1.0]},
                ],
            )
            rows = load_embedding_rows(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["chunk_id"], "c1")

    def test_cosine_similarity_detects_closer_vector(self):
        near = cosine_similarity([1.0, 0.0], [0.9, 0.1])
        far = cosine_similarity([1.0, 0.0], [0.0, 1.0])
        self.assertGreater(near, far)

    def test_retrieve_top_k_orders_chunks_by_similarity(self):
        rows = [
            {"chunk_id": "c1", "doc_id": "doc", "text": "IRPF", "embedding": [1.0, 0.0]},
            {"chunk_id": "c2", "doc_id": "doc", "text": "IVA", "embedding": [0.0, 1.0]},
            {"chunk_id": "c3", "doc_id": "doc", "text": "Sociedades", "embedding": [0.7, 0.2]},
        ]

        def fake_embedder(text, model, base_url, api_key):
            self.assertEqual(text, "pregunta sobre IRPF")
            return [1.0, 0.0]

        ranked = retrieve_top_k(
            question="pregunta sobre IRPF",
            embedding_rows=rows,
            top_k=2,
            embedder=fake_embedder,
        )
        self.assertEqual([row["chunk_id"] for row in ranked], ["c1", "c3"])
        self.assertGreaterEqual(ranked[0]["score"], ranked[1]["score"])

    def test_build_embedding_matrix_returns_one_row_per_chunk(self):
        rows = [
            {"chunk_id": "c1", "doc_id": "doc", "text": "uno", "embedding": [1.0, 0.0, 0.0]},
            {"chunk_id": "c2", "doc_id": "doc", "text": "dos", "embedding": [0.0, 1.0, 0.0]},
            {"chunk_id": "c3", "doc_id": "doc", "text": "tres", "embedding": [0.0, 0.0, 1.0]},
        ]
        matrix = build_embedding_matrix(rows)
        self.assertEqual(matrix.shape, (3, 3))

    def test_build_metadata_rows_preserves_page_and_preview(self):
        rows = [
            {"chunk_id": "c1", "doc_id": "doc", "page_start": 1, "page_end": 1, "text": "1. Recaudación total\nuno dos tres", "embedding": [1.0, 0.0]},
            {"chunk_id": "c2", "doc_id": "doc", "page_start": 2, "page_end": 4, "text": "2024 2023 2022 10 20 30", "embedding": [0.0, 1.0]},
        ]
        metadata_rows = build_metadata_rows(rows)
        self.assertEqual(len(metadata_rows), 2)
        self.assertEqual(metadata_rows[0]["page"], "1")
        self.assertIn("preview_corto", metadata_rows[0])
        self.assertEqual(set(metadata_rows[0].keys()), {"page", "preview_corto"})
        self.assertEqual(metadata_rows[1]["page"], "2")

    def test_build_page_label_uses_only_page_fields(self):
        self.assertEqual(build_page_label({"page_start": 3, "page_end": 3}), "3")
        self.assertEqual(build_page_label({"page_start": 3, "page_end": 5}), "3")
        self.assertEqual(build_page_label({}), "")

    def test_export_projector_files_writes_metadata_config_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            embeddings_path = pathlib.Path(tmp_dir) / "sample.jsonl"
            output_dir = pathlib.Path(tmp_dir) / "projector"
            write_jsonl(
                embeddings_path,
                [
                    {"chunk_id": "c1", "doc_id": "doc", "text": "uno", "embedding": [1.0, 0.0]},
                    {"chunk_id": "c2", "doc_id": "doc", "text": "dos", "embedding": [0.0, 1.0]},
                ],
            )
            result = export_projector_files(embeddings_path=embeddings_path, output_dir=output_dir)
            self.assertTrue(pathlib.Path(result["metadata_path"]).exists())
            self.assertTrue(pathlib.Path(result["config_path"]).exists())
            self.assertTrue(list(output_dir.glob("embedding.ckpt-*")))
            self.assertTrue(list(output_dir.glob("events.out.tfevents.*")))


if __name__ == "__main__":
    unittest.main()
