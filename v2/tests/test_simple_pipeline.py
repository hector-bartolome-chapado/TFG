import json
import pathlib
import sys
import tempfile
import unittest

import fitz
import requests


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recuperacion.retrieval import (
    bm25_scores,
    build_focused_context,
    cosine_similarity,
    expand_fiscal_query,
    is_legal_query,
    legal_relevance_score,
    load_embedding_rows,
    rank_chunks_fiscal_lexical,
    reciprocal_rank_fusion,
    retrieve_top_k,
    retrieve_top_k_with_fallback,
    table_relevance_score,
    tokenize_for_bm25,
)
from ingesta.blocks import split_into_paragraphs
from ingesta.chunks import build_chunks, split_long_text, split_text_by_tokens
from ingesta.embeddings import embed_chunks, request_embedding
from ingesta.simple_pipeline import process_document, write_jsonl


class SimplePipelineTests(unittest.TestCase):
    def test_split_into_paragraphs_separates_on_blank_lines(self):
        page_text = "Primer parrafo.\n\nSegundo parrafo.\n\nTercer parrafo."
        self.assertEqual(
            split_into_paragraphs(page_text),
            ["Primer parrafo.", "Segundo parrafo.", "Tercer parrafo."],
        )

    def test_build_chunks_creates_parent_child_metadata(self):
        blocks = [
            {
                "block_id": "doc::p1::b1",
                "doc_id": "doc",
                "page": 1,
                "text": "uno dos tres cuatro cinco seis siete ocho nueve diez",
            }
        ]
        chunks = build_chunks(blocks, max_child_tokens=4, overlap_tokens=1)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["parent_id"], "doc::p1::b1::parent")
        self.assertEqual(chunks[0]["parent_text"], blocks[0]["text"])
        self.assertEqual(chunks[1]["text"], "cuatro cinco seis siete")

    def test_split_long_text_breaks_oversized_block(self):
        text = "Frase uno. " + ("A" * 120) + " Fin."
        parts = split_long_text(text, max_chars=60)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(part) <= 60 for part in parts))

    def test_split_text_by_tokens_uses_overlap(self):
        text = "uno dos tres cuatro cinco seis siete ocho nueve diez"
        parts = split_text_by_tokens(text, max_tokens=4, overlap_tokens=1)
        self.assertEqual(parts, ["uno dos tres cuatro", "cuatro cinco seis siete", "siete ocho nueve diez"])

    def test_write_jsonl_serializes_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = pathlib.Path(tmp_dir) / "rows.jsonl"
            write_jsonl(path, [{"a": 1}, {"b": 2}])
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(lines[0]), {"a": 1})
            self.assertEqual(json.loads(lines[1]), {"b": 2})

    def test_embed_chunks_uses_requestor_and_preserves_parent_metadata(self):
        chunks = [
            {
                "chunk_id": "child-1",
                "parent_id": "parent-1",
                "doc_id": "doc",
                "page_start": 1,
                "page_end": 1,
                "text": "hijo",
                "parent_text": "texto padre completo",
                "retrieval_text": "hijo",
                "chunking_strategy": "parent_child_tokens",
            }
        ]

        embedded = embed_chunks(chunks, requestor=lambda text, model, base_url, api_key: [1.0])
        self.assertEqual(embedded[0]["embedding"], [1.0])
        self.assertEqual(embedded[0]["parent_id"], "parent-1")
        self.assertEqual(embedded[0]["parent_text"], "texto padre completo")

    def test_request_embedding_accepts_embedding_or_embeddings_response(self):
        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload
                self.status_code = 200
                self.text = json.dumps(payload)

            def json(self):
                return self.payload

        from unittest import mock

        with mock.patch("ingesta.embeddings.requests.post", return_value=FakeResponse({"embedding": [0.1]})) as post:
            self.assertEqual(request_embedding("hola", timeout=7), [0.1])
        self.assertEqual(post.call_args.kwargs["timeout"], 7)

        with mock.patch("ingesta.embeddings.requests.post", return_value=FakeResponse({"embeddings": [[0.2]]})):
            self.assertEqual(request_embedding("hola"), [0.2])

    def test_process_document_writes_blocks_and_chunks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            pdf_path = tmp_path / "sample.pdf"

            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), "Primer parrafo.\n\nSegundo parrafo.")
            doc.save(pdf_path)
            doc.close()

            result = process_document(pdf_path=pdf_path, output_root=tmp_path / "out", max_chars=100)

            self.assertEqual(result["page_count"], 1)
            self.assertGreaterEqual(result["block_count"], 1)
            self.assertGreaterEqual(result["chunk_count"], 1)
            self.assertTrue(pathlib.Path(result["blocks_path"]).exists())
            self.assertTrue(pathlib.Path(result["chunks_path"]).exists())

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

    def test_retrieve_top_k_orders_dense_chunks(self):
        rows = [
            {"chunk_id": "c1", "doc_id": "doc", "text": "IRPF", "embedding": [1.0, 0.0]},
            {"chunk_id": "c2", "doc_id": "doc", "text": "IVA", "embedding": [0.0, 1.0]},
            {"chunk_id": "c3", "doc_id": "doc", "text": "Sociedades", "embedding": [0.7, 0.2]},
        ]

        ranked = retrieve_top_k(
            question="pregunta sobre IRPF",
            embedding_rows=rows,
            top_k=2,
            embedder=lambda text, model, base_url, api_key: [1.0, 0.0],
            strategy="dense",
        )
        self.assertEqual([row["chunk_id"] for row in ranked], ["c1", "c3"])

    def test_tokenize_for_bm25_preserves_fiscal_terms_and_numbers(self):
        tokens = tokenize_for_bm25("El IRPF crecio un 8,4% en 2024 hasta 294.734 millones.")
        self.assertIn("irpf", tokens)
        self.assertIn("8,4%", tokens)
        self.assertIn("2024", tokens)
        self.assertIn("294.734", tokens)

    def test_bm25_scores_prioritize_exact_fiscal_matches(self):
        rows = [
            {"chunk_id": "c1", "doc_id": "doc", "text": "Recaudacion total y datos generales", "embedding": [0.9, 0.1]},
            {"chunk_id": "c2", "doc_id": "doc", "text": "El IRPF aumento un 8,4% en 2024", "embedding": [0.1, 0.9]},
        ]
        scores = bm25_scores("Cuanto aumento el IRPF en 2024?", rows)
        self.assertGreater(scores["c2"], scores["c1"])

    def test_reciprocal_rank_fusion_combines_rankings(self):
        fused = reciprocal_rank_fusion(vector_ids=["semantico", "exacto"], bm25_ids=["exacto", "semantico"])
        self.assertEqual(set(fused), {"semantico", "exacto"})
        self.assertAlmostEqual(fused["semantico"], fused["exacto"])

    def test_expand_fiscal_query_adds_acronym_and_tax_synonyms(self):
        expanded = expand_fiscal_query("Cuanto crecio el IS en 2024?")
        self.assertIn("Impuesto sobre Sociedades", expanded)
        self.assertIn("2024", expanded)

    def test_fiscal_hybrid_prioritizes_exact_tax_terms(self):
        rows = [
            {
                "chunk_id": "semantic",
                "doc_id": "doc",
                "text": "Los ingresos tributarios aumentaron por la actividad economica",
                "embedding": [1.0, 0.0],
            },
            {
                "chunk_id": "tax-exact",
                "doc_id": "doc",
                "text": "El Impuesto sobre Sociedades crecio un 11,5% en 2024",
                "embedding": [0.0, 1.0],
            },
        ]

        ranked = retrieve_top_k(
            question="Cuanto crecio el IS en 2024?",
            embedding_rows=rows,
            top_k=2,
            strategy="fiscal_hybrid",
            embedder=lambda text, model, base_url, api_key: [1.0, 0.0],
        )

        self.assertEqual(ranked[0]["chunk_id"], "tax-exact")
        self.assertIn("expanded_query", ranked[0])

    def test_fiscal_lexical_fallback_prioritizes_exact_tax_evidence(self):
        rows = [
            {
                "chunk_id": "semantic",
                "doc_id": "doc",
                "text": "Los ingresos tributarios aumentaron por la actividad economica",
                "embedding": [1.0, 0.0],
            },
            {
                "chunk_id": "tax-exact",
                "doc_id": "doc",
                "text": "El Impuesto sobre Sociedades crecio un 11,5% en 2024",
                "embedding": [0.0, 1.0],
            },
        ]

        ranked = rank_chunks_fiscal_lexical("Cuanto crecio el IS en 2024?", rows)

        self.assertEqual(ranked[0]["chunk_id"], "tax-exact")
        self.assertEqual(ranked[0]["retrieval_mode"], "lexical_fallback")

    def test_retrieval_uses_lexical_fallback_when_embedding_request_times_out(self):
        rows = [
            {
                "chunk_id": "general",
                "doc_id": "doc",
                "text": "Los ingresos tributarios aumentaron en el ejercicio.",
                "embedding": [1.0, 0.0],
            },
            {
                "chunk_id": "target",
                "doc_id": "doc",
                "text": "El IRPF crecio un 8,4% en 2024.",
                "embedding": [0.0, 1.0],
            },
        ]

        hits, mode = retrieve_top_k_with_fallback(
            question="Cuanto crecio el IRPF en 2024?",
            embedding_rows=rows,
            top_k=1,
            embedder=lambda *args: (_ for _ in ()).throw(requests.Timeout()),
        )

        self.assertEqual(mode, "lexical_fallback")
        self.assertEqual(hits[0]["chunk_id"], "target")

    def test_fiscal_hybrid_prioritizes_legal_articles_over_indexes(self):
        rows = [
            {
                "chunk_id": "index",
                "doc_id": "boe",
                "text": "Ley 37/1992 IVA INDICE TITULO PRELIMINAR. Naturaleza y ambito de aplicacion",
                "embedding": [1.0, 0.0],
            },
            {
                "chunk_id": "article",
                "doc_id": "boe",
                "text": (
                    "TITULO PRELIMINAR Naturaleza y ambito de aplicacion. "
                    "Articulo 1. Naturaleza del impuesto. El Impuesto sobre el Valor Anadido "
                    "es un tributo de naturaleza indirecta que recae sobre el consumo."
                ),
                "embedding": [0.0, 1.0],
            },
        ]

        ranked = retrieve_top_k(
            question="Que indica la Ley 37/1992 sobre el ambito de aplicacion del IVA?",
            embedding_rows=rows,
            top_k=2,
            strategy="fiscal_hybrid",
            embedder=lambda text, model, base_url, api_key: [1.0, 0.0],
        )

        self.assertTrue(is_legal_query("Que indica la Ley 37/1992 sobre el ambito de aplicacion del IVA?"))
        self.assertGreater(legal_relevance_score(rows[1]), legal_relevance_score(rows[0]))
        self.assertEqual(ranked[0]["chunk_id"], "article")

    def test_build_focused_context_keeps_child_and_relevant_parent_window(self):
        parent_text = (
            "Introduccion general sin cifras. "
            "El IRPF crecio un 7,6% en 2024 por las rentas del trabajo. "
            "Otra seccion larga sobre cuestiones no relacionadas."
        )
        focused = build_focused_context(
            child_text="IRPF 7,6%",
            parent_text=parent_text,
            question="Cuanto crecio el IRPF en 2024?",
            max_chars=90,
        )
        self.assertIn("IRPF 7,6%", focused)
        self.assertIn("7,6%", focused)
        self.assertLessEqual(len(focused), 120)

    def test_fiscal_hybrid_preserves_full_xlsx_table_context(self):
        table_text = (
            "Documento: presupuesto.xlsx\n"
            "Hoja: 11\n"
            "Fila 6: Capitulos | 2023 | 2025-P\n"
            "Fila 20: TOTAL PRESUPUESTO | 583543.3071 | 578183.02509\n"
        )
        rows = [
            {
                "chunk_id": "xlsx",
                "doc_id": "doc",
                "text": table_text,
                "parent_text": table_text,
                "retrieval_text": table_text,
                "chunking_strategy": "xlsx_rows_parent_child",
                "embedding": [1.0, 0.0],
            }
        ]

        ranked = retrieve_top_k(
            question="Cual fue el total presupuesto en 2025?",
            embedding_rows=rows,
            top_k=1,
            strategy="fiscal_hybrid",
            embedder=lambda text, model, base_url, api_key: [1.0, 0.0],
        )

        self.assertNotIn("context_text", ranked[0])
        self.assertIn("Fila 6: Capitulos", ranked[0]["parent_text"])
        self.assertIn("Fila 20: TOTAL PRESUPUESTO", ranked[0]["parent_text"])

    def test_fiscal_hybrid_boosts_matching_xlsx_table_rows(self):
        weak_table = (
            "Fila 6: Capitulos | 2023 | 2025-P\n"
            "Fila 11: PRESUPUESTO NO FINANCIERO | 200 | 100\n"
        )
        target_table = (
            "Fila 6: Capitulos | 2023 | 2025-P\n"
            "Fila 20: TOTAL PRESUPUESTO | 583543.3071 | 578183.02509\n"
        )
        rows = [
            {
                "chunk_id": "weak",
                "doc_id": "doc",
                "text": weak_table,
                "retrieval_text": weak_table,
                "parent_text": weak_table,
                "chunking_strategy": "xlsx_rows_parent_child",
                "embedding": [1.0, 0.0],
            },
            {
                "chunk_id": "target",
                "doc_id": "doc",
                "text": target_table,
                "retrieval_text": target_table,
                "parent_text": target_table,
                "chunking_strategy": "xlsx_rows_parent_child",
                "embedding": [0.0, 1.0],
            },
        ]

        question = "Cual fue el total presupuesto en 2025?"
        ranked = retrieve_top_k(
            question=question,
            embedding_rows=rows,
            top_k=2,
            strategy="fiscal_hybrid",
            embedder=lambda text, model, base_url, api_key: [1.0, 0.0],
        )

        self.assertGreater(table_relevance_score(question, rows[1]), table_relevance_score(question, rows[0]))
        self.assertEqual(ranked[0]["chunk_id"], "target")


if __name__ == "__main__":
    unittest.main()
