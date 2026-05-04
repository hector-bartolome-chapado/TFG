import json
import pathlib
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.corpus_pipeline import (
    block_from_text,
    build_chunks,
    build_document_payload,
    extract_json_object,
    extract_xlsx_blocks,
    enrich_chunks_with_llamus,
    should_enrich_chunk,
)


class CorpusPipelineTests(unittest.TestCase):
    def test_block_from_text_classifies_title_note_and_table(self):
        title = block_from_text(
            doc_id="doc",
            source_family="AEAT",
            page=7,
            section_path=["Presentación"],
            text="Presentación",
            block_index=0,
        )
        note = block_from_text(
            doc_id="doc",
            source_family="AEAT",
            page=8,
            section_path=["Notas"],
            text="NOTA INFORMATIVA 1: Impacto recaudatorio",
            block_index=1,
        )
        table = block_from_text(
            doc_id="doc",
            source_family="AEAT",
            page=9,
            section_path=["Cuadro 1"],
            text="IRPF 2024 120\nIVA 2024 99\nSociedades 2024 40",
            block_index=2,
        )

        self.assertEqual(title["block_type"], "title")
        self.assertEqual(note["block_type"], "note")
        self.assertEqual(table["block_type"], "table")

    def test_extract_xlsx_blocks_creates_table_blocks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            xlsx_path = tmp_path / "table.xlsx"
            df = pd.DataFrame({"Impuesto": ["IRPF", "IVA"], "Importe": [120, 90]})
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Resumen")

            blocks = extract_xlsx_blocks(
                xlsx_paths=[xlsx_path],
                doc_id="doc",
                source_family="PGE",
            )

            self.assertEqual(len(blocks), 1)
            self.assertEqual(blocks[0]["block_type"], "table")
            self.assertEqual(blocks[0]["sheet_name"], "Resumen")
            self.assertEqual(blocks[0]["table_data"][0]["Impuesto"], "IRPF")

    def test_build_chunks_preserves_table_units_and_merges_content(self):
        blocks = [
            {
                "block_id": "b1",
                "doc_id": "doc",
                "source_family": "AEAT",
                "page_start": 7,
                "page_end": 7,
                "section_path": ["Presentación"],
                "block_type": "content",
                "text": "Primer párrafo de contexto.",
                "table_data": None,
                "numeric_density": 0.0,
                "base_metadata": {"table_like": False},
            },
            {
                "block_id": "b2",
                "doc_id": "doc",
                "source_family": "AEAT",
                "page_start": 7,
                "page_end": 7,
                "section_path": ["Presentación"],
                "block_type": "content",
                "text": "Segundo párrafo de contexto.",
                "table_data": None,
                "numeric_density": 0.0,
                "base_metadata": {"table_like": False},
            },
            {
                "block_id": "b3",
                "doc_id": "doc",
                "source_family": "AEAT",
                "page_start": 8,
                "page_end": 8,
                "section_path": ["Cuadro 1"],
                "block_type": "table",
                "text": "IRPF\t120\nIVA\t90",
                "table_data": [{"Impuesto": "IRPF", "Importe": 120}],
                "numeric_density": 0.5,
                "base_metadata": {"table_like": True},
            },
        ]

        chunks = build_chunks(blocks, max_chars=200)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["block_type"], "content")
        self.assertIn("Primer párrafo", chunks[0]["text"])
        self.assertIn("Segundo párrafo", chunks[0]["text"])
        self.assertEqual(chunks[1]["block_type"], "table")
        self.assertIsNotNone(chunks[1]["table_data"])

    def test_should_enrich_chunk_is_selective(self):
        self.assertFalse(
            should_enrich_chunk(
                {"block_type": "index", "text": "ÍNDICE", "base_metadata": {}, "numeric_density": 0.0}
            )
        )
        self.assertFalse(
            should_enrich_chunk(
                {"block_type": "content", "text": "corto", "base_metadata": {}, "numeric_density": 0.0}
            )
        )
        self.assertTrue(
            should_enrich_chunk(
                {
                    "block_type": "content",
                    "text": "Este bloque contiene explicación extensa sobre IRPF y recaudación en 2024.",
                    "base_metadata": {},
                    "numeric_density": 0.15,
                }
            )
        )
        self.assertTrue(
            should_enrich_chunk(
                {"block_type": "table", "text": "IRPF\t120\nIVA\t90", "base_metadata": {}, "numeric_density": 0.5}
            )
        )

    def test_extract_json_object_handles_code_fences(self):
        payload = extract_json_object(
            "```json\n{\"summary\":\"ok\",\"answerable_questions\":[]}\n```"
        )
        self.assertEqual(payload["summary"], "ok")

    def test_enrich_chunks_with_llamus_preserves_base_fields(self):
        chunks = [
            {
                "chunk_id": "chunk-1",
                "doc_id": "doc",
                "source_family": "AEAT",
                "section_path": ["Resumen"],
                "block_type": "content",
                "text": "Los ingresos tributarios crecieron un 8,4% en 2024 y el IRPF tuvo un papel destacado.",
                "table_data": None,
                "page_start": 9,
                "page_end": 9,
                "base_metadata": {"table_like": False},
                "numeric_density": 0.1,
            }
        ]

        def fake_requestor(chunk, model, base_url, api_key):
            self.assertEqual(model, "llama3.1:8b")
            self.assertEqual(base_url, "https://llamus.cs.us.es")
            self.assertEqual(api_key, "secret")
            return {
                "summary": "Resumen",
                "answerable_questions": ["¿Cuánto crecieron los ingresos tributarios?"],
                "keywords": ["ingresos tributarios", "IRPF"],
                "entities": ["AEAT"],
                "taxes_mentioned": ["IRPF"],
                "years_mentioned": [2024],
                "amounts_mentioned": ["8,4%"],
                "organization": "AEAT",
                "is_numeric_evidence": True,
                "is_normative_evidence": False,
                "relationships": ["IRPF contribuye al crecimiento de ingresos"],
                "assumptions_or_conditions": [],
                "confidence_note": "alta",
                "analytic_tags": ["recaudación", "resumen"],
            }

        enriched = enrich_chunks_with_llamus(
            chunks=chunks,
            model="llama3.1:8b",
            base_url="https://llamus.cs.us.es",
            api_key="secret",
            requestor=fake_requestor,
        )

        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0]["chunk_id"], "chunk-1")
        self.assertIn("llm_metadata", enriched[0])
        self.assertEqual(enriched[0]["llm_metadata"]["organization"], "AEAT")

    def test_enrich_chunks_with_llamus_honors_enrich_limit(self):
        chunks = [
            {
                "chunk_id": "chunk-1",
                "doc_id": "doc",
                "source_family": "AEAT",
                "section_path": ["Resumen"],
                "block_type": "content",
                "text": "La recaudación del IRPF en 2024 creció un 8,4% respecto al año anterior.",
                "table_data": None,
                "page_start": 9,
                "page_end": 9,
                "base_metadata": {"table_like": False},
                "numeric_density": 0.1,
            },
            {
                "chunk_id": "chunk-2",
                "doc_id": "doc",
                "source_family": "AEAT",
                "section_path": ["Resumen"],
                "block_type": "content",
                "text": "El IVA mantuvo un peso relevante en los ingresos tributarios de 2024.",
                "table_data": None,
                "page_start": 10,
                "page_end": 10,
                "base_metadata": {"table_like": False},
                "numeric_density": 0.09,
            },
        ]

        def fake_requestor(chunk, model, base_url, api_key):
            return {"summary": chunk["chunk_id"], "organization": "AEAT"}

        enriched = enrich_chunks_with_llamus(
            chunks=chunks,
            model="llama3.1:8b",
            base_url="https://llamus.cs.us.es",
            api_key="secret",
            requestor=fake_requestor,
            enrich_limit=1,
        )

        self.assertIsNotNone(enriched[0]["llm_metadata"])
        self.assertIsNone(enriched[1]["llm_metadata"])

    def test_build_document_payload_contains_sections_and_blocks(self):
        blocks = [
            {
                "block_id": "b1",
                "doc_id": "doc",
                "source_family": "AEAT",
                "page_start": 1,
                "page_end": 1,
                "section_path": ["Resumen"],
                "block_type": "content",
                "text": "Texto.",
                "table_data": None,
                "numeric_density": 0.0,
                "base_metadata": {"table_like": False},
            }
        ]
        payload = build_document_payload("doc", pathlib.Path("sample.pdf"), blocks)
        self.assertEqual(payload["doc_id"], "doc")
        self.assertEqual(payload["block_count"], 1)
        self.assertEqual(payload["blocks"][0]["section_path"], ["Resumen"])


if __name__ == "__main__":
    unittest.main()
