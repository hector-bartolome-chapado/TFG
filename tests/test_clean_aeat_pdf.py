import json
import pathlib
import sys
import tempfile
import unittest

import fitz


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.clean_aeat_pdf import (
    DEFAULT_SKIP_PATTERNS,
    build_markdown,
    clean_page_text,
    detect_page_metadata,
    extract_pdf_to_outputs,
)


class CleanAeatPdfTests(unittest.TestCase):
    def test_clean_page_text_removes_known_header_footer_and_blank_lines(self):
        raw_text = (
            "AGENCIA TRIBUTARIA - INFORME ANUAL 2024\n\n"
            "Recaudacion tributaria y contexto general.\n\n"
            "Pagina 3\n"
        )

        cleaned = clean_page_text(
            raw_text,
            skip_patterns=[
                r"^AGENCIA TRIBUTARIA - INFORME ANUAL 2024$",
                r"^Pagina \d+$",
            ],
        )

        self.assertEqual(cleaned, "Recaudacion tributaria y contexto general.")

    def test_extract_pdf_to_outputs_writes_markdown_and_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            pdf_path = tmp_path / "sample.pdf"
            md_path = tmp_path / "sample.md"
            json_path = tmp_path / "sample.json"

            doc = fitz.open()
            page = doc.new_page()
            page.insert_text(
                (72, 72),
                "AGENCIA TRIBUTARIA - INFORME ANUAL 2024\n\nIngresos tributarios.\n\nPagina 1",
            )
            doc.save(pdf_path)
            doc.close()

            result = extract_pdf_to_outputs(
                pdf_path=pdf_path,
                markdown_path=md_path,
                json_path=json_path,
                skip_patterns=[
                    r"^AGENCIA TRIBUTARIA - INFORME ANUAL 2024$",
                    r"^Pagina \d+$",
                ],
            )

            self.assertEqual(result["page_count"], 1)
            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())

            markdown = md_path.read_text(encoding="utf-8")
            payload = json.loads(json_path.read_text(encoding="utf-8"))

            self.assertIn("# sample", markdown)
            self.assertIn("## Pagina 1", markdown)
            self.assertIn("Ingresos tributarios.", markdown)

            self.assertEqual(payload["doc_id"], "sample")
            self.assertEqual(len(payload["pages"]), 1)
            self.assertEqual(payload["pages"][0]["text"], "Ingresos tributarios.")

    def test_build_markdown_renders_page_sections(self):
        markdown = build_markdown(
            doc_id="aeat_informe_anual_2024",
            pages=[
                {
                    "page": 1,
                    "text": "Texto de prueba.",
                    "page_kind": "content",
                    "section_title": "Presentación",
                    "table_like": False,
                }
            ],
        )

        self.assertIn("# aeat_informe_anual_2024", markdown)
        self.assertIn("## Pagina 1 - Presentación", markdown)
        self.assertIn("- Tipo: content", markdown)
        self.assertIn("Texto de prueba.", markdown)

    def test_default_patterns_remove_running_header_with_page_number(self):
        raw_text = (
            "INFORME ANUAL DE RECAUDACION TRIBUTARIA 2024 Pagina 5\n"
            "Presentacion\n\n"
            "Texto de prueba."
        )

        cleaned = clean_page_text(raw_text, DEFAULT_SKIP_PATTERNS)

        self.assertEqual(cleaned, "Presentacion\n\nTexto de prueba.")

    def test_detect_page_metadata_marks_index_and_table_like_content(self):
        text = (
            "ÍNDICE\n\n"
            "Presentación ................................ 5\n"
            "Resumen ..................................... 7\n"
            "1. Los ingresos tributarios en 2024 ......... 8"
        )

        metadata = detect_page_metadata(text)

        self.assertEqual(metadata["page_kind"], "index")
        self.assertEqual(metadata["section_title"], "ÍNDICE")
        self.assertTrue(metadata["table_like"])

    def test_detect_page_metadata_finds_section_title_in_content_page(self):
        text = (
            "Presentación\n\n"
            "El objetivo del Informe Anual de Recaudación Tributaria "
            "es ofrecer información sobre el nivel y la evolución anual."
        )

        metadata = detect_page_metadata(text)

        self.assertEqual(metadata["page_kind"], "content")
        self.assertEqual(metadata["section_title"], "Presentación")
        self.assertFalse(metadata["table_like"])

    def test_extract_pdf_to_outputs_includes_page_metadata_in_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            pdf_path = tmp_path / "sample.pdf"
            md_path = tmp_path / "sample.md"
            json_path = tmp_path / "sample.json"

            doc = fitz.open()
            page = doc.new_page()
            page.insert_text(
                (72, 72),
                "Presentación\n\nIngresos tributarios y estructura del informe.",
            )
            doc.save(pdf_path)
            doc.close()

            extract_pdf_to_outputs(
                pdf_path=pdf_path,
                markdown_path=md_path,
                json_path=json_path,
                skip_patterns=[],
            )

            payload = json.loads(json_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["pages"][0]["page_kind"], "content")
            self.assertEqual(payload["pages"][0]["section_title"], "Presentación")
            self.assertFalse(payload["pages"][0]["table_like"])


if __name__ == "__main__":
    unittest.main()
