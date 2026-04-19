from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path


def run_pandoc(docx_path: Path) -> str:
    result = subprocess.run(
        ["pandoc", "--track-changes=all", str(docx_path), "-t", "markdown", "-o", "-"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return result.stdout


def extract_references(markdown: str) -> list[dict[str, str | int]]:
    marker = "#### Obras citadas\n\n"
    idx = markdown.find(marker)
    if idx == -1:
        raise ValueError("No se encontró la sección 'Obras citadas' en el documento.")

    refs_block = markdown[idx + len(marker) :].strip()
    items = re.split(r"\n(?=\d+\.\s)", refs_block)
    parsed: list[dict[str, str | int]] = []

    for item in items:
        match = re.match(r"(\d+)\.\s+(.*)", item, re.S)
        if not match:
            continue

        ref_id = int(match.group(1))
        body = " ".join(line.strip() for line in match.group(2).splitlines()).strip()
        body = re.sub(r"\s+", " ", body)

        url_match = re.search(r"\((https?://[^)\s]+)\)", body)
        url = url_match.group(1) if url_match else ""

        title = re.sub(r", fecha de acceso:.*$", "", body)
        title = re.sub(r"\s*\[\[https?://.*$", "", title)

        parsed.append(
            {
                "id": ref_id,
                "title": title,
                "url": url,
                "entry": body,
            }
        )

    return parsed


def write_markdown(references: list[dict[str, str | int]], output_path: Path, source_docx: Path) -> None:
    lines = [
        "# Fuentes extraídas del paper",
        "",
        f"Documento origen: `{source_docx}`",
        "",
        f"Total de referencias: **{len(references)}**",
        "",
    ]

    for ref in references:
        lines.append(f"{ref['id']}. {ref['entry']}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_json(references: list[dict[str, str | int]], output_path: Path) -> None:
    output_path.write_text(json.dumps(references, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(references: list[dict[str, str | int]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "title", "url", "entry"])
        writer.writeheader()
        writer.writerows(references)


def main() -> int:
    if len(sys.argv) < 3:
        print("Uso: python scripts/extract_docx_sources.py <paper.docx> <output_dir>")
        return 1

    docx_path = Path(sys.argv[1]).expanduser().resolve()
    output_dir = Path(sys.argv[2]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown = run_pandoc(docx_path)
    references = extract_references(markdown)

    write_markdown(references, output_dir / "paper_sources.md", docx_path)
    write_json(references, output_dir / "paper_sources.json")
    write_csv(references, output_dir / "paper_sources.csv")

    print(f"Extraídas {len(references)} referencias en {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

