from __future__ import annotations

import csv
import json
import pathlib
from typing import Any

import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


# Lee el JSONL de embeddings que ya generó el extractor simple.
#
# Entra:
# - embeddings_path: ruta al fichero `.jsonl`.
# Sale:
# - una lista de filas con texto y embedding.
# Por qué existe:
# - este módulo solo convierte datos existentes al formato de TensorBoard Projector.
def load_embedding_rows(embeddings_path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with embeddings_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row.get("embedding"), list):
                raise ValueError(f"Fila sin embedding usable en {embeddings_path}.")
            rows.append(row)
    return rows


# Convierte las listas de floats en una matriz numérica.
#
# Entra:
# - embedding_rows: filas del JSONL.
# Sale:
# - una matriz `numpy` con una fila por chunk.
# Por qué existe:
# - TensorFlow necesita una matriz homogénea para construir el checkpoint.
def build_embedding_matrix(embedding_rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([row["embedding"] for row in embedding_rows], dtype=np.float32)


# Construye el texto de página que se verá en Projector.
#
# Entra:
# - row: fila del embedding con `page_start` y `page_end`.
# Sale:
# - una página (`3`), un rango (`3-5`) o vacío si faltan datos.
# Por qué existe:
# - queremos una etiqueta simple de página sin más heurísticas.
def build_page_label(row: dict[str, Any]) -> str:
    page_start = row.get("page_start")
    page_end = row.get("page_end")
    if page_start and page_end and page_start != page_end:
        return f"{page_start}-{page_end}"
    if page_start or page_end:
        return str(page_start or page_end)
    return ""


# Genera la metadata mínima que vamos a enseñar en TensorBoard Projector.
#
# Entra:
# - embedding_rows: filas del JSONL original.
# Sale:
# - una lista de filas con `page` y `preview_corto`.
# Por qué existe:
# - el visor solo necesita esas dos columnas.
def build_metadata_rows(embedding_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    metadata_rows: list[dict[str, str]] = []
    for row in embedding_rows:
        text = str(row["text"]).replace("\n", " ").strip()
        metadata_rows.append({"page": build_page_label(row), "preview_corto": text[:90]})
    return metadata_rows


# Exporta metadata, checkpoint y config para abrir los embeddings en Projector.
#
# Entra:
# - embeddings_path: JSONL del extractor simple.
# - output_dir: carpeta de salida para TensorBoard.
# Sale:
# - rutas principales creadas por el exportador.
# Por qué existe:
# - este es el único paso necesario entre tu `.jsonl` y TensorBoard Projector.
def export_projector_files(embeddings_path: pathlib.Path, output_dir: pathlib.Path) -> dict[str, str]:
    embedding_rows = load_embedding_rows(embeddings_path)
    if not embedding_rows:
        raise ValueError("No hay embeddings para exportar a Projector.")

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.tsv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["page", "preview_corto"], delimiter="\t")
        writer.writeheader()
        writer.writerows(build_metadata_rows(embedding_rows))

    weights = tf.Variable(build_embedding_matrix(embedding_rows), name="embedding")
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(str(output_dir / "embedding.ckpt"))

    writer = tf.summary.create_file_writer(str(output_dir))
    with writer.as_default():
        tf.summary.text("projector_run", tf.constant([embeddings_path.stem]), step=0)
    writer.flush()
    writer.close()

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = metadata_path.name
    projector.visualize_embeddings(str(output_dir), config)

    return {
        "doc_id": embeddings_path.stem,
        "output_dir": str(output_dir),
        "metadata_path": str(metadata_path),
        "config_path": str(output_dir / "projector_config.pbtxt"),
        "tensorboard_logdir": str(output_dir.parent),
    }
