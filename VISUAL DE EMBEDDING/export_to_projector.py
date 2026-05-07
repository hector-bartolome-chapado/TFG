from __future__ import annotations

import csv
import json
import pathlib
from typing import Any

import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


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


def build_embedding_matrix(embedding_rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([row["embedding"] for row in embedding_rows], dtype=np.float32)


def build_page_label(row: dict[str, Any]) -> str:
    page_start = row.get("page_start")
    page_end = row.get("page_end")
    if page_start is not None:
        return str(page_start)
    if page_end is not None:
        return str(page_end)
    return ""


def build_metadata_rows(embedding_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    metadata_rows: list[dict[str, str]] = []
    for row in embedding_rows:
        text = str(row["text"]).replace("\n", " ").strip()
        metadata_rows.append({"page": build_page_label(row), "preview_corto": text[:90]})
    return metadata_rows


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
