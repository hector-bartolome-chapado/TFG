# TFG RAG financiero-fiscal

Repositorio tecnico del TFG orientado a un pipeline RAG sobre documentacion publica economico-fiscal del Estado espanol.

## Objetivo

El proyecto estudia como limpiar, estructurar, fragmentar y enriquecer documentacion oficial para construir un corpus util para retrieval y evaluacion posterior. El foco actual del repositorio esta en:

- limpieza determinista de PDFs y hojas de calculo
- generacion de artefactos intermedios inspeccionables
- chunking estructural
- enriquecimiento selectivo con `llamus.cs.us.es`

## Estructura

```text
TFG/
├─ PDF LIMPIO/
│  ├─ markdown/
│  ├─ json/
│  ├─ blocks/
│  ├─ chunks/
│  └─ chunks_enriched/
├─ scripts/
├─ tests/
└─ docs/
```

## Que contiene el repositorio

- `PDF LIMPIO/`: salidas procesadas del corpus listas para inspeccion, chunking y retrieval.
- `scripts/`: utilidades del pipeline de limpieza y enriquecimiento.
- `tests/`: verificacion automatica del extractor y del pipeline.
- `docs/`: documentacion tecnica del flujo actual.

## Que no contiene

- `PDF SIN LIMPIAR/`: fuentes originales brutas.
- materiales de escritura no tecnica del TFG.
- adjuntos pesados o duplicados sin valor para el pipeline.

## Estado actual

Documentos piloto ya procesados:

- `AEAT_informe_anual_2024`
- `01 Presupuestos Generales del Estado Consolidados 2023`
- `C.G.E. 2023`

Artefactos ya generados:

- `markdown`
- `json`
- `blocks.jsonl`
- `chunks.jsonl`
- `chunks_enriched.jsonl` para el piloto AEAT

## Verificacion

Tests disponibles:

- `tests/test_clean_aeat_pdf.py`
- `tests/test_corpus_pipeline.py`

Ejecucion:

```powershell
python -m unittest tests.test_clean_aeat_pdf
python -m unittest tests.test_corpus_pipeline
```

## Siguiente fase

- generar embeddings con `mxbai-embed-large:v1`
- decidir vector store baseline
- definir metricas iniciales de retrieval y fidelidad
