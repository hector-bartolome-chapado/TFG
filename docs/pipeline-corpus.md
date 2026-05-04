# Pipeline del corpus

## Resumen

El corpus se construye en dos etapas separadas:

1. extraccion y normalizacion determinista
2. enriquecimiento LLM posterior al chunking

La separacion evita mezclar limpieza documental con generacion de metadata y mantiene el sistema depurable.

## Entrada

El pipeline trabaja con:

- PDFs narrativos
- XLSX asociados cuando existen tablas fiables en formato estructurado

El bruto se conserva fuera del repositorio en `PDF SIN LIMPIAR/`.

## Salidas

Por documento se generan:

- `markdown/<doc>.md`
- `json/<doc>.json`
- `blocks/<doc>.jsonl`
- `chunks/<doc>.jsonl`
- `chunks_enriched/<doc>.jsonl` cuando se ejecuta enriquecimiento

## Scripts principales

### `scripts/clean_aeat_pdf.py`

Extractor inicial para el PDF AEAT. Produce una representacion limpia por pagina y sirve como base para el tratamiento de documentos narrativos.

### `scripts/corpus_pipeline.py`

Pipeline principal del corpus:

- integra PDF y XLSX
- construye bloques estructurales
- genera chunks por seccion o bloque
- selecciona chunks aptos para enriquecimiento
- llama al servidor `llamus` para anadir metadata

## Enriquecimiento con llamus

El enriquecimiento se ejecuta solo sobre chunks selectivos, por ejemplo:

- contenido narrativo
- tablas
- notas relevantes
- bloques densos o normativos

Se excluyen:

- portadas
- indices
- chunks vacios
- fragmentos triviales

La metadata generada se agrega al chunk, no sustituye el texto base.

## Modelos base

- embeddings previstos: `mxbai-embed-large:v1`
- generacion de metadata: `llama3.1:8b`

## Documentos piloto

- `AEAT_informe_anual_2024.pdf`
- `01 Presupuestos Generales del Estado Consolidados 2023.pdf`
- `C.G.E. 2023.pdf`
