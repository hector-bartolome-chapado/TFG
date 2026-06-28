# Sistema RAG fiscal

Prototipo RAG para consultar documentacion publica fiscal y presupuestaria espanola.

## Estructura

- `interfaz/`: aplicacion Streamlit y servicio RAG.
- `scripts/RECUPERADOR/`: recuperacion densa, BM25, hibrida fiscal y ajuste para consultas juridicas.
- `scripts/simple_extractor/`: utilidades de ingesta, chunking y generacion de embeddings.
- `RESULTADOS EMBEDDING/embeddings/`: embeddings JSONL usados como indice local del RAG.
- `tests/`: pruebas unitarias de ingesta, recuperacion e interfaz.

## Modulos de mejora

Las mejoras del sistema final estan separadas en archivos especificos para que se pueda revisar la progresion tecnica:

- `interfaz/question_routing.py`: clasificacion de preguntas por tipo de respuesta.
- `interfaz/table_answers.py`: extraccion determinista de celdas en XLSX presupuestarios.
- `interfaz/legal_answers.py`: respuesta juridica basada en articulos recuperados del BOE.
- `interfaz/controlled_generation.py`: logica de decision, rechazo y respuesta controlada.
- `scripts/RECUPERADOR/fiscal_scoring.py`: expansion de consultas fiscales y coincidencia exacta de impuestos/cifras.
- `scripts/RECUPERADOR/legal_scoring.py`: priorizacion de articulos juridicos frente a indices.
- `scripts/RECUPERADOR/table_scoring.py`: boost de filas y cabeceras tabulares.
- `scripts/RECUPERADOR/text_matching.py`: BM25, tokenizacion y normalizacion compartida.

## Configuracion

El sistema usa el servidor Llamus configurado en `scripts/simple_extractor/config.py`.
Para ejecutar consultas reales, define una de estas opciones:

```powershell
$env:LLAMUS_API_KEY = "tu_api_key"
```

o crea `.llamus_api_key` a partir de `.llamus_api_key.example`.

## Ejecucion

```powershell
streamlit run .\interfaz\app.py
```

## Tests

```powershell
python -m unittest discover -s tests
```

## Notas

Los documentos originales no se incluyen en el repositorio limpio. El RAG funciona con los embeddings JSONL ya generados, que actuan como persistencia local auditable para el tamano del corpus del TFG.
