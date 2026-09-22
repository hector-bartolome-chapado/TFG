# Sistema RAG fiscal

Prototipo RAG para consultar documentacion publica fiscal y presupuestaria espanola.

## Estructura

- `ingesta/`: utilidades de ingesta, chunking y generacion de embeddings.
- `recuperacion/`: recuperacion densa, BM25, hibrida fiscal y ajuste para consultas juridicas.
- `generacion/`: clasificacion de preguntas, construccion de contexto, respuestas controladas y formatos especializados.
- `interfaz/`: aplicacion Streamlit.
- `RESULTADOS EMBEDDING/embeddings/`: embeddings JSONL usados como indice local del RAG.
- `tests/`: pruebas unitarias de ingesta, recuperacion e interfaz.

## Modulos de mejora

Las mejoras del sistema final estan separadas en archivos especificos para que se pueda revisar la progresion tecnica:

- `generacion/question_routing.py`: clasificacion de preguntas por tipo de respuesta.
- `generacion/table_answers.py`: extraccion determinista de celdas en XLSX presupuestarios.
- `generacion/legal_answers.py`: respuesta juridica basada en articulos recuperados del BOE.
- `generacion/controlled_generation.py`: logica de decision, rechazo y respuesta controlada.
- `recuperacion/fiscal_scoring.py`: expansion de consultas fiscales y coincidencia exacta de impuestos/cifras.
- `recuperacion/legal_scoring.py`: priorizacion de articulos juridicos frente a indices.
- `recuperacion/table_scoring.py`: boost de filas y cabeceras tabulares.
- `recuperacion/text_matching.py`: BM25, tokenizacion y normalizacion compartida.

## Configuracion

El sistema usa el servidor Llamus configurado en `ingesta/config.py`.
Para ejecutar consultas reales, define una de estas opciones:

```powershell
$env:LLAMUS_API_KEY = "tu_api_key"
```

o crea `.llamus_api_key` a partir de `.llamus_api_key.example`.

## Ejecucion

```powershell
streamlit run .\interfaz\app.py
```

## Demostración web pública

La interfaz pública se encuentra en `interfaz/public_app.py`. Utiliza los índices JSONL
incluidos en el repositorio y pide al servidor institucional únicamente el embedding de
cada consulta. El servidor y el modelo están fijados en el código para que un visitante
no pueda cambiar el destino de la clave de acceso.

Para publicarla en Streamlit Community Cloud, selecciona este repositorio y el archivo
de entrada `v2/interfaz/public_app.py`. En la configuración avanzada de la aplicación,
guarda la clave como secreto (nunca en Git):

```toml
LLAMUS_API_KEY = "clave_autorizada"
```

El archivo `interfaz/requirements.txt` contiene las dependencias de la web. La app
requiere que el servidor institucional acepte peticiones desde el alojamiento externo.

## Tests

```powershell
python -m unittest discover -s tests
```

## Notas

Los documentos originales no se incluyen en el repositorio limpio. El RAG funciona con los embeddings JSONL ya generados, que actuan como persistencia local auditable para el tamano del corpus del TFG.
