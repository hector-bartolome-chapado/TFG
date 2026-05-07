# Prototipo RAG v1

Prototipo mínimo para entender y evaluar el flujo básico de un sistema RAG sobre documentos PDF:

- extracción de texto por página
- división en bloques
- chunking simple
- embeddings con `llamus`
- retrieval por similitud coseno
- interfaz `Streamlit` para preguntar al documento
- visualización de embeddings con `TensorBoard Embedding Projector`

## Estructura

- `scripts/simple_extractor/`: extracción, chunking, embeddings y configuración compartida
- `scripts/RECUPERADOR/`: retrieval mínimo
- `scripts/simple_pipeline.py`: punto de entrada programático del pipeline
- `interfaz/`: laboratorio RAG visual en `Streamlit`
- `RESULTADOS EMBEDDING/`: artefactos generados del prototipo
- `VISUAL DE EMBEDDING/`: exportador para `TensorBoard Projector`



## Flujo actual

1. `PDF -> páginas`
2. `páginas -> bloques`
3. `bloques -> chunks`
4. `chunks -> embeddings`
5. `pregunta -> embedding -> top_k`
6. `top_k -> contexto -> respuesta final`

## Requisitos

- `TFG/.llamus_api_key` o `LLAMUS_API_KEY`
- dependencias Python del entorno actual

## Interfaz

Desde la raíz de `TFG`:

```powershell
streamlit run .\interfaz\app.py
```

## Visualización de embeddings

La visualización se hace con `TensorBoard Embedding Projector` a partir de los artefactos ya preparados en:

- `RESULTADOS EMBEDDING/projector/AEAT_informe_anual_2024/`

## Estado

Este repositorio refleja el primer prototipo operativo:

- embedding real de `AEAT_informe_anual_2024.pdf`
- retrieval funcionando
- primera interfaz RAG para pruebas manuales
