# Interfaz RAG v1

Interfaz `Streamlit` para el laboratorio RAG del proyecto.

## Qué hace

- selecciona un `.jsonl` de embeddings desde `RESULTADOS EMBEDDING/embeddings`
- recibe una pregunta
- ejecuta retrieval `top_k`
- muestra los chunks recuperados y sus scores
- enseña el contexto exacto enviado al modelo
- pide una respuesta final a `llamus` con `llama3.1:8b`

## Arranque

Desde la raíz de `TFG`:

```powershell
streamlit run .\interfaz\app.py
```

## Requisitos

- tener al menos un `.jsonl` en `RESULTADOS EMBEDDING/embeddings`
- tener API key en:
  - `TFG/.llamus_api_key`
  - o `LLAMUS_API_KEY`

## Idea clave

La interfaz no sustituye al extractor ni al recuperador. Solo los conecta en una vista única para depurar retrieval y respuesta final.

