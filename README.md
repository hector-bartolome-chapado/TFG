# Fiscal RAG System

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-Retrieval%20Augmented%20Generation-111827)
![Streamlit](https://img.shields.io/badge/Streamlit-Interface-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/status-academic%20TFG-blue)

Technical repository for my Computer Engineering final degree project at Universidad de Sevilla.

The project builds a Retrieval-Augmented Generation system over public fiscal documentation. It covers document ingestion, chunking, embeddings, hybrid retrieval, controlled answer generation, evaluation and a Streamlit interface.

## Why It Matters

Public fiscal documents are long, heterogeneous and difficult to query with simple keyword search. This project explores how a RAG pipeline can retrieve relevant passages and generate grounded answers while keeping the processing steps inspectable.

For recruiters, this repository shows applied AI engineering beyond a notebook: data preparation, retrieval design, evaluation, UI, tests and project organization.

## Core Features

- PDF and XLSX ingestion for fiscal and budget documentation.
- Local JSONL persistence for processed document units.
- Parent-child chunking strategy.
- Dense retrieval plus BM25 lexical retrieval.
- Hybrid retrieval fusion adapted to fiscal queries.
- Controlled generation over retrieved context.
- Streamlit interface for interactive querying.
- RAGAS-based evaluation over a curated question set.
- Unit tests for ingestion, retrieval and generation modules.

## Architecture

```text
Documents
   |
   v
Ingestion -> Chunking -> Embeddings -> Local stores
                                      |
                                      v
                              Hybrid retrieval
                                      |
                                      v
                            Controlled generation
                                      |
                                      v
                              Streamlit interface
```

## Repository Structure

```text
v1/  initial prototype and exploratory artifacts
v2/  final system organized by pipeline stage
```

The final version is under `v2/`:

```text
v2/
  ingesta/        document ingestion and preprocessing
  recuperacion/   dense, lexical and hybrid retrieval
  generacion/     answer generation and response control
  interfaz/       Streamlit application
  tests/          focused unit tests
```

## Tech Stack

- Python
- Streamlit
- BM25 retrieval
- Vector embeddings
- JSONL local persistence
- RAGAS evaluation
- Public fiscal documentation as the domain corpus

## Run Locally

From the repository root:

```powershell
cd .\v2
streamlit run .\interfaz\app.py
```

The public evidence-desk interface uses `v2/interfaz/public_app.py`:

```powershell
cd .\v2
streamlit run .\interfaz\public_app.py
```

It searches all available embedding JSONL files together, then shows the retrieved excerpts with PDF page or Excel sheet/row locations and parent context. A browser session keeps its own conversation; follow-up references are rewritten with `gemma3:12b` before the existing hybrid retrieval and controlled generation run. Independent questions do not call the rewriting model. The original PDF/XLSX files are not served by the public app, and the listed passages are retrieved evidence rather than verified sentence-level citations. Live queries require the university Llamus embedding endpoint; if it fails, the app reports the failing phase instead of generating an answer from a fallback search. The expanded-corpus chat and follow-ups are demo capabilities, not part of the controlled 80-question evaluation.

Run the test suite:

```powershell
cd .\v2
python -m unittest discover -s tests
```

## Credentials

No private credentials are versioned.

To use real calls against the Llamus server, define `LLAMUS_API_KEY` or create a local `.llamus_api_key` file from:

```text
v2/.llamus_api_key.example
```

## Portfolio Notes

This is the strongest project in my portfolio for roles involving:

- applied AI and RAG systems;
- Python backend/data pipelines;
- document processing;
- search and retrieval;
- evaluation of LLM-based systems;
- user-facing technical prototypes.

## Author

**Hector Bartolome Chapado**

Computer Engineering graduate, Universidad de Sevilla

- GitHub: [hector-bartolome-chapado](https://github.com/hector-bartolome-chapado)
- LinkedIn: [linkedin.com/in/infohbc](https://www.linkedin.com/in/infohbc/)
