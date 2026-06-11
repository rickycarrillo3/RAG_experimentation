# Personal Knowledge Base — RAG with Qwen 2.5

A local RAG (Retrieval-Augmented Generation) application. Drop in PDFs or transcripts, ask questions, and get answers grounded in your documents. Runs entirely locally — no API keys or internet connection required.

---

## Setup

### 1. Install Ollama and pull the model

Download Ollama from https://ollama.com, then pull the model (~9 GB):

```bash
ollama pull qwen2.5:14b
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Drop PDFs into docs/

```
docs/
  my-paper.pdf
  company-handbook.pdf
  ...
```

---

## Usage

### Step 1 — Ingest documents

Loads PDFs and TXT files, chunks them, embeds with a local model, and saves to ChromaDB.

```bash
python ingest.py
```

Re-run this whenever you add new documents.

### Step 2a — CLI chat

```bash
python query.py
```

Type questions, get answers with source citations. Type `quit` to exit.

### Step 2b — Streamlit UI

```bash
streamlit run app.py
```

Opens a browser chat interface at `http://localhost:8501`.

---

## How RAG works in this project

RAG = **Retrieve** relevant context, then **Augment** the LLM prompt with it, then **Generate** an answer.

```
Your PDFs / transcripts
   ↓ (ingest.py)
Text chunks  →  Embeddings (all-MiniLM-L6-v2)  →  ChromaDB (vector store)

User question
   ↓ (query.py / app.py)
Question embedding  →  Similarity search in ChromaDB  →  Top 4 chunks
   ↓
Chunks injected into prompt  →  Qwen 2.5 14B (local via Ollama)  →  Answer + sources
```

---

## Project structure

```
knowledge-base/
├── docs/           ← drop your PDFs here
├── chroma_db/      ← auto-created; holds the vector index (gitignored)
├── ingest.py       ← ingestion pipeline
├── query.py        ← CLI chat
├── app.py          ← Streamlit UI
└── requirements.txt
```
