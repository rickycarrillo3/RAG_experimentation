# Knowledge Base — Vision Extraction Variant

An experiment that replaces pymupdf4llm's text-layer extraction with a multimodal vision model. Each PDF page is rendered as a pixel image and sent to a local vision LLM, which reads the page visually and returns text plus LaTeX-formatted equations. The rest of the pipeline (chunking, embedding, retrieval, answering) is identical to the base.

---

## How it differs from the base

| | Base (`knowledge-base`) | This variant |
|---|---|---|
| **PDF extraction** | pymupdf4llm reads the text layer | Each page rendered to PNG → sent to llava via Ollama vision API |
| **Equation handling** | Raw text/markdown from the PDF | Vision model writes equations in LaTeX (`$...$` / `$$...$$`) |
| **Collections** | Single ChromaDB collection | Single ChromaDB collection |
| **Embedding model** | all-MiniLM-L6-v2 | all-MiniLM-L6-v2 |
| **Answer LLM** | Qwen 2.5 14B | Qwen 2.5 14B |
| **Extra model** | — | llava (vision extraction at ingest) |

The main bet here is that a vision model can recover equations that the text layer misses or garbles. The trade-off is that ingestion is much slower (one Ollama call per page) and accuracy depends on the vision model's OCR quality.

---

## Models

| Role | Model | Where |
|---|---|---|
| PDF-to-text extraction | `llava` | Ollama (local) |
| Embedding | `all-MiniLM-L6-v2` | HuggingFace (local) |
| Answer generation | `qwen2.5:14b` | Ollama (local) |

---

## Setup

### 1. Pull the required Ollama models

```bash
ollama pull llava
ollama pull qwen2.5:14b
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Drop PDFs into docs/

---

## Usage

```bash
# Ingest (slow — one vision call per page)
python ingest.py

# Query
python query.py
```

---

## Pipeline

```
PDF pages
   ↓ rendered to PNG at 150 DPI (pymupdf)
Base64 image  →  llava (Ollama vision API)  →  text + LaTeX equations
   ↓
Chunks  →  all-MiniLM-L6-v2 embeddings  →  ChromaDB

User question
   ↓
Question embedding  →  ChromaDB similarity search (top 4)
   ↓
Context (with LaTeX equations)  →  qwen2.5:14b  →  Answer + sources
```
