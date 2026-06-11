# Knowledge Base — Hybrid Dual-Collection Variant

An experiment that routes equation chunks and prose chunks into separate ChromaDB collections, each with a different embedding model purpose-built for its content type. At query time both collections are searched independently and their results are interleaved before being passed to the LLM.

---

## How it differs from the base

| | Base (`knowledge-base`) | This variant |
|---|---|---|
| **PDF extraction** | pymupdf4llm text layer | pymupdf4llm text layer (same) |
| **Equation handling** | All chunks embedded identically | Equation chunks detected by regex and routed to a dedicated collection |
| **ChromaDB collections** | 1 (`default`) | 2 (`text_collection` + `equation_collection`) |
| **Text embedding model** | all-MiniLM-L6-v2 | all-MiniLM-L6-v2 |
| **Equation embedding model** | all-MiniLM-L6-v2 | `AnReu/math_pretrained_bert` (BERT pre-trained on arXiv math papers) |
| **Retrieval** | Single similarity search, top 4 | Two parallel searches (top 4 each), results interleaved |
| **Equation metadata** | None | Preceding and following chunk text stored as context |
| **Answer LLM** | Qwen 2.5 14B | Qwen 2.5 14B |

The main bet is that a math-specialized embedding model produces more meaningful similarity scores for equation chunks, so math-heavy queries find the right equations even when phrased in natural language. The trade-off is higher memory usage (two models loaded simultaneously) and increased ingest/query complexity.

---

## Models

| Role | Model | Where |
|---|---|---|
| Text embedding | `all-MiniLM-L6-v2` | HuggingFace (local) |
| Equation embedding | `AnReu/math_pretrained_bert` | HuggingFace (local) — BERT pre-trained on arXiv |
| Answer generation | `qwen2.5:14b` | Ollama (local) |

---

## Setup

### 1. Pull the required Ollama model

```bash
ollama pull qwen2.5:14b
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The HuggingFace models (`all-MiniLM-L6-v2` and `AnReu/math_pretrained_bert`) are downloaded automatically on first run.

### 3. Drop PDFs into docs/

---

## Usage

```bash
# Ingest — loads both embedding models; classifies and routes chunks
python ingest.py

# Query
python query.py
```

---

## Pipeline

```
PDF pages
   ↓ pymupdf4llm → markdown text
Chunks
   ├─ no equation  → text_collection   (embedded with all-MiniLM-L6-v2)
   └─ has equation → equation_collection (embedded with AnReu/math_pretrained_bert)
                          + preceding/following context stored in metadata

User question
   ↓
   ├─ text_collection.similarity_search(question, k=4)
   └─ equation_collection.similarity_search(question, k=4)
        ↓ results interleaved [text, eq, text, eq, …], duplicates removed
Context formatted:
  [Context before]: …  \
  [Equation]: …         }  for equation chunks
  [Context after]: …   /
  <raw text>               for prose chunks
   ↓
qwen2.5:14b → Answer + sources  (with text/equation chunk counts reported)
```

PROBLEM:The problem with this approach is the following: Text is extracted with Tesseract OCR (including equations) and the equation detector is looking for latex syntax, which is not there. Thus, the number of equations extracted is minimal and incorrect.