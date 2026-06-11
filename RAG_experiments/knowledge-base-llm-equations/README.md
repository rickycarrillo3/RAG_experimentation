# Knowledge Base — LLM-Described Equations Variant

An experiment that keeps the same text-layer extraction as the base but adds an LLM pass over equation chunks at ingest time. Chunks detected as containing equations are sent to Qwen, which writes a plain-English description of what the equation expresses. That description — not the raw LaTeX — is what gets embedded and indexed. The original LaTeX is preserved in metadata and surfaced at query time alongside the description.

---

## How it differs from the base

| | Base (`knowledge-base`) | This variant |
|---|---|---|
| **PDF extraction** | pymupdf4llm text layer | pymupdf4llm text layer (same) |
| **Equation handling** | Embedded as raw LaTeX text | Equation chunks re-written as NL descriptions by Qwen; original LaTeX kept in metadata |
| **What is indexed** | Raw chunk text | NL description (equations) / raw text (prose) |
| **Collections** | Single ChromaDB collection | Single ChromaDB collection |
| **Embedding model** | all-MiniLM-L6-v2 | all-MiniLM-L6-v2 |
| **Answer LLM** | Qwen 2.5 14B | Qwen 2.5 14B |
| **Extra LLM call** | — | One Qwen call per equation chunk at ingest |

The main bet is that embedding a natural-language description of an equation improves semantic retrieval — a query phrased in plain English matches the NL description better than a blob of LaTeX symbols. The trade-off is slower ingest (one LLM call per equation chunk) and potential description inaccuracy.

---

## Models

| Role | Model | Where |
|---|---|---|
| Equation description generation | `qwen2.5:14b` | Ollama (local) — used at **ingest** |
| Embedding | `all-MiniLM-L6-v2` | HuggingFace (local) |
| Answer generation | `qwen2.5:14b` | Ollama (local) — used at **query** |

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

### 3. Drop PDFs into docs/

---

## Usage

```bash
# Ingest (slower if docs have many equations)
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
   ├─ no equation → embed raw text
   └─ has equation → qwen2.5:14b describes it in plain English
                          ↓ NL description embedded; original LaTeX in metadata
All chunks → all-MiniLM-L6-v2 → ChromaDB (single collection)

User question
   ↓
Question embedding → ChromaDB similarity search (top 4)
   ↓
Context formatted:
  [Equation description]: <NL text>
  [Original notation]:    <LaTeX>   (for equation chunks)
  <raw text>                        (for prose chunks)
   ↓
qwen2.5:14b → Answer + sources
```
