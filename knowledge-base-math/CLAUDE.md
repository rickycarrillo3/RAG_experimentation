# knowledge-base-math

A math-focused RAG QA system for family use. Goal: performance comparable to Claude/OpenAI, but built entirely on free/open-source models and infrastructure (no paid APIs).

## Architecture

- **extract.py** — PDF → `.mmd` (Markdown+LaTeX) via `pymupdf4llm`/`nougat-ocr`, output to `docs/extracted/`.
- **ingest.py** — chunks `.mmd` files (`RecursiveCharacterTextSplitter`, size 400/overlap 80) and builds two indexes per user: a BM25 pickle (`bm25_indexes/user_<name>.pkl`) and a Chroma collection (`chroma_db/`, collection `user_<name>`).
- **query.py** — CLI for hybrid retrieval: BM25 + dense (`BAAI/bge-small-en-v1.5` embeddings) merged via Reciprocal Rank Fusion (RRF, k=60), top 5 chunks passed to the LLM.
- **app.py** — Gradio 6.18 web UI (port 7860). Duplicates the retrieval/RRF logic from query.py and ingest.py inline (not imported) except for `extract`/`build_bm25`/`build_chroma`/`load_mmd_files`, which it does import. Per-user isolation by username string (lowercased, no auth). Falls back to answering from the model's own knowledge if a user has no uploaded documents yet.
- **test_chat.py** — basic smoke tests for the chat flow.

## Model & infra

- LLM: `t1c/deepseek-math-7b-rl:Q4` served via Ollama.
- Embeddings: `BAAI/bge-small-en-v1.5` (HuggingFace, normalized).
- Designed to run on a RunPod GPU pod; see `SETUP.md` for one-time pod setup and `startup.sh` for per-session start (starts Ollama if not already running, then `python app.py`).
- Everything must survive on `/workspace` (persistent volume) between pod restarts, including `OLLAMA_MODELS`.

## Conventions / constraints

- Prioritize free/open-source models and infra when suggesting changes (Ollama, HuggingFace-hosted models, open embedding models). Do not recommend paid APIs (OpenAI, Anthropic) as the generation backbone.
- Math retrieval quality (notation, proofs, formulas) is a primary concern — consider math-aware chunking/embeddings over generic RAG defaults.
- User wants the option to fine-tune models themselves (embedding and/or generation LLM) down the line.
- Multi-user isolation is per-username directory/collection naming, not real auth — keep this in mind before suggesting security-sensitive changes.
