# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`knowledge-base-math` is a math-focused RAG QA system for family use. Goal: performance comparable to Claude/OpenAI, built entirely on free/open-source models and infrastructure (no paid APIs). It is the only active project in this repo — everything else (other RAG experiments: hybrid retrieval, LLM-equations, vision, speech-to-text) was moved to the `archive/experiments` branch when this one outgrew being a side experiment. The top-level `0X_*.md` files are general RAG concept notes kept as background reading.

## Commands

All commands run from `knowledge-base-math/`, with the venv active:

```bash
cd knowledge-base-math
source venv/bin/activate
```

```bash
# Install deps
pip install -r requirements.txt

# Extract a PDF to .mmd (Markdown+LaTeX)
python extract.py docs/raw/textbook.pdf
python extract.py docs/raw/textbook.pdf --force-pymupdf   # skip Nougat

# Ingest .mmd file(s) into a user's BM25 + Chroma indexes
python ingest.py --user alice docs/extracted/textbook.mmd
python ingest.py --user alice docs/extracted/                # whole directory

# Query from the CLI (hybrid retrieval + LLM)
python query.py --user alice
python query.py --user alice --retrieval-only    # print retrieved chunks, skip LLM
python query.py --user alice --no-bm25           # dense-only (for comparison)
python query.py --user alice --no-dense          # BM25-only (for comparison)

# End-to-end smoke test: ingests docs/extracted/test.mmd for a "test" user, then interactive CLI chat
python test_chat.py
python test_chat.py --retrieval-only

# Web UI (upload + chat), http://localhost:7860
python app.py
```

Requires [Ollama](https://ollama.com) running locally with the model pulled:
```bash
ollama pull t1c/deepseek-math-7b-rl:Q4
```

On a RunPod GPU pod, see `knowledge-base-math/SETUP.md` for one-time setup and `knowledge-base-math/startup.sh` for per-session start (run from inside `knowledge-base-math/`; starts Ollama if not already running, then `python app.py`).

There is no automated test suite/linter configured — `test_chat.py` is a manual smoke test, not a pytest suite.

## Architecture

Pipeline: PDF → `extract.py` → `.mmd` → `ingest.py` → per-user BM25 + Chroma indexes → `query.py`/`app.py` → hybrid retrieval + RRF → cross-encoder rerank → LLM answer.

- **extract.py** — PDF → `.mmd` (Markdown+LaTeX). Samples the first few pages for LaTeX signals (`\frac`, `\int`, `$$`, etc.) to decide the extractor: `nougat-ocr` for math-heavy PDFs (proper LaTeX output), falling back to `pymupdf4llm` on Nougat failure or for non-math PDFs. Output goes to `docs/extracted/`.
- **ingest.py** — chunks `.mmd` files (`RecursiveCharacterTextSplitter`, chunk_size=400/overlap=80) and builds two indexes per user: a BM25 pickle (`bm25_indexes/user_<name>.pkl`, contains `{"bm25": BM25Okapi, "chunks": [...]}`) and a Chroma collection (`chroma_db/`, collection name `user_<name>`).
- **query.py** — CLI for hybrid retrieval: BM25 top-10 + dense top-10 (`BAAI/bge-small-en-v1.5` embeddings) merged via Reciprocal Rank Fusion (RRF, k=60); the RRF top-20 candidate pool is then rescored by a cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) and the top 5 passed to the LLM as context. `--no-rerank` falls back to RRF-only order (for A/B comparison), mirroring `--no-bm25`/`--no-dense`.
- **app.py** — Gradio 6.18 web UI (port 7860). Reimplements the retrieval/RRF/rerank logic from `query.py` inline rather than importing it (only `extract`, `build_bm25`, `build_chroma`, `load_mmd_files` are imported from `extract.py`/`ingest.py`); the reranker is loaded once at module level like `embeddings`/`llm`. Per-user isolation is by username string (lowercased, no auth) — each user gets an isolated BM25 pickle and Chroma collection. Falls back to answering from the model's own knowledge if a user has no uploaded documents yet. Keeps a separate "clean" conversation history (sources stripped) for LLM context, distinct from the display history shown in the chatbot.
- **test_chat.py** — ingests a fixed test doc for a `test` user and drops into the same interactive CLI loop as `query.py`, for manual end-to-end verification.

Note: retrieval logic (search, RRF, rerank) is duplicated between `query.py` and `app.py` rather than shared — keep both in sync when changing retrieval behavior (e.g. `TOP_K`, `TOP_N`, `RRF_K`, `RERANK_MODEL`, `RERANK_TOP_C`, chunking params). This duplication has now grown to three stages; it should be extracted into a shared `retrieval.py` module imported by both files.

## Model & infra

- LLM: `t1c/deepseek-math-7b-rl:Q4` served via Ollama.
- Embeddings: `BAAI/bge-small-en-v1.5` (HuggingFace, normalized).
- Reranker: `BAAI/bge-reranker-v2-m3` cross-encoder via `sentence_transformers.CrossEncoder` (~2.2GB, downloads to the HF cache on first use — keep the HF cache on `/workspace` on the pod so it survives restarts).
- Designed to run on a RunPod GPU pod; everything must survive on `/workspace` (persistent volume) between pod restarts, including `OLLAMA_MODELS`.

## Conventions / constraints

- Prioritize free/open-source models and infra when suggesting changes (Ollama, HuggingFace-hosted models, open embedding models). Do not recommend paid APIs (OpenAI, Anthropic) as the generation backbone.
- Math retrieval quality (notation, proofs, formulas) is a primary concern — consider math-aware chunking/embeddings over generic RAG defaults.
- User wants the option to fine-tune models themselves (embedding and/or generation LLM) down the line.
- Multi-user isolation is per-username directory/collection naming, not real auth — keep this in mind before suggesting security-sensitive changes.
- `chroma_db/`, `bm25_indexes/`, PDFs, and Nougat-extracted `.mmd` files under `docs/raw`/`docs/extracted` are all gitignored (private user document data) — don't try to commit them.
