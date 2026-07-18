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
python extract.py docs/raw/textbook.pdf --force-pymupdf   # skip Marker

# Ingest .mmd file(s) into a user's BM25 + Chroma indexes
python ingest.py --user alice docs/extracted/textbook.mmd
python ingest.py --user alice docs/extracted/                # whole directory

# Query from the CLI (hybrid retrieval + LLM)
python query.py --user alice
python query.py --user alice --retrieval-only    # print retrieved chunks, skip LLM
python query.py --user alice --no-bm25           # dense-only (for comparison)
python query.py --user alice --no-dense          # BM25-only (for comparison)

# Evaluate retrieval (see EVALUATION.md for the protocol — read it before trusting numbers)
python make_evalset.py --user alice --n 50   # build eval/goldset.jsonl, THEN HAND-CLEAN IT
python eval.py --user alice --all            # sweep configs: recall@5, recall@pool, MRR, nDCG, latency
python eval.py --user alice --all --answers  # also LLM-judge the end-to-end answers (slow)

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

- **extract.py** — PDF → `.mmd` (Markdown+LaTeX). Samples the first few pages to decide the extractor: `marker-pdf` for math-heavy PDFs (proper Markdown+LaTeX via the Surya models), falling back to `pymupdf4llm` on Marker failure or for non-math PDFs. Marker downloads its models to the HuggingFace cache on first run (GPU-fast, CPU-slow-but-correct). Output goes to `docs/extracted/`.
  - **⚠️ DO NOT detect math by grepping for LaTeX (`\frac`, `$$`, `\int`, …) in `pymupdf4llm` output — this mistake has been made repeatedly.** `pymupdf4llm` output contains **NO LaTeX whatsoever**; it renders equations as Unicode glyphs (`γ`, `≡`, `∫`, `∂`, sub/superscripts). Any LaTeX-pattern regex over a pymupdf sample matches nothing, so it silently classifies every math PDF as "not math" and routes it to the wrong (pymupdf) extractor. Math detection on a pymupdf sample must key off **Unicode math glyphs** (or a real layout/OCR signal) — never LaTeX tokens. LaTeX only exists downstream, in Marker's output.
- **ingest.py** — chunks `.mmd` files (`RecursiveCharacterTextSplitter`, chunk_size=400/overlap=80) and builds two indexes per user: a BM25 pickle (`bm25_indexes/user_<name>.pkl`, contains `{"bm25": BM25Okapi, "chunks": [...]}`) and a Chroma collection (`chroma_db/`, collection name `user_<name>`).
- **retrieval.py** — **the retrieval pipeline, and the single source of truth for it.** BM25 top-10 + dense top-10 (`BAAI/bge-small-en-v1.5`) merged via Reciprocal Rank Fusion (RRF, k=60); the RRF top-20 candidate pool is rescored by a cross-encoder (`BAAI/bge-reranker-v2-m3`) and the top 5 go to the LLM. Imported by `query.py`, `app.py`, and `eval.py` — change retrieval behaviour here and nowhere else. Tunables live here too (`TOP_K`, `RERANK_TOP_C`, `TOP_N`, `RRF_K`).
  - `retrieve()` returns the final top-N. `retrieve_detailed()` additionally returns the pre-rerank candidate pool, the full ranked list, and per-stage timings — that's what `eval.py` needs.
  - **`top_k` bounds the candidate pool**: RRF can emit at most `top_k × 2` candidates, so raising `RERANK_TOP_C` above that is a silent no-op. Raise `top_k` too.
- **query.py** — interactive CLI around `retrieval.py` + DeepSeek-Math generation. `--no-rerank` / `--no-bm25` / `--no-dense` toggle stages for A/B comparison.
- **app.py** — Gradio 6.18 web UI (port 7860). Imports the pipeline from `retrieval.py`; loads embeddings/reranker/LLM once at module level. Per-user isolation is by username string (lowercased, no auth) — each user gets an isolated BM25 pickle and Chroma collection. Falls back to the model's own knowledge if a user has no uploaded documents. Keeps a separate "clean" conversation history (sources stripped) for LLM context, distinct from the display history.
- **make_evalset.py** — builds `eval/goldset.jsonl` (question → gold `chunk_id`) by sampling chunks (biased toward math-bearing ones) and having an *instruct* model (`qwen2:7b`, **not** deepseek-math, which is a solver) write the question each chunk answers. Output is a **draft**: questions that parrot chunk wording flatter BM25 and must be rewritten by hand.
- **eval.py** — runs the gold set against retrieval configs (`bm25`, `dense`, `hybrid`, `hybrid+rerank`, pool variants) and reports recall@1/@5, **recall@pool**, MRR, nDCG@5, per-stage latency, peak VRAM, and optionally LLM-judged answer quality (`--answers`). **recall@pool** is the key diagnostic: the reranker can only reorder what BM25/dense already found, so if the gold chunk isn't in the pool the miss is upstream (extraction/chunking), not the reranker's fault.
- **EVALUATION.md** — the evaluation protocol: what's measured, why, the gold-set caveats, cost expectations, and how to act on the numbers. **Read before changing retrieval.**
- **test_chat.py** — ingests a fixed test doc for a `test` user and drops into the same interactive CLI loop as `query.py`, for manual end-to-end verification.

Note: retrieval logic used to be duplicated between `query.py` and `app.py`. It is now shared via `retrieval.py` — **do not reintroduce a second copy.** An `eval.py` that measures a reimplementation of the pipeline rather than the pipeline itself will drift from what ships and quietly start lying.

## Model & infra

- LLM: `t1c/deepseek-math-7b-rl:Q4` served via Ollama.
- Embeddings: `BAAI/bge-small-en-v1.5` (HuggingFace, normalized).
- Reranker: `BAAI/bge-reranker-v2-m3` cross-encoder via `sentence_transformers.CrossEncoder` (~2.2GB, downloads to the HF cache on first use — keep the HF cache on `/workspace` on the pod so it survives restarts).
- Eval-only model: `qwen2:7b` (Ollama) — used by `make_evalset.py` as the *instruct* question-writer and by `eval.py` as the LLM-as-judge. Deliberately **not** deepseek-math (a solver, not a writer; and a model must not grade its own output). Only needed when running the eval harness, not the serving pipeline — `ollama pull qwen2:7b`.
- Designed to run on a RunPod GPU pod; everything must survive on `/workspace` (persistent volume) between pod restarts, including `OLLAMA_MODELS`.

## Conventions / constraints

- Prioritize free/open-source models and infra when suggesting changes (Ollama, HuggingFace-hosted models, open embedding models). Do not recommend paid APIs (OpenAI, Anthropic) as the generation backbone.
- Math retrieval quality (notation, proofs, formulas) is a primary concern — consider math-aware chunking/embeddings over generic RAG defaults.
- User wants the option to fine-tune models themselves (embedding and/or generation LLM) down the line.
- Multi-user isolation is per-username directory/collection naming, not real auth — keep this in mind before suggesting security-sensitive changes.
- `chroma_db/`, `bm25_indexes/`, PDFs, and extracted `.mmd` files under `docs/raw`/`docs/extracted` are all gitignored (private user document data) — don't try to commit them.
