# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`knowledge-base-math` is a math-focused RAG QA system for family use. Goal: performance comparable to Claude/OpenAI, built entirely on free/open-source models and infrastructure (no paid APIs). It is the only active project in this repo — everything else (other RAG experiments: hybrid retrieval, LLM-equations, vision, speech-to-text) was moved to the `archive/experiments` branch when this one outgrew being a side experiment. The `Base RAG explained/` folder holds the numbered `0X_*.md` general RAG concept notes, kept as background reading.

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

# Evaluate retrieval — the eval harness lives in evaluation/, run from knowledge-base-math/
# (see evaluation/EVALUATION.md for the protocol — read it before trusting numbers)
python evaluation/make_evalset.py --user alice --n 50   # build evaluation/goldset.jsonl, THEN HAND-CLEAN IT
python evaluation/eval.py --user alice --all            # sweep configs: recall@5, recall@pool, MRR, nDCG, latency
python evaluation/eval.py --user alice --all --answers  # also LLM-judge the end-to-end answers (slow)
python evaluation/embed_chunk_sweep.py                  # 3 chunkers × 3 embedders → evaluation/results/sweep_results.json

# Pre-download all HuggingFace models (embedder + reranker + Marker/surya, ~6GB cold)
python prefetch_models.py
python prefetch_models.py --skip-marker   # query-only; skips the ~3-4GB extraction models

# End-to-end smoke test: ingests docs/extracted/test.mmd for a "test" user, then interactive CLI chat
python test_chat.py
python test_chat.py --retrieval-only

# API service (this is what gets deployed) — http://localhost:8000, docs at /docs
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Web UI (upload + chat), http://localhost:7860 — an HTTP CLIENT of the API above,
# so the API must be running first. Set KBM_API_TOKEN in both shells if it is set at all.
python app.py
```

Requires [Ollama](https://ollama.com) running locally with the model pulled:
```bash
ollama pull t1c/deepseek-math-7b-rl:Q4
```

On a RunPod GPU pod, see `knowledge-base-math/SETUP.md` for one-time setup and `knowledge-base-math/startup.sh` for per-session start (run from inside `knowledge-base-math/`; GPU check → auth warning → starts Ollama if not already running → `python app.py`). `startup.sh --allow-cpu` runs it locally without CUDA.

There is no automated test suite/linter configured — `test_chat.py` is a manual smoke test, not a pytest suite.

## Architecture

Pipeline: PDF → `extract.py` → `.mmd` → `ingest.py` → per-user BM25 + Chroma indexes → `api/` (or `query.py`) → hybrid retrieval + RRF → cross-encoder rerank → LLM answer.

Serving is split: **`api/` is the deployable unit**; `app.py` and the future TypeScript
frontend are both just HTTP clients of it. See `DEPLOYMENT.md` for hosting, cost, and env vars.

- **extract.py** — PDF → `.mmd` (Markdown+LaTeX). Samples the first few pages to decide the extractor: `marker-pdf` for math-heavy PDFs (proper Markdown+LaTeX via the Surya models), falling back to `pymupdf4llm` on Marker failure or for non-math PDFs. Marker downloads its models to the HuggingFace cache on first run (GPU-fast, CPU-slow-but-correct). Output goes to `docs/extracted/`.
  - **⚠️ DO NOT detect math by grepping for LaTeX (`\frac`, `$$`, `\int`, …) in `pymupdf4llm` output — this mistake has been made repeatedly.** `pymupdf4llm` output contains **NO LaTeX whatsoever**; it renders equations as Unicode glyphs (`γ`, `≡`, `∫`, `∂`, sub/superscripts). Any LaTeX-pattern regex over a pymupdf sample matches nothing, so it silently classifies every math PDF as "not math" and routes it to the wrong (pymupdf) extractor. Math detection on a pymupdf sample must key off **Unicode math glyphs** (or a real layout/OCR signal) — never LaTeX tokens. LaTeX only exists downstream, in Marker's output.
- **chunking.py** — the three splitting strategies (`baseline` = the old `RecursiveCharacterTextSplitter` 400/80; `eqaware` = never split inside `$$…$$`/`$…$`; `eqaware_context` = eqaware plus the sentence before/after each equation) and `assign_chunk_ids`. The equation-blind baseline will cut a `$$…$$` in half; the eqaware variants keep equations atomic (the overlap seed is equation-aware too). Chunk ids are stable within a strategy but **not comparable across strategies** — eval matches by content overlap for that reason.
- **latex_norm.py** — `normalize_latex(text)` via pylatexenc, rendering LaTeX to readable glyphs/words (`\theta`→`θ`, `\cos`→`cos`, `\frac{a}{b}`→`a/b`). Used **only** to compute embeddings (see `retrieval.NormalizingEmbeddings`); `page_content` stays raw LaTeX so BM25, the reranker, LLM context, and eval overlap-matching all see the original.
- **ingest.py** — chunks `.mmd` files via `chunking.CHUNKERS[--chunker]` (default `baseline`) and builds two indexes per user: a BM25 pickle (`bm25_indexes/user_<name>.pkl`, `{"bm25": BM25Okapi, "chunks": [...]}`) and a Chroma collection (`chroma_db/`, collection `user_<name>`). Index paths are **imported from `retrieval.py`** (which re-exports them from `config.py`), not redefined here — a second local copy of those constants would mean ingest writing where retrieval never reads. `--embed-model` / `--normalize-latex` select the embedder via `retrieval.load_embeddings` (no more duplicate model constant); returns build stats for the sweep.
- **retrieval.py** — **the retrieval pipeline, and the single source of truth for it.** BM25 top-10 + dense top-10 (`BAAI/bge-small-en-v1.5`) merged via Reciprocal Rank Fusion (RRF, k=60); the RRF top-20 candidate pool is rescored by a cross-encoder (`BAAI/bge-reranker-v2-m3`) and the top 5 go to the LLM. Imported by `query.py`, `app.py`, and `eval.py` — change retrieval behaviour here and nowhere else. Tunables live here too (`TOP_K`, `RERANK_TOP_C`, `TOP_N`, `RRF_K`). `load_embeddings(model_name, normalize_latex)` is the one place embedders are built (query and ingest share it, so query normalization always matches the index); `NormalizingEmbeddings` wraps a base embedder to LaTeX-normalize text before encoding while leaving `page_content` raw.
  - `retrieve()` returns the final top-N. `retrieve_detailed()` additionally returns the pre-rerank candidate pool, the full ranked list, and per-stage timings — that's what `eval.py` needs.
  - **`top_k` bounds the candidate pool**: RRF can emit at most `top_k × 2` candidates, so raising `RERANK_TOP_C` above that is a silent no-op. Raise `top_k` too.
- **query.py** — interactive CLI around `retrieval.py` + DeepSeek-Math generation. `--no-rerank` / `--no-bm25` / `--no-dense` toggle stages for A/B comparison.
- **api/** — the FastAPI service, and **the deployable unit**. Imports the pipeline from `retrieval.py`; never reimplements it. Loads embeddings/reranker/LLM once at startup.
  - `api/main.py` app + lifespan, `api/routes.py` endpoints, `api/schemas.py` the wire contract (what the TS frontend is generated from), `api/chat.py` prompts + history + provenance, `api/deps.py` model singletons + auth, `api/settings.py` env config.
  - `POST /chat` streams **SSE**: one `sources` frame, then `token` frames, then `done` (carrying `mode`, timings, and the telemetry `event_id`). `POST /upload` returns a job handle polled via `GET /jobs/{id}` — Marker ingest is minutes long. Also `GET /users/{u}/status`, `POST /feedback`, `GET /healthz`.
  - **Answer provenance is a server fact, not a model claim.** `chat.decide_mode` returns `grounded` (retrieval produced a chunk scoring ≥ `KBM_RELEVANCE_FLOOR`) or `general`, and in `general` mode the server *prepends* the "not from your documents" marker itself. Asking deepseek-math to emit that line was measured and it ignores the instruction — it is a solver, not an instruction-follower. This is also what makes faithfulness measurable at all: without the label, a grounded answer and a hallucination are indistinguishable downstream.
  - ⚠️ `KBM_RELEVANCE_FLOOR` is on a **sigmoid (0–1) scale**, not raw logits: `sentence_transformers.CrossEncoder` applies the model's activation and bge-reranker-v2-m3 carries a Sigmoid. Setting it to a logit-like `0.0` makes every score pass and abstention impossible.
- **app.py** — Gradio 6.18 web UI (port 7860), now an **HTTP client of `api/`**, not an importer of the pipeline. Kept because a working UI on the real API proves the API is complete before any TypeScript exists; delete it when the TS frontend lands rather than porting it. Keeps a separate "clean" conversation history (sources stripped) that it sends back as context, distinct from the display history.
- **telemetry.py** — append-only JSONL event log (`$DATA_DIR/telemetry/events.jsonl`): one `query` record per answer (hashed user, mode, retrieved chunk ids/scores, per-stage timings) plus `feedback` records keyed by `event_id`. Exists to be *already collecting* by the time it's needed: logged questions are the real query distribution that `EVALUATION.md §8` names as the gap synthetic gold sets cannot close, and thumbs-up (question, chunk) pairs are the contrastive data for an embedding fine-tune (`EVALUATION.md §10.9`). Never let a telemetry failure break a query.
- **ops/idle_stop.py** — watchdog that stops the pod after `KBM_IDLE_STOP_MINUTES` without a `/chat`. This is the entire cost model: on-demand ~$20/mo vs ~$115/mo always-on. Defers while an ingest job is active.
- **evaluation/** — the whole eval harness lives here (scripts, gold set, and a `results/` subfolder for run outputs). The scripts add the parent dir to `sys.path` and use paths relative to `knowledge-base-math/`, so **run them from `knowledge-base-math/`** (e.g. `python evaluation/eval.py …`). Only `evaluation/goldset.jsonl` and the scripts are tracked; `evaluation/results/`, `goldset.raw.jsonl`, and `goldset_review.md` are gitignored.
- **evaluation/make_evalset.py** — builds `evaluation/goldset.jsonl` (question → gold `chunk_id`) by sampling chunks (biased toward math-bearing ones) and having an *instruct* model (`qwen2:7b`, **not** deepseek-math, which is a solver) write the question each chunk answers. Output is a **draft**: questions that parrot chunk wording flatter BM25 and must be rewritten by hand.
- **evaluation/eval.py** — runs the gold set against retrieval configs (`bm25`, `dense`, `hybrid`, `hybrid+rerank`, pool variants) and reports recall@1/@5, **recall@pool**, MRR, nDCG@5, per-stage latency, peak VRAM, and optionally LLM-judged answer quality (`--answers`); writes to `evaluation/results/`. **recall@pool** is the key diagnostic: the reranker can only reorder what BM25/dense already found, so if the gold chunk isn't in the pool the miss is upstream (extraction/chunking), not the reranker's fault.
  - `--match` selects how a retrieval counts as the gold hit: `id` (exact chunk id, same-chunking only) or `overlap` (default; token-containment ≥0.70 against the gold `chunk_text`, so one frozen gold set scores any chunking). `--embed-model`/`--normalize-latex` must match how the target index was built.
- **evaluation/embed_chunk_sweep.py** — the chunking×embedding sweep: builds 9 indexes (3 chunkers × {bge-small, bge-m3, bge-m3+pylatexenc}) from the gold set's source doc into `sweep_<chunk>_<embed>` users, evals each at dense-only + hybrid+rerank via overlap matching, and writes a quality+latency table to `evaluation/results/sweep_results.json`. This is how the "same embedder for LaTeX and prose?" question is decided by measurement.
- **evaluation/eval.sh** — pod runner that wraps the GPU eval: CUDA check → data check → **prefetch all models before any testing** → the sweep, with an optional `--answers` run. See `SETUP.md § GPU eval run`.
- **evaluation/EVALUATION.md** — the evaluation protocol: what's measured, why, the gold-set caveats, cost expectations, and how to act on the numbers. **Read before changing retrieval.**
- **LATENCY.md** — where query time actually goes (generation is ~95%, retrieval ~4%), the fixes applied, and the measurements behind them. **Read before editing `SYSTEM_PROMPT` in `app.py`/`query.py`:** the prompt order is load-bearing — static text → history → context → question — because Ollama only reuses the KV cache of a common prompt *prefix*. Putting per-query context before stable content silently re-prefills the whole prompt every turn.
- **prefetch_models.py** — downloads every HuggingFace model the app needs, before anyone uses it. Exists because the three model sets are fetched at wildly different moments: the embedder (~130MB) and reranker (~2.2GB) load at `app.py` *import*, so a cold cache just makes startup look hung — but **Marker/surya (~3-4GB) is built inside `extract_marker()` and therefore does not download until the first PDF upload**, where `handle_upload`'s `except` reports the stall as a generic `Ingestion failed: <e>`. Fetches through the pipeline's own loaders, so it pulls exactly the files that will be used and doubles as a "do these load here?" check. Called by `startup.sh` (stage 4, skippable with `--no-prefetch`) and by `evaluation/eval.sh` with `--skip-marker` — **one prefetch implementation; don't reintroduce an inline copy in a shell script.**
- **test_chat.py** — ingests a fixed test doc for a `test` user and drops into the same interactive CLI loop as `query.py`, for manual end-to-end verification.

Note: retrieval logic used to be duplicated between `query.py` and `app.py`. It is now shared via `retrieval.py` — **do not reintroduce a second copy.** An `eval.py` that measures a reimplementation of the pipeline rather than the pipeline itself will drift from what ships and quietly start lying.

The same trap caught the index paths: `CHROMA_DIR`/`BM25_DIR` were declared in *both* `retrieval.py` and `ingest.py`, so the reader and the writer could disagree about where the data lived. They now come from `config.py` alone. `test_chat.py` had a third copy of the problem — it built its own `HuggingFaceEmbeddings` instead of calling `load_embeddings`, so the smoke test would have exercised a different loader (and different device) than the app it was meant to smoke-test. **When you add a path, a model, or a host, define it once and import it.**

## Model & infra

- LLM: `t1c/deepseek-math-7b-rl:Q4` served via Ollama.
- Embeddings: `BAAI/bge-small-en-v1.5` (HuggingFace, normalized).
- Reranker: `BAAI/bge-reranker-v2-m3` cross-encoder via `sentence_transformers.CrossEncoder` (~2.2GB, downloads to the HF cache on first use — keep the HF cache on `/workspace` on the pod so it survives restarts).
- Eval-only model: `qwen2:7b` (Ollama) — used by `make_evalset.py` as the *instruct* question-writer and by `eval.py` as the LLM-as-judge. Deliberately **not** deepseek-math (a solver, not a writer; and a model must not grade its own output). Only needed when running the eval harness, not the serving pipeline — `ollama pull qwen2:7b`.
- Designed to run on a RunPod GPU pod; everything must survive on `/workspace` (persistent volume) between pod restarts — `OLLAMA_MODELS`, `HF_HOME`, and **`DATA_DIR`**. The first two cost a ~10GB re-download if missed; `DATA_DIR` is the one that loses the family's uploaded documents outright, because the container filesystem is wiped on restart and the indexes default to the current directory.

## Conventions / constraints

- Prioritize free/open-source models and infra when suggesting changes (Ollama, HuggingFace-hosted models, open embedding models). Do not recommend paid APIs (OpenAI, Anthropic) as the generation backbone.
- Math retrieval quality (notation, proofs, formulas) is a primary concern — consider math-aware chunking/embeddings over generic RAG defaults.
- User wants the option to fine-tune models themselves (embedding and/or generation LLM) down the line.
- Multi-user isolation is per-username directory/collection naming, not real auth — keep this in mind before suggesting security-sensitive changes.
- `chroma_db/`, `bm25_indexes/`, PDFs, and extracted `.mmd` files under `docs/raw`/`docs/extracted` are all gitignored (private user document data) — don't try to commit them.
