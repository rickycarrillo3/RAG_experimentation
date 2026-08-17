> # ⚠️ SUPERSEDED — historical record, not guidance
>
> This plan was **implemented** in commit `1571a24` ("Add cross-encoder reranking
> stage"). It is kept for the reasoning in its Context section, which still explains
> *why* the reranker exists better than anything else in the repo.
>
> **Do not follow its "Decisions" section.** It specifies duplicating the rerank logic
> across `query.py` and `app.py` to match a then-current convention. That convention is
> gone: retrieval was consolidated into `retrieval.py` in `31354dd`, and `CLAUDE.md` now
> states plainly **"do not reintroduce a second copy."** Following this file as written
> would undo that consolidation and re-create the drift between the CLI, the web UI, and
> the eval that `retrieval.py` exists to prevent.
>
> For how retrieval actually works today, read `CLAUDE.md` § Architecture and
> `knowledge-base-math/retrieval.py`.

# Plan: Add a cross-encoder reranking stage to the math RAG pipeline

## Context

Today retrieval is **hybrid** — BM25 (sparse) top-10 + bge-small (dense) top-10, merged
with Reciprocal Rank Fusion, and the RRF top-5 go straight to the LLM
(`query.py:89-112`, `app.py:94-104`). RRF only ever uses **ordinal rank position**
(`scores[key] += 1/(k+rank)`), so it discards how relevant each chunk actually is, and
both of its inputs are weak judges: BM25 is bag-of-words (blind to `∫` vs "integral"),
and bge-small is a **bi-encoder** that embeds query and chunk *separately*, capturing
topical gist but blurring precise distinctions.

The failure mode this causes in math QA is retrieving the *topically* right but
*specifically* wrong chunk — the general section on integration instead of the worked
example matching the student's exact problem.

### Current (hybrid) vs. reranking — the difference

- **Hybrid/RRF (now):** two cheap retrievers each rank candidates independently; RRF
  fuses them by rank arithmetic. No model ever encodes the query and a chunk
  **together**: dense retrieval compares two embeddings each produced in isolation, and
  BM25 just counts word overlap — nothing attends across the query–chunk pair. Fast, no
  joint query-time model, but coarse.
- **Reranking (added):** a **cross-encoder** takes `(query, chunk)` **together** in one
  forward pass and outputs a true relevance score, attending across the pair. Far more
  accurate, but too expensive to run on the whole corpus — so it only rescores a
  shortlist that hybrid retrieval already narrowed down.

This is **additive**: hybrid retrieval stays exactly as-is and does the "cast a wide
net" job; reranking is a new final stage that picks the best 5 from a wider candidate
pool. `sentence-transformers` is **already a dependency**, so no new package is needed.

```
BEFORE:  BM25 top-10 ┐                                   AFTER:  ... ┐
                     ├─ RRF ─→ top-5 ─→ LLM                          ├─ RRF ─→ top-20 ─→ cross-encoder ─→ top-5 ─→ LLM
         dense top-10┘                                          dense┘         (candidates)   rescores each
```

## Decisions (confirmed with user)

- **Model:** `BAAI/bge-reranker-v2-m3` (via `sentence_transformers.CrossEncoder`). Same
  BAAI family as the bge-small embedder; GPU pod makes its size fine.
- **Structure:** duplicate the rerank logic in both `query.py` and `app.py`, matching the
  repo's existing RRF duplication convention — **and** add a note to `CLAUDE.md`
  recommending the duplication be extracted into a shared `retrieval.py` module long-term.

## Changes

### 1. `knowledge-base-math/query.py`

- **Import:** `from sentence_transformers import CrossEncoder`.
- **Constants** (near lines 29-31): add
  - `RERANK_MODEL = "BAAI/bge-reranker-v2-m3"`
  - `RERANK_TOP_C = 20` — candidate pool size taken from RRF and fed to the reranker.
  - Keep `TOP_N = 5` as the final count to the LLM.
- **New `load_reranker()`** → returns `CrossEncoder(RERANK_MODEL)`.
- **New `rerank(query, candidates, reranker, top_n=TOP_N)`**: takes the RRF output
  (`list[tuple[Document, float]]`), builds `[(query, doc.page_content), ...]` pairs, calls
  `reranker.predict(pairs)`, sorts by score desc, returns the top `top_n` as
  `list[tuple[Document, float]]` where the float is now the **rerank score**.
- **Modify `retrieve()`** (lines 89-112): add params `reranker=None, use_rerank=True`.
  RRF still produces the full merged list; if `use_rerank and reranker is not None`, take
  `merged[:RERANK_TOP_C]` and pass through `rerank(...)`; otherwise fall back to
  `merged[:TOP_N]` (unchanged behaviour). Reranking also helps the single-retriever
  `--no-bm25`/`--no-dense` cases, so it stays enabled there.
- **CLI (`main`)**: add a `--no-rerank` flag mirroring the existing `--no-bm25`/`--no-dense`
  comparison flags; load the reranker once before the loop unless `--no-rerank`; pass
  `reranker`/`use_rerank` into `retrieve()`. Update the `Ready [...]` status line to show
  rerank on/off.
- **`print_retrieved`** (lines 135-143): relabel the per-chunk score line so it reads as a
  generic score (rerank score when reranking, RRF otherwise) rather than hard-coded `RRF=`.

### 2. `knowledge-base-math/app.py`

Mirror the above, matching app.py's module-level pattern:
- Import `CrossEncoder`; add `RERANK_MODEL` / `RERANK_TOP_C` constants (near lines 33-35).
- Load the reranker **once at module level** next to `embeddings`/`llm` (lines 54-59):
  `reranker = CrossEncoder(RERANK_MODEL)`.
- Add a `_rerank(query, candidates, top_n=TOP_N)` helper matching `query.py`'s `rerank`.
- **Modify `retrieve()`** (lines 94-104): after `_rrf(...)`, take `merged[:RERANK_TOP_C]`
  and return `_rerank(query, that, TOP_N)` instead of `merged[:TOP_N]`. Always on in the UI
  (no flag).

### 3. `knowledge-base-math/CLAUDE.md`

- In **Architecture**, update the `query.py` and `app.py` bullets to mention the new
  cross-encoder rerank stage (RRF top-20 → rerank → top-5).
- Extend the existing "retrieval/RRF logic is duplicated between query.py and app.py"
  **Note** to include the reranker in the list of things to keep in sync, **and** add a
  recommendation: this duplication has now grown to three stages (search, RRF, rerank) and
  should be extracted into a shared `retrieval.py` module imported by both files.
- Add `RERANK_MODEL` / `RERANK_TOP_C` to the list of retrieval params to keep in sync.

### 4. `requirements.txt` — no change

`sentence-transformers` is already listed. `bge-reranker-v2-m3` (~2.2GB) downloads from
HuggingFace to the HF cache on first use.

## Operational note (RunPod)

First run downloads the reranker to the HF cache. On the pod, ensure the HF cache lives on
the persistent `/workspace` volume (like `OLLAMA_MODELS` per `SETUP.md`) so it survives pod
restarts — e.g. `HF_HOME=/workspace/.cache/huggingface`. Flag this in `SETUP.md` during
implementation if it isn't already set.

## Verification

```bash
cd knowledge-base-math && source venv/bin/activate

# Ensure a populated index exists (test.mmd ships in docs/extracted/)
python ingest.py --user test docs/extracted/test.mmd

# A/B the ordering: reranked vs. RRF-only on the same query
python query.py --user test --retrieval-only                # reranked order + rerank scores
python query.py --user test --retrieval-only --no-rerank    # original RRF order — expect a visibly different top-5

# Full pipeline still answers correctly
python query.py --user test        # ask a math question from the test doc

# Web UI: reranker loads once at startup, upload + chat works without errors
python app.py                      # http://localhost:7860
```

Success criteria: reranking runs without error, the `--no-rerank` vs default top-5 differ
in order/membership on at least one query (proving the stage is active), and the full
pipeline + web UI still produce answers.
