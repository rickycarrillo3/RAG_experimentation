# Evaluation Protocol

How we measure whether a change to the RAG pipeline actually helped, instead of squinting at
the top-5 and deciding it "looks better."

This document explains the protocol: what the pipeline is, what we measure, why those metrics,
what the numbers *don't* tell you, and what it costs to run.

---

## 1. Why this exists

Every change to retrieval so far — hybrid search, RRF, and most recently the cross-encoder
reranker — has been judged by eye. That has two problems:

1. **It cannot detect a regression.** A change that helps three queries and quietly breaks ten
   looks fine if you only try three.
2. **It cannot justify a cost.** `bge-reranker-v2-m3` costs 2.2GB of VRAM and disk and adds a
   model to the query path. Nobody can currently say what it buys.

The eval turns both questions into numbers. It is deliberately built so that **"the reranker
didn't help" is a possible, publishable outcome** — if the harness can't tell you to remove
something, it isn't measuring, it's cheerleading.

---

## 2. The pipeline under test

**Ingest (offline, per document)**

```
PDF ─→ extract.py ─→ .mmd (Markdown + LaTeX) ─→ ingest.py ─→ ┬─ BM25 index  (bm25_indexes/user_<name>.pkl)
       └─ Marker (Surya models) if math                       └─ Chroma      (chroma_db/, collection user_<name>)
          pymupdf4llm otherwise / on failure                     chunk 400 chars, overlap 80
```

**Query (online)**

```
query ─┬─→ BM25            top-10 ─┐
       │                            ├─→ RRF (k=60) ─→ top-20 ─→ cross-encoder ─→ top-5 ─→ DeepSeek-Math-7B ─→ answer
       └─→ bge-small dense top-10 ─┘   ordinal only    pool     bge-reranker-v2-m3
```

Three stages, three different jobs:

- **BM25** — lexical. Matches literal tokens. Blind to the fact that `∫` and "integral" are the
  same idea.
- **bge-small (dense)** — a **bi-encoder**: it embeds the query and each chunk *separately* and
  compares vectors. Captures topical gist, blurs precise distinctions. It never sees the query
  and the chunk together.
- **RRF** — fuses the two ranked lists using **rank position only** (`score += 1/(60 + rank)`).
  It throws away *how* relevant each retriever thought a chunk was. It is a tiebreaker, not a
  judge.
- **Cross-encoder reranker** — the new stage. Reads `(query, chunk)` **together** in one forward
  pass, attending across the pair, and emits a true relevance score. Far more discriminating,
  but O(candidates), so it never touches the corpus — only the 20 chunks RRF already shortlisted.

The failure mode reranking is *supposed* to fix is the topically-right-but-specifically-wrong
chunk: retrieving the general section on integration when the student needs the worked example
matching their actual problem. **The eval must be able to see that distinction, or it's useless.**

All of this lives in one place — `retrieval.py` — imported by `query.py`, `app.py`, and
`eval.py`. That is load-bearing: an eval that measures a *reimplementation* of the pipeline
rather than the pipeline itself will drift from what you ship and quietly start lying.

---

## 3. The gold set

**`eval/goldset.jsonl` — 50 records of `question → the chunk that should be retrieved`.**

Built by `make_evalset.py`:

1. Sample ~50 chunks from a user's index, **biased ~70% toward math-bearing chunks** (those
   containing `$$`, `\frac`, `\int`, …). A gold set of pure prose would look healthy while
   telling you nothing about the case the system exists to serve.
2. For each chunk, ask an **instruct** model (`qwen2:7b`) to write the question a student would
   ask that this chunk answers. The source chunk is the gold label.
3. Chunks carry a deterministic `chunk_id` (`<source>::<n>`, stamped by `ingest.py`), so labels
   survive re-ingestion.

We use an instruct model rather than `deepseek-math-7b-rl` on purpose: DeepSeek-Math is a
*solver*, not an instruction-follower, and writes poor questions.

### The step you must not skip

**The generated file is a draft. Read it.** Delete or rewrite every question that parrots the
chunk's wording.

This is the protocol's central weakness and it should be stated plainly: questions generated
*from* a chunk inherit that chunk's vocabulary. A question that reuses the chunk's rare terms is
trivially findable by **BM25** — which will make lexical retrieval look far better than it is,
and make the reranker look useless by comparison. Real family questions ("how do I do the one
with the squiggly line") share almost no vocabulary with the target chunk.

Two consequences, and they set how you're allowed to read the output:

- **Absolute scores are optimistic.** Do not quote recall@5 as "the system's accuracy."
- **Deltas between configs are the signal**, because every config is handicapped identically.

The 20 minutes spent rewriting questions into natural student phrasing is what makes the
absolute numbers mean anything. If you skip it, only trust the deltas.

---

## 4. What is measured

### Retrieval quality (no LLM — cheap, deterministic, run this constantly)

| Metric | Question it answers |
|---|---|
| **recall@1** | Was the gold chunk the single best hit? |
| **recall@5** | Did the gold chunk reach the LLM at all? (top-5 is what gets sent) |
| **recall@pool** | Was the gold chunk in the candidate pool the reranker even saw? |
| **MRR** | How high did it rank? (1/rank, averaged) |
| **nDCG@5** | Rank-sensitive: catches "found it, but demoted it to #5" |

**recall@5 is the headline** — it is literally "did the model get the information it needed."

**recall@pool is the diagnostic, and it's the one people forget.** The reranker can only reorder
what BM25 and dense retrieval already found. If the gold chunk isn't in the pool, reranking
*cannot* fix it, and a disappointing recall@5 is not the reranker's fault — the miss happened
upstream in retrieval or chunking. Reading recall@5 without recall@pool leads directly to
blaming (or crediting) the wrong stage.

- `recall@pool` high, `recall@5` low → **the reranker is the problem.** It's demoting good chunks.
- `recall@pool` low → **retrieval/chunking is the ceiling.** Fix extraction and chunking; a better
  reranker is wasted money. Consider raising `top_k` to widen the pool.

### Configs swept

| Config | What it isolates |
|---|---|
| `bm25` | lexical baseline |
| `dense` | bi-encoder baseline |
| `hybrid` | RRF, **no reranker** — *the number to beat* |
| `hybrid+rerank` | the system as it ships today |
| `rerank_pool10` / `rerank_pool50` | is a 20-candidate pool the right size? |

Note `top_k` bounds everything: RRF can only emit up to `top_k × 2` candidates, so raising
`rerank_top_c` beyond that does nothing. The pool variants raise `top_k` accordingly — this is
easy to get wrong and produces a "no effect" result that is really a no-op.

### Answer quality (end-to-end, `--answers`)

Generates the real answer through the real pipeline, then has a **judge model** (`qwen2:7b`)
score it 1–5 for factual correctness against the gold chunk.

**A 7B judge is a noisy instrument.** Use it to catch large regressions, not to split hairs — a
4.1 vs 4.2 is meaningless. It is deliberately **not** the generator: a model grading its own
output is not evidence. Retrieval metrics are the reliable signal here; treat the judge as a
smoke alarm.

### Latency and resources

Per-stage wall-clock (BM25 / dense / RRF / rerank / generation) and peak VRAM, so the reranker's
cost is measured rather than assumed.

---

## 5. Running it

```bash
cd knowledge-base-math && source venv/bin/activate

# 1. Ingest a document (stamps chunk_ids)
python ingest.py --user test docs/extracted/textbook.mmd

# 2. Build the gold set — THEN READ AND CLEAN IT (see §3)
python make_evalset.py --user test --n 50

# 3. Measure
python eval.py --user test --all              # full sweep + verdict table
python eval.py --user test --all --answers    # also score answers end-to-end (slow)
```

Results land in `eval/results_<config>.json`, plus a printed comparison table and an explicit
reranker verdict.

**Keep the gold set fixed** while comparing configs. Regenerating it between runs changes the
exam, not the student — and any comparison across different gold sets is meaningless.

---

## 6. Cost expectations (RTX 4090 / A5000, 24GB)

### Models

| Model | Role | When | Disk | VRAM |
|---|---|---|---|---|
| Marker / Surya | PDF → LaTeX | ingest | ~3–4 GB | ~3–5 GB peak |
| `bge-small-en-v1.5` (33M) | dense embed | ingest + query | ~130 MB | ~0.5 GB |
| `bge-reranker-v2-m3` (568M) | cross-encoder | query | ~2.2 GB | ~2.2 GB fp32 / ~1.2 GB fp16 |
| `deepseek-math-7b-rl:Q4` | generation | query | ~4.5 GB | ~5–5.5 GB (incl. KV cache) |
| `qwen2:7b` | question-gen + judge | eval only | ~4.4 GB | ~5 GB |

- **Steady-state query VRAM ≈ 8 GB.** Comfortable on 24GB.
- **Peak during a web-UI upload ≈ 12–13 GB** — `app.py` holds the reranker at module scope while
  Ollama holds the LLM and Marker spins up on top. Fits, but it's the tightest moment. If it ever
  OOMs, set Ollama `keep_alive: 0` during ingest.
- **Total model disk ≈ 15 GB.** Put `HF_HOME=/workspace/.cache/huggingface` on the persistent
  volume next to `OLLAMA_MODELS` — otherwise the 2.2GB reranker and 3–4GB of Surya models
  **re-download on every pod restart**.
- Index storage is a rounding error: a 300-page textbook ≈ 2k chunks ≈ a few MB across Chroma
  and the BM25 pickle.

### Latency, per query

| Stage | Cost |
|---|---|
| BM25 | ~1–10 ms |
| Query embed + Chroma | ~10–25 ms |
| RRF | <1 ms |
| **Rerank (20 pairs)** | **~50–150 ms** (one batched forward pass; fp16 roughly halves it) |
| **Generation (~300 tokens)** | **~4–6 s** — utterly dominant |

**The reranker adds ~2–3% to query latency.** Its real cost is the 2.2GB, not the milliseconds.
So the question the eval must answer is never "is it fast enough" — it's **"does it move recall@5
enough to justify carrying the model at all."**

### Run time

| Job | Cost |
|---|---|
| Gold set generation (50 q) | ~2–5 min |
| Retrieval sweep, all 6 configs | **< 1 min** — no LLM in the loop; this is the loop you iterate on |
| `--answers`, 2 configs | ~15 min (50 × [~5s generate + ~3s judge] × 2) |
| Marker ingest | ~0.3–1 s/page on GPU → a 300-page textbook in ~3–10 min. Measure it; math-dense pages trigger far more equation OCR. (For scale: ~11 min for **6 pages** on a Mac CPU.) |

---

## 7. First baseline run

50 questions, `sample.mmd` (arXiv Double-DQN paper, 219 chunks), gold set **not yet hand-cleaned**,
run on a Mac CPU. Treat as a shakedown of the harness, not the system's real scores.

| config | R@1 | R@5 | R@pool | MRR | nDCG@5 | retrieval ms |
|---|---|---|---|---|---|---|
| bm25 | 0.38 | 0.60 | 0.62 | 0.469 | 0.499 | 0.7 |
| dense | 0.62 | 0.76 | 0.84 | 0.685 | 0.696 | 70.8 |
| hybrid | 0.48 | 0.74 | 0.84 | 0.587 | 0.615 | 16.3 |
| **hybrid+rerank** | 0.64 | **0.84** | 0.84 | **0.730** | 0.758 | 820.7 |
| rerank_pool10 | 0.66 | 0.80 | 0.80 | 0.727 | 0.746 | 527.8 |
| **rerank_pool50** | 0.68 | **0.90** | **0.96** | **0.781** | 0.807 | 3373.7 |

Three things fall out of this, and two of them were not expected:

1. **The reranker pays for itself.** recall@5 `0.74 → 0.84`, MRR `0.587 → 0.730`. It is doing
   exactly the job it was added for: the gold chunk was already in the pool (R@pool is `0.84`
   either way) and reranking *promotes it into the top-5*. Keep the model.

2. **BM25 is actively hurting hybrid.** `dense` alone (R@5 `0.76`, MRR `0.685`) **beats** `hybrid`
   (`0.74`, `0.587`). RRF is mixing a good ranker with a bad one and getting something worse than
   the good one. Worth testing a weighted fusion, or dropping BM25 — but see the caveat below
   before acting: an un-cleaned gold set is supposed to *flatter* BM25, and BM25 still lost.

3. **The candidate pool is the real bottleneck, not the reranker.** `rerank_pool50` (`top_k=25`)
   lifts R@pool `0.84 → 0.96` and R@5 to `0.90` — the single biggest win in the table. The default
   `top_k=10` is starving the reranker: ~16% of gold chunks never reach it. Raising `top_k` costs
   only retrieval time, which is noise next to generation.

**Latency here is CPU-bound and not representative** — 780ms of reranking becomes ~50–150ms on the
GPU pod. The interesting cost is that `rerank_pool50` reranks 2.5× more pairs, which stays cheap
relative to a ~5s generation.

**Caveat, load-bearing:** this gold set has *not* been hand-cleaned, and spot-checking shows the
leakage §3 predicts — questions like "What are the average scores for each game in this benchmark?"
refer to the chunk rather than standing alone. Clean the gold set before treating any of these
absolute numbers as the system's accuracy.

---

## 8. What this protocol does *not* tell you

Worth being honest about, so the numbers aren't over-read:

- **Extraction quality.** If Marker mangled a formula on the way in, no retrieval metric will
  notice — the gold chunk is mangled too, and retrieval happily "succeeds." Extraction bounds
  everything downstream and is measured nowhere here.
- **Chunking quality.** A gold chunk that is half a proof is still a gold chunk. Recall@5 can look
  perfect while the LLM receives a formula severed from the sentence defining its symbols.
- **Whether the answer actually helps a human.** The judge scores factual agreement with a
  passage, not whether a 14-year-old understood the explanation.
- **Real query distribution.** LLM-generated questions are not how your family talks (§3).

Each of these is a reason the eval is a *floor*, not a verdict on the product.

---

## 9. How to act on the results

- **`hybrid+rerank` beats `hybrid` on recall@5 / MRR** → the reranker earns its 2.2GB. Keep it.
- **No meaningful delta** → drop the reranker. That is a real result, not a failure, and it
  redirects effort to the likelier win: **math-aware chunking** (ROADMAP §3).
- **`recall@pool` is low across the board** → stop tuning retrieval. The ceiling is upstream:
  extraction (ROADMAP §2) and chunking (§3).
- **`bm25` ≈ `hybrid`** → suspect the gold set (§3): the questions are probably parroting chunk
  vocabulary. Rewrite them before trusting anything else in the table.

The eval is also the training data for the embedding fine-tune (ROADMAP §6) — a contrastive
fine-tune of `bge-small` needs exactly these `(question, correct chunk)` pairs.
