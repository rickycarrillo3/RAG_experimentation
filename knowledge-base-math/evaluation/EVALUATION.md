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

All of this lives in one place — `kbm/retrieval.py` — imported by `query.py`, `app.py`, and
`eval.py`. That is load-bearing: an eval that measures a *reimplementation* of the pipeline
rather than the pipeline itself will drift from what you ship and quietly start lying.

---

## 3. The gold set

**`evaluation/goldset.jsonl` — records of `question → the chunk that should be retrieved`.**

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

### Leakage — the central weakness, and the two defences against it

Questions generated *from* a chunk inherit that chunk's vocabulary. A question that reuses the
chunk's rare terms is trivially findable by **BM25** — which makes lexical retrieval look far
better than it is, and makes the reranker look useless by comparison. Real family questions
("how do I do the one with the squiggly line") share almost no vocabulary with the target chunk.

This is not hypothetical. The **first, unfiltered** gold set measured:

| | v1 (unfiltered) |
|---|---|
| mean question→chunk word overlap | **0.43** |
| questions with ≥60% overlap | **18/50** |
| chunk-referential ("this benchmark", "as shown in the middle plots") | **10/50** |

Its worst entry — *"What do the straight orange lines in the top row of plots represent?"* — was a
**figure-caption question that no retrieval system should be expected to answer.** It was scored
as a fair test.

`make_evalset.py` now defends on two fronts, because they catch different failures:

**1. An eligibility gate on the *chunk* (`is_answerable`).** Some chunks cannot yield a fair
question at all: figure captions, reference lists, bare number tables, symbol soup. These are
rejected before generation. No prompt can rescue a bad target — the "orange lines" question was
unfixable because the *chunk* was never a legitimate answer to anything.

**2. A leakage filter on the *question*, with retries.** Each generated question is scored:

```
leak_score = |question words ∩ chunk words| / |question words|     (stopwords stripped)
```

Rejected if `leak_score ≥ 0.6`, or if it matches a referential-phrase regex ("this passage",
"as shown", "the top row"). On rejection it is **regenerated** (up to 3 attempts) with a stricter
prompt that quotes the offending attempt back. Still failing → the chunk is dropped.

Each record carries its `leak_score`, `model`, and `prompt_version`, so a future run can tell
whether a score moved because the *system* changed or because the *exam* changed.

### The step you still must not skip

**The filtered file is still a draft.** `make_evalset.py` writes `evaluation/goldset_review.md` —
every question with its chunk, **sorted worst-leakage-first**. Read it and rewrite anything that
still reads like a lookup key rather than a student's question. Twenty minutes, targeted at the
worst cases.

Two consequences, and they set how you're allowed to read the output:

- **Absolute scores are optimistic** until that pass is done. Don't quote recall@5 as "the
  system's accuracy."
- **Deltas between configs are the signal**, because every config sits the same exam.

Every `eval.py` run prints a **GOLD SET HEALTH** header (n, mean leak score, high-leak count) so
a recall number can never be read without the quality of the exam beside it.

---

## 4. What is measured

### Retrieval quality (no LLM — cheap, deterministic, run this constantly)

| Metric | Question it answers |
|---|---|
| **recall@1** | Was the gold chunk the single best hit? |
| **recall@5** | Did the gold chunk reach the LLM at all? (top-5 is what gets sent) |
| **recall@5_soft** | Same, but an *adjacent* chunk (gold ± 1) also counts |
| **recall@pool** | Was the gold chunk in the candidate pool the reranker even saw? |
| **MRR** | How high did it rank? (1/rank, averaged) |
| **nDCG@5** | Rank-sensitive: catches "found it, but demoted it to #5" |

**recall@5 is the headline** — it is literally "did the model get the information it needed."

**recall@5_soft exists because chunks overlap.** At 400 chars with 80 overlap, an answer routinely
straddles two chunks. Under strict matching, a config that retrieves *the adjacent half of the
same worked example* scores a total miss — which understates recall and can rank configs wrongly.
Soft matching credits the neighbours.

Report both, and read the **gap** between them: a large `recall@5_soft − recall@5` means retrieval
is finding the right *region* and chunking is splitting the answer badly. That is a **chunking**
finding (see `kbm/chunking.py`'s `eqaware` strategies), not a retrieval one, and no reranker will fix it.

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
| `dense+rerank` | **does BM25 contribute anything once a reranker exists?** |
| `hybrid_bm25_lite` | down-weight BM25 in the fusion (RRF weights `[0.3, 1.0]`) rather than dropping it |
| `rerank_pool10` / `rerank_pool50` | is a 20-candidate pool the right size? |

Note `top_k` bounds everything: RRF can only emit up to `top_k × 2` candidates, so raising
`rerank_top_c` beyond that does nothing. The pool variants raise `top_k` accordingly — this is
easy to get wrong and produces a "no effect" result that is really a no-op.

`pool_size` is printed per config because it is **not comparable across configs**: a single-
retriever config has half the candidates of a hybrid one, so its `recall@pool` is measured over a
smaller pool.

### Failure dump

Every run writes `evaluation/results/failures_<config>.json`: each missed question, where the gold chunk
*actually* ranked (or that it was absent from the pool entirely), its leak score, and what was
retrieved instead. Aggregate metrics tell you **that** something is wrong; this tells you **what**.
Without it, every debugging round restarts from zero.

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
python evaluation/make_evalset.py --user test --n 50

# 3. Measure
python evaluation/eval.py --user test --all              # full sweep + verdict table
python evaluation/eval.py --user test --all --answers    # also score answers end-to-end (slow)
```

Results land in `evaluation/results/results_<config>.json`, plus a printed comparison table and an
explicit reranker verdict.

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

## 7. Baseline runs

### v1 — SUPERSEDED, kept as a cautionary record

50 questions, `sample.mmd` (arXiv Double-DQN paper, 219 chunks), **unfiltered** gold set, Mac CPU.

**Do not cite these numbers.** Two independent flaws, both since fixed:
- **Wrong corpus.** An ML research paper — prose, benchmark tables, figures. Almost no worked
  examples, theorems, or proofs. Not the retrieval problem this system exists to solve.
- **Leaky gold set** (mean leak 0.43; 18/50 above 0.60), which systematically *flatters BM25* and
  *understates the reranker*.

Kept because it shows what an un-audited eval looks like when it is confidently wrong.

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

### v2 — current baseline (calculus slice, clean gold set), Mac CPU

34 questions over a ~10-page slice of **OpenStax Calculus Vol 1** (§3.6 Chain Rule),
`calculus_chainrule.mmd`, **66 chunks**. Gold set auto-filtered by the §3 leakage defences:
**mean `leak_score` 0.15** (down from 0.43), no referential or figure-caption questions. Not yet
hand-cleaned, but already an honest exam — cite these, not v1. This is the **control** the GPU
parity run compares against.

| config | pool | R@1 | R@5 | R@5 soft | R@pool | MRR | nDCG@5 |
|---|---|---|---|---|---|---|---|
| bm25 | 10 | 0.24 | 0.56 | 0.68 | 0.71 | 0.37 | 0.40 |
| dense | 10 | 0.44 | 0.82 | 0.88 | 0.88 | 0.61 | 0.66 |
| hybrid | 16.5 | 0.44 | 0.82 | 0.91 | 0.94 | 0.60 | 0.64 |
| dense+rerank | 10 | 0.59 | 0.85 | 0.94 | 0.88 | 0.71 | 0.75 |
| **hybrid+rerank** | 16.5 | **0.59** | **0.88** | 0.94 | 0.94 | **0.72** | 0.76 |
| hybrid_bm25_lite | 16.5 | 0.59 | 0.88 | 0.94 | 0.94 | 0.72 | 0.76 |
| rerank_pool10 (top_k=5) | 8.6 | 0.56 | 0.91 | 0.94 | 0.91 | 0.70 | 0.75 |
| **rerank_pool50 (top_k=25)** | 36.2 | 0.59 | **0.91** | 0.94 | **0.97** | **0.74** | **0.78** |

Three findings, two of which **reverse or refine** what v1 (the leaky RL-paper eval) claimed:

1. **The reranker is confirmed.** hybrid → hybrid+rerank lifts R@1 `0.44 → 0.59`, R@5
   `0.82 → 0.88`, MRR `0.599 → 0.723`. Same story as v1, now on an honest exam. Keep the model.

2. **BM25 is vindicated — v1's "BM25 hurts" finding does NOT replicate.** With a reranker present,
   `hybrid+rerank` (R@5 `0.88`, R@pool `0.94`) **beats** `dense+rerank` (`0.85`, `0.88`): BM25 pulls
   gold chunks into the pool that dense alone misses (R@pool `0.94 > 0.88`), and the reranker then
   promotes them. v1 measured the opposite because a leaky gold set is exactly the case that should
   flatter BM25 — yet here, on the *clean* set, BM25 helps. Down-weighting it (`hybrid_bm25_lite`,
   RRF weights `[0.3, 1.0]`) is **identical** to full weight — no reason to touch the fusion.

3. **A chunking gap is now visible.** hybrid R@5 `0.82` vs R@5 **soft** `0.91` — a 9-point jump
   from counting the adjacent chunk as a hit. That gap is the answer being split across a
   400-char/80-overlap boundary, not a retrieval miss. This is a *chunking* finding (see `kbm/chunking.py`),
   surfaced only because the harness now reports soft matching.

**Inconclusive on this corpus:** the `top_k` pool-size win looked huge in v1 but does **not**
replicate cleanly here — `rerank_pool10` and `rerank_pool50` tie on R@5 (`0.91`), though pool50
still edges R@pool (`0.97 > 0.91`) and MRR (`0.74 > 0.70`). On a 66-chunk haystack this is within
noise. **Re-measure on a larger corpus before raising the `top_k` default** (deferred).

**Read the deltas, not the absolutes.** 66 chunks is a small, easy haystack, so recall is
flattering across the board; the *differences* between configs are the trustworthy signal.
Latency is omitted from this table on purpose — it was CPU-bound and the two pool configs ran
under background load, so their reranker times are not comparable. See §6 for GPU expectations.

### Experiment — embedding × chunking sweep (2026-07-21, Mac CPU, cleaned 19-Q gold set)

Question: we embed LaTeX equations and plain prose with the *same* generic English model
(`bge-small-en-v1.5`), whose tokenizer has never meaningfully seen LaTeX. Does a
LaTeX-tolerant embedder and/or equation-aware chunking retrieve math better? Run it yourself:

```bash
python evaluation/embed_chunk_sweep.py     # 9 indexes × {dense, hybrid+rerank}
```

3 chunkers × 3 embedders, the curated **19-question** calculus gold set (pruned from 34 to the
unambiguous questions), overlap matching (§4). `dense` isolates the embedding; `hybrid+rerank`
is what ships. `dense` ms = query-embed latency. (An earlier run on the noisier 34-question set
showed the same ordering with lower absolutes — cleaning the exam widened the bge-m3 gap.)

| chunker | embed | config | R@5 | R@5soft | R@pool | MRR | dense ms |
|---|---|---|---|---|---|---|---|
| baseline | bge-small | dense | 0.84 | 0.89 | 1.00 | 0.704 | 24 |
| baseline | **bge-m3** | dense | **0.95** | 0.95 | 1.00 | **0.787** | 84 |
| baseline | **bge-m3+norm** | dense | 0.95 | 0.95 | 1.00 | **0.876** | 57 |
| baseline | bge-small | hybrid+rerank | 1.00 | 1.00 | 1.00 | 0.866 | — |
| baseline | bge-m3 | hybrid+rerank | 1.00 | 1.00 | 1.00 | **0.886** | — |
| baseline | bge-m3+norm | hybrid+rerank | 1.00 | 1.00 | 1.00 | 0.882 | — |
| eqaware | bge-m3 | hybrid+rerank | 0.89 | 0.95 | 0.95 | 0.841 | — |
| eqaware_context | bge-m3 | hybrid+rerank | 0.89 | 0.95 | 0.95 | 0.783 | — |

**Findings (directional — n=19, one doc: trust deltas and MRR, not the binary near-ceilings):**

1. **bge-m3 clearly beats bge-small on *dense* retrieval** — the axis that isolates the embedder.
   R@5 0.84→0.95, MRR 0.704→0.787, R@pool 0.91→1.00. Cleaning the gold set *widened* the gap
   (it was 0.82→0.88 at n=34), so the LaTeX-tokenizer hypothesis holds more strongly.
   **Recommend switching the dense embedder to `bge-m3`.**
2. **The reranker hides the embedder's weakness — completely, here.** After `hybrid+rerank` all
   three embedders reach **R@5 1.00 / R@pool 1.00**; only MRR still separates them (m3 best at
   0.886). So the bge-m3 upgrade matters most *pre-rerank* — dense-only deployments and a
   perfect candidate pool — while the cross-encoder already recovers bge-small on the shipping path.
3. **pylatexenc normalization sharpens *ranking*, and more than before.** Same R@5 as raw bge-m3,
   but the best dense MRR on the board (0.787→**0.876**, a bigger jump than the 34-Q run's
   0.694→0.748). The gain still washes out after reranking and it adds CPU-bound query latency
   (high variance, up to ~350–600 ms on some rows — pylatexenc runs on CPU and won't benefit from
   the pod GPU). **Worth it specifically for dense-only ranking quality.**
4. **Latency is the tradeoff.** Query-embed: bge-small ~15–24 ms → bge-m3 ~84–310 ms →
   +norm ~57–600 ms (very noisy on CPU); index build 3 s → 10–26 s. Mac-CPU numbers; the embed
   gap shrinks on the target GPU, but pylatexenc's does not.
5. **This sweep still cannot fairly rank the chunkers.** The gold `chunk_text` *is* a baseline
   chunk, so baseline scores containment 1.0 for free while eqaware/eqaware_context reshape
   boundaries and lose *strict* overlap — baseline looks best on R@5 but the gap closes on
   **R@5soft** (eqaware ties it). eqaware provably keeps every equation intact (0 splits vs
   baseline's) and is at worst neutral. **To settle chunking, rebuild the gold set labeled by
   answer-span containment**, not whole-baseline-chunk overlap.

> **On n=19:** many cells are at or near 1.00, which is partly small-sample ceiling — each
> question is worth ~5.3 points, so R@5 1.00 vs 0.95 is a *single*-question difference. Read the
> continuous **MRR** column (which has headroom) over the binary recalls, and treat the absolute
> recalls as "no obvious failures on this small clean set," not "solved."

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
  redirects effort to the likelier win: **math-aware chunking** (`kbm/chunking.py`, `eqaware*`).
- **`recall@pool` is low across the board** → stop tuning retrieval. The ceiling is upstream:
  extraction (`extract.py`, Marker) and chunking (`kbm/chunking.py`).
- **`bm25` ≈ `hybrid`** → suspect the gold set (§3): the questions are probably parroting chunk
  vocabulary. Rewrite them before trusting anything else in the table.

The eval is also the training data for a future embedding fine-tune — a contrastive
fine-tune of `bge-small` needs exactly these `(question, correct chunk)` pairs.

---

## 10. Roadmap: making this harness able to *pick* components

Everything above measures whether a change helped. It cannot yet answer the question the
project actually needs answered: **which embedder, which reranker, which generator.**

This section is the design for that, written to be handed to someone (or something) that
has not been part of the conversation. It is a plan, not a record — nothing in §10 has
been run yet. Sections 1–9 are the measured present; §10 is the intended next state.

### 10.1 Why the current harness cannot rank components

Not because the metrics are wrong. Because **the corpus is too small.**

The gold set is 19 questions over `calculus_chainrule.mmd` — a ~10-page slice, **66
chunks**. On a haystack that small almost everything succeeds, and §7 records the damage
in its own words: the embedding sweep is flagged "directional," the `top_k` finding
"does not replicate cleanly … within noise," and the chunker comparison "still cannot
fairly rank the chunkers." Three separate questions went unanswered for the same reason.

At n=19 each question is worth ~5.3 points, so `R@5 1.00` vs `0.95` is one question.
Several cells sit at 1.00 with no headroom to move.

**Fix the corpus before touching anything else.** No harness change compensates for an
exam everyone passes.

### 10.2 Prerequisite: a real corpus

- Ingest **3–5 full textbooks** (OpenStax Calculus Vol 1–3, Algebra & Trig, Precalculus
  are free and openly licensed, which fits the project's constraint). Target a few
  thousand chunks, not 66.
- Extract with Marker **on the GPU pod**, not a laptop: ~0.3–1 s/page on GPU versus a
  measured ~11 min for 6 pages on Mac CPU. (Confirmed again during the API work: a
  10-page PDF took ~40 min on CPU.)
- **Freeze the `.mmd` files afterwards.** Chunk ids are `<source>::<n>`; re-extracting
  shifts boundaries and silently invalidates every gold label. `SETUP.md` warns about
  this for the parity run — it is now permanent.
- **This applies to the repo-relative `docs/extracted/` corpus built with the `extract.py`
  CLI, which is the only corpus the eval harness reads.** Documents uploaded through the
  web UI are *not* retained (`DEPLOYMENT.md §7`): their `.mmd` is deleted with the ingest
  job, so they cannot be re-chunked or used as gold-set sources without being uploaded
  again. Build the eval corpus with the CLI, deliberately, and keep it.

Expected effect: recall comes down off 1.00 and config differences exceed the noise
floor. **If recall is still ~1.00 after this, the corpus is still too easy and component
ranking is still unsafe.**

### 10.3 Gold set v3 — 50 questions, stratified

Extend `make_evalset.py`. **Keep all existing machinery**: the `is_answerable` chunk
gate, the distinctive-word `leak_score` with retry, the referential-phrase regex, and
`goldset_review.md` sorted worst-first. Those solved a real problem (§3) and none of it
is superseded. Add:

**Multi-document sampling** across the new corpus, so retrieval must discriminate between
similar sections in *different* books — a case one document structurally cannot produce.

**A `qtype` field, sampled to quota:**

| `qtype` | n | What it tests |
|---|---|---|
| `worked_example` | ~15 | the case the system exists for |
| `definition` | ~10 | conceptual prose retrieval |
| `notation` | ~8 | symbol/formula lookup — where the LaTeX-vs-prose embedder question bites |
| `multi_hop` | ~10 | **two gold chunks**; answerable only by combining them |
| `no_answer` | ~7 | **no gold chunk**; material genuinely absent from the corpus |

Two of these need schema changes, and both are load-bearing:

- **`multi_hop` requires `gold_chunk_id` → `gold_chunk_ids: [...]`**, scored both all-of
  and any-of. Every current label is a single chunk, which is *structurally incapable* of
  measuring whether the pipeline assembles evidence across chunks.
- **`no_answer` is scored inversely**: success is the system declining to answer from
  documents. This slice is also what calibrates `KBM_RELEVANCE_FLOOR` (see §10.7).

**The hand-cleaning pass in §3 is still mandatory.** Budget ~45 min for 50 questions.

### 10.4 The principle: isolate one component at a time

Each benchmark holds everything else fixed. All reuse `retrieval.retrieve_detailed()`
(which already returns the pre-rerank pool, the full ranked list, and per-stage timings)
and the `--match overlap` scorer.

**1. Embedder — `eval_embedders.py`** (generalize `embed_chunk_sweep.py`)

Isolating axis: **the `dense` config only.** §7 already found `hybrid+rerank` masks the
embedder completely — all three candidates reached `R@5 1.00`. Measuring an embedder on
the shipping path measures the reranker.

Candidates beyond bge-small / bge-m3 / bge-m3+pylatexenc: `Qwen3-Embedding-0.6B`,
`gte-modernbert-base`, `jina-embeddings-v3`, `nomic-embed-text-v2-moe`.

Report dense R@5, R@5soft, R@pool, **MRR and nDCG (the columns with headroom)**, query
latency, index build time, index size, VRAM — **broken out by `qtype`**. The `notation`
slice is where a LaTeX-tolerant tokenizer should show up; the aggregate hides it.

**2. Reranker — `eval_rerankers.py`** (new)

Isolating axis: **fix the candidate pool, vary only the rescorer.** Generate the pool
once per question, cache it, then score every reranker over the *identical* pool.
Re-running retrieval per reranker measures retrieval noise instead.

Candidates: `bge-reranker-v2-m3` (current), `bge-reranker-base`,
`jina-reranker-v2-base-multilingual`, `mxbai-rerank-base-v2`, `Qwen3-Reranker-0.6B`, and
**no reranker** as control.

Report ΔMRR / ΔnDCG@5 / ΔR@1 **at fixed R@pool**, plus latency and VRAM. Keep §6's
framing: not "is it fast enough" but "does it move ranking enough to justify 2.2GB."

**3. Generator — `eval_generators.py`** (new; the real gap)

Isolating axis: **fix the context, vary only the model.** Two conditions per question:

- `oracle` — feed the gold chunk(s). Pure reasoning quality, retrieval removed.
- `retrieved` — feed the real top-5. The shipping system.

**The `oracle − retrieved` gap attributes end-to-end quality loss to retrieval versus
generation** — nothing currently measures that.

Candidates: `deepseek-math-7b-rl:Q4` (current, set in `kbm/retrieval.py:OLLAMA_MODEL`), the
same at Q8, `Qwen2.5-Math-7B-Instruct`, `Qwen3-8B`. Q4 quantization costs accuracy on
exactly the multi-step arithmetic the model exists to do, so quantization is a candidate
axis in its own right, not a fixed background condition.

**A/B the prompt as a variable too.** `SYSTEM_PROMPTS` in `api/chat.py` has never been
evaluated — it was written once and kept. Prompt changes routinely buy more than a model
swap, and it is the cheapest axis here. When varying prompts,
preserve the static → history → context → question order (`LATENCY.md`); a prompt A/B
that reorders those blocks measures latency, not quality.

> **This benchmark's winner constrains deployment.** `DEPLOYMENT.md` sizes the pod for a
> 7B-class model on a 24GB card. If the winner needs more, the cost table must be redone.
> Treat that as an exit criterion of the benchmark, not a detail.

### 10.5 Standard reasoning benchmarks vs. this corpus — use both, for different things

Public math benchmarks (GSM8K, MATH, TheoremQA, OlympiadBench, MMLU-STEM, MathQA) are the
**right tool for the generator axis** and better than a bespoke set:

- Standardized and comparable to published numbers — you inherit others' baselines.
- **Verifiable answers** (numeric / exact-match), which removes the LLM judge from the
  correctness axis entirely and with it the calibration problem in §10.6.

They **cannot** measure the embedder or the reranker. They are closed-book: no corpus, no
chunks, so recall@k / MRR / nDCG / recall@pool are undefined. There is no substitute for
real documents on the retrieval axis.

Two traps:

- **GSM8K questions are self-contained by construction**, so retrieval contributes
  nothing to them. Evaluating the *whole system* on GSM8K would show retrieval adding ~0
  — an artifact of the benchmark, not a finding about the pipeline.
- **Contamination.** GSM8K/MATH test items are widely present in modern training data,
  and `deepseek-math-7b-rl` was RL-tuned on math and publishes numbers on them. Ranking
  candidate generators on a contaminated set can measure memorization and systematically
  favors whichever model saw the test data. Prefer newer or held-out sets (OlympiadBench,
  recent AIME, TheoremQA) when the goal is *ranking models against each other*.

**Division of labour: public benchmarks decide which generator; this corpus decides which
embedder and reranker. Neither substitutes for the other.**

### 10.6 Faithfulness — `eval_answers.py` (new)

Replaces the single 1–5 correctness score with a claim-level measure. Scored on
**`mode: "grounded"` answers only** (see §10.7).

Judge splits the answer into atomic claims, labels each against the retrieved context:

| Label | Meaning |
|---|---|
| `supported` | stated in the context |
| `derived` | not stated, but follows from it by arithmetic/algebra |
| `unsupported` | neither stated nor derivable — the hallucination case |
| `contradicted` | conflicts with the context — the worst case |

**`derived` is not optional.** A math tutor *must* go beyond its context — it performs
arithmetic the chunk does not contain. A binary supported/unsupported metric penalizes
exactly the behaviour the product exists for and would push the system toward useless
quotation. `faithfulness = (supported + derived) / total`, reporting `contradicted`
separately because one contradiction matters more than several unsupported asides.

Alongside: **answer relevance** (does it answer the question, independent of grounding)
and **abstention accuracy** on the `no_answer` slice.

### 10.7 Judge calibration — `judge_calibration.py` (new)

**The step that separates a benchmark from a vibe, and the one most likely to be skipped.**

§4 already warns a 7B judge is "a noisy instrument … a smoke alarm." That warning is
currently *unquantified* — nobody knows whether it is a smoke alarm or a random number
generator.

- Hand-label ~30 answers on the judge's own rubric.
- Measure agreement: **Cohen's κ** for categorical labels, **Spearman** for 1–5 scores.
- **Publish the agreement number next to every judged score in this file.** If κ is poor,
  either upgrade the judge or demote judged metrics to regression-detection only and let
  retrieval metrics carry the decisions.
- The judge must stay a different model from the generator under test (§4). This matters
  *more* now that generators are being compared.

Note §10.5 reduces the blast radius: with verifiable-answer benchmarks carrying
correctness, the judge is needed only for faithfulness and relevance.

### 10.8 Dependency on answer provenance (already shipped)

`api/` now labels every answer `grounded` or `general` (`api/chat.py:decide_mode`), and
**the server writes the provenance itself** rather than asking the model to — one
`Sources:` line appended to a `grounded` answer, naming the documents it retrieved.
Measured: instructed to state its provenance, `deepseek-math-7b-rl` ignored the
instruction and answered directly — it is a solver, not an instruction-follower, exactly
as §3 says.

A `general` answer carries no footer: the line names documents, and there are none to
name. That is a rendering choice only — the label itself is unchanged and still on the
`sources` and `done` SSE frames, which is where the eval reads it.

This is what makes faithfulness measurable at all. Without a trustworthy label, an answer
grounded in documents and one confabulated from parametric memory are indistinguishable,
and any faithfulness number is computed over a mixture of the two.

**`KBM_RELEVANCE_FLOOR` is calibrated — 0.15**, by `evaluation/calibrate_floor.py`.
It is on a **sigmoid (0–1) scale**, not raw logits (`sentence_transformers.CrossEncoder`
applies the model's activation and `bge-reranker-v2-m3` carries a Sigmoid).

The calibration did not wait for the `no_answer` slice, because an external benchmark
turned out to be the better instrument for this particular threshold and the slice still
does not exist. Two datasets, bounding the answer from opposite sides:

- **Our corpus, real pipeline** (19 on-topic gold-set questions, 18 written to be absent
  from the ingested chapter): off-topic max **0.0395**, on-topic min **0.5965**. An empty
  gap of more than an order of magnitude; 0.15 is its geometric midpoint. The old 0.01
  sat *below* the off-topic max and abstained on only 67% of them.
- **ARQMath-1** (`hcju/mseqa`, 34,813 human-graded pairs): pairwise AUC **0.60**, and the
  median human-judged-*irrelevant* pair scores **0.63**.

Those two look contradictory and are not. ARQMath's negatives are pooled hard negatives —
documents about the right topic that fail to answer the question — so it measures a
fine-grained judgement, and the AUC says bge-reranker-v2-m3 is close to useless at it.
Our floor only needs the coarse one (is this chunk about what was asked), where the same
model separates cleanly. **The floor protects against the wrong book, not the wrong
paragraph of the right book.** Do not quote an abstention number for the latter.

Still to do: re-calibrate from logged family questions (§10.9); 19 questions from one
chapter is a bracket, not a distribution. The number is also a property of the reranker's
output scale — **re-run the calibration if the reranker changes.**

> ⚠️ The same ARQMath run is evidence on a second question. MIRB (arXiv:2505.15585)
> reports that reranking with bge-reranker-v2-m3 *degrades* nDCG@10 across their math
> tasks, and an AUC of 0.60 on fine-grained relevance is consistent with that. Our
> pipeline puts that model last **and** uses its score as the abstention gate. §10.4's
> `eval_rerankers.py` is now the highest-value item in this document.

### 10.9 Usage telemetry (already shipped)

`kbm/telemetry.py` writes one JSONL record per query (hashed user, mode, retrieved chunk ids
and scores, per-stage timings) plus `feedback` records keyed by `event_id`. It exists now
because it **cannot be backfilled**. Two payoffs:

1. **Real questions replace synthetic ones.** §8 names "LLM-generated questions are not
   how your family talks" as a limitation this protocol cannot fix from the inside.
   Logged questions are the fix — **gold set v4 should be drawn from them.**
2. **Fine-tuning data.** Thumbs-up `(question, retrieved chunk)` pairs are exactly the
   contrastive pairs an embedding fine-tune needs. Fine-tuning `bge-small` (or whichever
   embedder §10.4 selects) on real `(question, correct chunk)` pairs is far cheaper than
   touching the 7B generator, and this log is where that training data comes from.

### 10.10 Order of work

1. **Corpus + pod.** Marker-extract the textbooks on GPU; freeze the `.mmd`. *Unblocks
   everything else.*
2. **Gold set v3.** Stratified, multi-hop and no-answer slices, hand-cleaned.
3. **Component benchmarks, in this order** — embedder → reranker (over a pool built with
   the winning embedder) → generator (retrieval fixed). Each one's output is the next
   one's input; running them in parallel compares configurations that will never ship
   together.
4. **Faithfulness + judge calibration.** Calibrate *before* citing any judged number.
5. **Re-run the §7 v2 baseline as a parity check** after harness changes. Those numbers
   should reproduce within noise; a shift means the harness changed, not the system.

Step 1 gates 2–4. Nothing in 2–4 is safe to interpret before step 1 lands.

---

## 11. Self-consistency: is majority voting worth k× the decode?

Everything above measures **retrieval**. This section measures the other half of a wrong
answer: the generator. `deepseek-math-7b-rl` is a solver, and when it is wrong it is often
wrong *unstably* — re-sample the same question at a non-zero temperature and the wrong
answers scatter while the right one repeats. **Self-consistency** (Wang et al., 2022)
exploits that: sample k chains of thought, take the majority final answer.

It is the cheapest accuracy fix available to this project — no new model, no fine-tune, no
labelled data — and also a **k× multiplier on the single most expensive stage in the
system**: generation is ~95% of query time (`LATENCY.md`). So the question is never "does
it help" but "does it help *enough here*". `evaluation/self_consistency.py` answers that.

```bash
# from knowledge-base-math/
python evaluation/self_consistency.py                    # 20 questions × (1 greedy + 10 sampled)
python evaluation/self_consistency.py --easy              # grade-school regression tier
python evaluation/self_consistency.py --limit 5          # smoke run
python evaluation/self_consistency.py --dry-run          # validate the set, generate nothing
python evaluation/verify_reasoning_set.py                # recompute every gold answer
bash evaluation/eval.sh --skip-sweep --self-consistency  # on the pod (baseline tier)
bash evaluation/eval.sh --skip-sweep --self-consistency --sc-easy
```

### 11.1 Two tiers, and why the distinction decides what a number means

| flag | file | n | role |
|---|---|---|---|
| `--baseline` (default) | `reasoning_set_college.jsonl` | 30 | **The set that decides things.** |
| `--easy` | `reasoning_set_easy.jsonl` | 20 | Regression tier. Not a quality score. |

**`--baseline` is college-level** — multivariable calculus, linear algebra, ODEs, analysis,
probability, abstract algebra — because that is the real target distribution. A benchmark
only discriminates near the incumbent's ~50% mark: DeepSeek scores **0.85 on the easy tier**,
which means three candidate models would all land in 0.85–0.95 and every difference would sit
inside the noise. A set the incumbent nearly aces cannot rank anything.

**`--easy` is kept, not deleted, as a regression tier.** A change that lifts hard questions
while breaking basic arithmetic is a bad change, and only the easy tier can see it. Read it
as a tripwire, never as evidence of quality — it is near ceiling by design.

Both tiers are **closed-book, deliberately.** The thing under test is the model's reasoning
stability. Route these through the RAG pipeline and a wrong answer could be the retriever's
fault, which is exactly the confound that makes a result unactionable. Retrieval quality is
§4's job; this is the generator in isolation.

They are also a **different kind of gold set from `goldset.jsonl`**: §3's caveats
(machine-generated questions, vocabulary leakage, hand-cleaning) do not apply, because these
were written by hand against known answers rather than generated *from* chunks.

**The answer key is derived, not trusted.** `evaluation/verify_reasoning_set.py` recomputes
every answer in both tiers with sympy, from an independent formulation that never reads the
JSONL, and exits non-zero on any disagreement or any question lacking a verification. This is
not ceremony: a wrong key marks a *right* model wrong, sends you hunting a model bug that
does not exist, and silently corrupts accuracy, the self-consistency delta, and the model
bake-off alike. It has already earned its place — it caught a bad determinant in the first
draft of the college set. **Edit a question, edit the verification, and keep them agreeing.**

What still applies is size: 30 questions is a small exam, and a ±0.05 difference is inside
the noise. Treat the output as a go/no-go signal, not an effect size.

### 11.1b Scope: closed-ended only

College math is substantially proof-based, and "prove that the sequence converges" has no
boxed answer. Both tiers are therefore **closed-ended by construction**, which bounds what
this section can speak to:

- Closed-ended numeric/symbolic questions → this harness works, and self-consistency applies.
- Proof and derivation questions → need LLM-judged grading (§10.6), and **majority voting is
  simply inapplicable** — you cannot vote on a proof.

Before treating a self-consistency result as a decision about the product, check what
fraction of real queries is which. `kbm/telemetry.py` is already logging real questions; that is
the only thing that can answer it (§8, §10.9).

### 11.2 How k=1 / 5 / 10 are compared fairly

The naive protocol generates 1, then 5, then 10 answers per question — 16 generations, and a
k=1 number estimated from a single sample, which on 20 questions is almost pure variance.

Instead the script draws **N samples once** per question and estimates majority-vote accuracy
at each k by **resampling k of those N without replacement** (400 draws per question × k).
Same generations, far tighter estimates, and every k is scored against the *identical* pool
of model outputs — so a gap between k=1 and k=10 is the voting, not sampling luck.

A **greedy (temperature 0) run is scored separately**, because that is what ships today. The
sampled k=1 row is the control for the voting mechanism itself (same temperature, no vote);
greedy is the baseline the change would have to beat in production. Reporting only sampled
k=1 would flatter self-consistency, since raising the temperature costs accuracy before
voting wins it back.

Voting details that decide whether the measurement is honest at all:

- **Answers are normalized to numbers before they vote.** If `5/16` and `0.3125` count as
  different votes the majority splits and self-consistency measures the parser, not the
  model. `\boxed{}` first (deepseek-math emits it natively), then a `Final answer:` line,
  then the last number in the text; LaTeX fractions and `$…$` are stripped, values compared
  numerically with tolerance.
- **Unparseable samples do not vote.** A model that never stated an answer has not cast one.
  The unparseable rate is reported next to accuracy — if it is high, fix the prompt before
  reading anything else.
- **Ties break toward the first-seen answer**, which is what a real streaming implementation
  would do.

### 11.3 Reading the output

The table gives accuracy for greedy and for each k, with the generation count and estimated
serial seconds per query. The part that decides the answer is the **ERROR STRUCTURE** block:

- **fixed by voting** (greedy wrong → vote right) — the win, question by question.
- **broken by voting** (greedy right → vote wrong) — the cost nobody budgets for. Sampling
  can lose a question greedy got right.
- **confidently wrong** (≥80% of samples agree on a wrong answer) — stable errors: the model
  is not guessing, it is reliably mistaken.
- **never right** (≤10% of samples correct) — the samples scatter across many *wrong*
  answers and the right one is essentially absent. Voting reorders a pool; it cannot add to
  it. These look nothing like the confident case (low modal share, high diversity) but are
  just as unreachable — counting only unanimity understates the ceiling, which is why both
  are reported and summed into **BEYOND VOTING'S REACH**. That count, not the accuracy
  delta, is the hard limit on what self-consistency can ever deliver on this set.
- **mean distinct answers per question** — the diversity the method depends on. Near 1.0
  means the samples are effectively deterministic and voting is a no-op; raise the
  temperature or stop.

### 11.4 How to act on the result

- **Gain < 5 points** → **don't implement.** A multi-x bill on the stage that already owns
  95% of latency, for a difference inside the noise of a 20-question set.
- **5–10 points** → **make it opt-in, not the default.** A "check my work" button that
  spends 5× decode on demand, rather than paying it on every family question.
- **> 10 points** → **implement**, and check whether a smaller k captures most of it — the
  accuracy/k curve is usually steeply diminishing (k=5 typically gets most of k=10's win at
  half the cost).
- **Most errors are BEYOND VOTING'S REACH** → self-consistency is the wrong lever entirely.
  Look at the prompt, at a larger/better generator, or at whether retrieval should have been
  supplying the fact in the first place.

### 11.5 EASY-TIER run — 2026-08-19, Mac (Metal), `t1c/deepseek-math-7b-rl:Q4`

⚠️ **This is the `--easy` tier, not `--baseline`.** It predates the college set and its
verdict does **not** carry over — see the scoping note at the end of this section. Kept
because it is a real measurement and the easy tier's regression role needs a reference point.

20 questions × (1 greedy + 10 sampled at T=0.8/top_p=0.95), 220 generations, 31.6 min wall
clock, ~8.2 s/generation. `--report-only
evaluation/results/self_consistency_easy_deepseek-math-7b-rl_Q4.json` re-prints this without
regenerating anything. (Result files gained the model name in §12; a run from before that
change is at the old `self_consistency_easy.json`.)

| setting | accuracy | vs greedy | gens/q | est. serial s/query |
|---|---|---|---|---|
| greedy (ships today) | 0.85 | — | 1 | 8.2 |
| majority vote k=1 | 0.85 | −0.00 | 1 | 8.7 |
| majority vote k=5 | 0.88 | +0.03 | 5 | 43.3 |
| majority vote k=10 | 0.90 | +0.05 | 10 | 86.5 |

Error structure: 1/20 fixed by voting, **0/20 broken** by voting, 2/20 beyond reach
(r03 inclusion–exclusion, r07 modular exponentiation — both scattered, 0% and 10% of samples
correct). Mean 2.0 distinct answers per question, 0% unparseable.

**Verdict: do not implement as the default path.** Three greedy errors; voting fixes exactly
one (r04, a decoding slip the samples unanimously got right), and the other two are number
theory the model simply cannot do — it never produces the right answer at any temperature, so
k could be 100 and they would still be wrong. +5 points is inside the noise of a 20-question
set, and it costs 10× the decode on the stage that already owns ~95% of query time: ~86 s per
answer serially, versus 8. Even k=5's +3 points costs 43 s.

### 11.6 What the failures actually are — execution, not knowledge

Worth reading the wrong answers, not just counting them. The answer distributions say the
model is not ignorant of the methods; it cannot execute arithmetic reliably.

**r07 — remainder of 7^100 mod 13** (gold 9). Ten samples returned 1 ×3, 3 ×3, 7 ×3, 9 ×1 —
every one of those is a member of the cycle of 7 mod 13, i.e. the model found the right
*structure* every time and picked the wrong position in it. The greedy trace is unambiguous:

> find the smallest k with 7^k ≡ 1 (mod 13) … k = 12 … divide 100 by 12, quotient 8
> remainder 4 … so 7^100 ≡ 7^4 … 7^4 = 2401 ≡ 3 (mod 13)

Order of the group: right. Exponent reduction: right. 2401 = 13·184 + **9**, not 3. The
entire method is correct and one long division at the end is wrong.

**r03 — sum of 1..100 divisible by 3 or 5** (gold 2418). Samples included 2385 and 2413 —
inclusion–exclusion applied, arithmetic slipped. 285 is the sum of the multiples of 5 alone,
i.e. a run that stopped before the union.

The implication for this project: **sampling is the wrong lever for an arithmetic-execution
error.** Voting helps when errors are random *and* the right answer is in the pool; here the
model reaches a different wrong number each time, so the pool never contains the answer to
vote for. The lever that matches this failure is **tool-integrated reasoning** — let the
model emit Python for the arithmetic step and execute it. DeepSeekMath's own paper reports a
large program-aided gain over chain-of-thought on MATH for exactly this reason. Second lever:
this is a **Q4 quantization**, and 4-bit degrades multi-digit arithmetic more than prose —
re-running §11.5 at Q8 or fp16 would separate "the model can't" from "the quantization can't"
and is a cheap experiment.

Two findings that generalize beyond the go/no-go:

1. **Sampling at T=0.8 is free here** (k=1 sampled ≈ greedy, and 0/20 broken by voting). The
   usual objection — "raising the temperature loses questions greedy got right" — did not
   materialize, so an opt-in path would not be trading accuracy away.
2. **The model's failures are knowledge-shaped, not decoding-shaped.** 2 of 3 errors are
   "doesn't know the method", which sampling cannot touch. That is an argument for a better
   generator or for making sure retrieval supplies the method — not for spending 10× decode.

The actionable form, if it is wanted later, is an opt-in **"check my work"** button at k=5:
paid per request rather than per query, on the questions a user already doubts.

**Scope warning — this verdict is measured on the wrong population.** Self-consistency pays
off in the *middle* of the accuracy range: it needs the right answer to be present in the
sample pool but not to be the single-sample favourite. At 0.85 there is almost no headroom,
and the two failures here were "never right" — the answer absent from the pool entirely.
Both are regimes where voting is structurally useless, and this tier contained only those
two regimes. On `--baseline`, where the model is expected nearer 0.5, the same measurement
could land very differently. **Re-run on the college tier before treating "don't implement"
as settled.** The harness is unchanged; it is one command.

Two things this section does *not* measure, and both matter before shipping:

1. **Interaction with retrieval.** These questions are closed-book; a grounded question's
   errors may be differently distributed (the context may already stabilize the model,
   shrinking the win). Re-check on grounded questions before making it the default path.
2. **Streaming.** `/chat` streams SSE token-by-token (`api/routes.py`). Majority voting
   cannot stream — the answer does not exist until every sample is complete — so adopting it
   changes the UX from "tokens appear immediately" to "nothing for k× the latency, then an
   answer." That is a product decision, not just a cost one.


---

## 12. Model bake-off: is another 7B generator better than deepseek-math?

§11.6 ended with a diagnosis rather than a fix. deepseek-math-7b-rl:Q4's errors on the easy
tier were *execution* errors — right method, wrong long division — and *knowledge* errors,
and it named two levers that could move them: tool-integrated reasoning, and a better
generator. `evaluation/model_bakeoff.py` measures the second.

```bash
# from knowledge-base-math/
python evaluation/model_bakeoff.py --check        # published numbers + which arms are pulled
python evaluation/model_bakeoff.py --dry-run      # plan and cost, generates nothing
python evaluation/model_bakeoff.py                # the run
python evaluation/model_bakeoff.py --report-only  # re-compare saved runs, generates nothing
python evaluation/model_bakeoff.py --easy         # regression tripwire, AFTER a baseline win
```

### 12.1 The arms, and why each one

All Q4_K_M, because the incumbent is a Q4 GGUF and comparing it against an fp16 challenger
measures the quantization and calls it the model.

| arm | role | the question it answers |
|---|---|---|
| `t1c/deepseek-math-7b-rl:Q4` | incumbent | the thing to beat |
| `hf.co/bartowski/Qwen2.5-Math-7B-Instruct-GGUF:Q4_K_M` | math specialist, newer | does a year-newer specialist close §11.6's arithmetic gap? |
| `qwen2.5:7b-instruct-q4_K_M` | strong generalist | does a generalist beat a math specialist *at math*? |

`qwen2-math:7b-instruct-q4_K_M` is a fallback for the second row: Qwen2.5-Math has no entry
in the official Ollama library (Ollama pulls its GGUF from HuggingFace directly), and if that
route is unavailable on the pod, Qwen2-Math is one library `pull` away at a small cost in
paper score.

The generalist arm is not a throwaway. **Qwen2.5-7B-Instruct has a 128K context window;
deepseek-math and Qwen2.5-Math both have 4,096.** For a closed-book reasoning tier that is
irrelevant, but the serving path sends retrieved chunks *plus* conversation history *plus*
the prompt, and 4K is the budget `api/chat.py` is already working inside. A challenger that
ties on math and gives 32× the context is not a tie.

### 12.2 Published benchmarks — reference, not evidence

From the **Qwen2.5-Math Technical Report ([arXiv:2409.12122](https://arxiv.org/abs/2409.12122),
Table 3)**, few-shot chain-of-thought, English, unquantized. The value of this particular
table is that *one harness scored every row*, so the rows are comparable to each other —
which is exactly what a table assembled from each model's own README would not be.

| model | GSM8K | MATH | context |
|---|---|---|---|
| **DeepSeekMath-7B-RL** (incumbent) | 88.2 | 52.4 | 4,096 |
| **Qwen2.5-Math-7B-Instruct** | **95.2** | **83.6** | 4,096 |
| Qwen2.5-7B-Instruct | 91.6 | 75.5 | 131,072 |
| Qwen2-Math-7B-Instruct | 89.9 | 75.1 | 4,096 |
| Mathstral-7B-v0.1 | 84.9 | 56.6 | 32,768 |
| Llama-3.1-8B-Instruct | 76.6 | 47.2 | 131,072 |

Two footnotes on the incumbent row: DeepSeekMath's own paper
([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)) reports **88.2 / 51.7**, and Qwen's
re-scoring gives 52.4 — the sub-point difference between a model's self-report and an
independent harness is itself a useful calibration. Qwen2.5-Math-7B-Instruct also reports
**94.6 / 85.2 under TIR** (tool-integrated reasoning, i.e. the model writes and executes
Python), which is the *other* lever §11.6 named, and TIR is worth +1.6 MATH on top of an
already-strong CoT score.

**Why these numbers do not decide anything here:**

1. **Wrong distribution.** GSM8K is grade-school word problems; MATH is competition problems.
   The college tier samples multivariable calculus, linear algebra, ODEs, analysis and
   probability. A model tuned for competition tricks is not obviously the model for a
   family's textbook questions.
2. **Contamination.** Both are public test sets with known leakage into training corpora.
   The 30 college questions are hand-written and not on the internet, which is the entire
   reason they exist (§11.1).
3. **Quantization.** Every published row is unquantized; every arm here is Q4. §11.6 already
   flags 4-bit as a suspect in the multi-digit-arithmetic failures this bake-off is trying to
   fix, so the Q4 gap could differ from the fp16 gap in either direction.
4. **Prompting.** Table 3 is **few-shot** CoT; `SC_PROMPT` is **zero-shot** — one instruction
   and the problem, no worked exemplars. Few-shot exemplars mostly buy *format compliance*,
   which is the exact axis §12.4's parse-rate guard watches, so a model that looks strong at
   5-shot can lose points here for reasons that have nothing to do with mathematics. Zero-shot
   is the right choice anyway, because it is what `api/chat.py` actually sends — but it means
   our absolute numbers should be expected to sit *below* every published row, for all arms.

And the one confusion this table invites, stated explicitly: **every number in it is CoT, not
TIR.** Qwen2.5-Math's TIR scores (94.6 / 85.2) are quoted above and were deliberately *not*
used to choose arms, because nothing in this system executes tool calls — picking a model on
its TIR column would be choosing on a capability we have not built. If TIR is ever adopted
(`MAIN_LLM_ANALYSIS.md §4.3`), the arms must be re-chosen against the TIR column, and this
bake-off's result does not carry over.

Read as: a +31 MATH gap on paper is a strong reason to spend two hours of GPU time. It is not
a result.

### 12.3 Why a separate script, and the statistics that force it

`self_consistency.py --model X` already runs any generator, and the `--model` flag was added
for exactly that. What it cannot do is *rank* two of them, and reading two of its reports
side by side is the wrong test.

**On 30 questions, an unpaired comparison needs roughly an 18-point gap to clear 95%
confidence.** A real 10-point improvement would be dismissed as noise, and a fluke would look
like a win. But both models answer the *same* 30 questions, which makes this a paired design
— and the paired test is both the honest one and the far more sensitive one.

`model_bakeoff.py` uses **McNemar's exact test**. It throws away every question both models
got right and every question both got wrong (those carry no information about which model is
better — and they are what makes the unpaired test so blunt) and asks only about the
disagreements: if the models were equally good, each disagreement is a coin flip. Six wins
and zero losses is p = 0.031 and decides the question; six wins and four losses is p = 0.75
and decides nothing, no matter how good the accuracy column looks.

The output reports `+wins / −losses / =ties` **with the question ids**, because *which*
questions a challenger fixes and breaks is more informative than the count. Six wins spread
across topics is a better model; six wins all in linear algebra is a model to route to.

### 12.4 The confound that ruins naive bake-offs: parse rate

`SC_PROMPT` asks for the answer in `\boxed{}`. deepseek-math emits that natively (it is
RL-trained on the format) and both Qwen math models are trained on it — but a **general**
instruct model may reason correctly and answer in prose, extracting to `None` and scoring
zero. That is a prompting failure reported as a reasoning failure, and it would show up as a
clean, believable, entirely wrong accuracy table.

So the report prints per-model unparseable rates **above** the accuracy table, and refuses to
declare a winner when the spread exceeds 5 points. If the generalist arm trips this, the fix
is a prompt change for that arm and a re-run — not a footnote.

The same principle in the other direction: the prompt is held *identical* across arms, along
with the questions, extraction, scoring, sample count, temperature and top_p. Only the model
tag varies. Per-model prompt tuning would be a fairer comparison of best-case behaviour and a
worse comparison of *this system* — but if an arm can only be scored fairly with its own
prompt, that is a finding to record, not to hide.

### 12.5 What "better" has to mean before anything ships

A win on the college tier is necessary, not sufficient. Before swapping `OLLAMA_MODEL`:

1. **`beyond` must shrink, not just accuracy rise.** The report's `beyond` column is
   §11.3's BEYOND VOTING'S REACH count — questions the model is stably wrong about or never
   gets right. A challenger that wins on accuracy while `beyond` holds steady won the
   decoding lottery; one that shrinks `beyond` actually raised the ceiling.
2. **The `--easy` regression tier must not fall.** A generator that wins on college
   questions and breaks grade-school arithmetic is a loss for a family QA system. This is
   the tier's only job (§11.1) — run it second, never first.
3. **Context length against the real serving prompt.** 4K is the budget for retrieved chunks
   + history + system prompt in `api/chat.py`. Verify the winner inside that budget, not
   just closed-book.
4. **Output style against `api/chat.py`.** deepseek-math's `\boxed{}` habit and its
   documented refusal to follow the provenance instruction (CLAUDE.md, `chat.decide_mode`)
   are things the serving layer is *built around*. A model with different habits may make
   `chat.PrefillEcho`/`QuestionEcho` and the truncation-continuation logic behave differently.
5. **Re-run the retrieval eval.** `eval.py --answers` judges end-to-end answers; the generator
   is half of that number.

### 12.6 Cost

330 generations per arm (30 questions × (1 greedy + 10 sampled)), 990 for the three-arm
default. At the ~8 s/generation measured on the Mac in §11.5 that is ~2.2 hours; a 4090-class
pod is roughly 10× faster, so ~15 minutes plus model pulls (~4.5GB each). Arms are unloaded
between runs (`keep_alive=0`) so three 7B models do not sit in VRAM at once.

`--skip-existing` reuses an arm's saved results JSON, which is how a fourth model gets added
later without paying for the first three again. Results are per (tier, model) —
`self_consistency_<tier>_<model>.json` — so no arm can overwrite the arm it is measured
against.

---

## 13 vs 12 — two different questions

§12 asks **which model**, and its instrument is a paired test over the same 30 questions.
§13 asks **whether a tool helps, and through which protocol**, holding the model fixed.
They share `evaluation/self_consistency.py` for generation and scoring; §12 adds
`model_bakeoff.py` for the ranking statistics, and §13 adds the `--tir` / `--tools` arms.

Read §12's **§12.4 (parse rate)** before reading any accuracy number in §13 too — the
confound is the same, and a tool arm is if anything more exposed to it, because a model
mid-tool-call has more ways to end a turn without a `\boxed{}`.


---

## 13. Tool protocols: is the Python sandbox worth it, and via which protocol?

§11.6 diagnosed deepseek's failures as **execution, not knowledge** — r07 reduced
`7^100 mod 13` correctly and then computed `2401 mod 13` as 3 instead of 9 — and named
the matching lever: *"let the model emit Python for the arithmetic step and execute it."*
§10.4 already listed the candidate generators. This section is that experiment.

The independent reason to run it is instruction-following, and it is not a preference.
Three separate places in this repo record the same finding — `ERRORS.md` (the model
ignores "say this isn't from your documents", so the server writes the `Sources:` line
itself), §5 (it writes poor gold-set questions, so `qwen2:7b` writes them), §10.8 (same
again for provenance tags). `api/chat.py:SYSTEM_PROMPTS` is therefore mostly decorative
today. An instruct model would let the prompt do its job.

### 13.1 What was built

- **`kbm/tools/sandbox.py`** — a subprocess with an AST allow-list, rlimits, a scrubbed environment
  and a 400-character output cap. A gate, not a jail; the threat it is sized for is a 7B
  model emitting a runaway loop, not an adversary. Read its docstring before widening
  `ALLOWED_IMPORTS`.
- **`kbm/tools/tir.py`** — the protocol, and only the protocol: stop word `` ```output ``, the
  ` ```python ` → ` ```output ` splice, `MAX_TOOL_ROUNDS = 3`. Every constant is Qwen's
  own (`QwenLM/Qwen2.5-Math`, `evaluation/math_eval.py`), because the model was fine-tuned
  against that exact shape.
- **`kbm/llm_profiles.py`** — window size, decode budget and TIR capability per model, so
  naming a generator configures it. `KBM_LLM_MODEL` selects; env vars override.
- The tool loop in `api/routes.py` is the **continuation loop with a second arm**, not a
  parallel machine, and `evaluation/self_consistency.py --tir` drives the same `kbm/tools/tir.py`
  primitives. An eval that reimplemented the protocol would drift from what ships — the
  argument `kbm/retrieval.py` exists for.

### 13.2 The arms

| arm | `GEN_MODEL` | flag | what it isolates |
|---|---|---|---|
| A | `t1c/deepseek-math-7b-rl:Q4` | — | the incumbent, unchanged |
| B | `hf.co/bartowski/Qwen2.5-Math-7B-Instruct-GGUF:Q4_K_M` | — | model swap alone |
| C | same as B | `--tir` | the sandbox's contribution (B vs C) |
| D | `qwen3:8b` | `--tir` | whether a *math-only* model is the right call at all |
| E | `qwen3:8b` | `--tools` | **the protocol, isolated.** Same model, same sandbox, same budget as D — the only difference is how the model asks for it (`kbm/tools/agent.py`'s JSON tool calls vs `kbm/tools/tir.py`'s text blocks) |

Quantization is matched to Q4 across A–C on purpose: Q8 would confound the model swap
with a quantization change, and §11.6 already flagged Q4 as a suspect in its own right.

```bash
GEN_MODEL=t1c/deepseek-math-7b-rl:Q4 bash evaluation/eval.sh --skip-sweep --self-consistency
GEN_MODEL=hf.co/bartowski/Qwen2.5-Math-7B-Instruct-GGUF:Q4_K_M \
  bash evaluation/eval.sh --skip-sweep --self-consistency
GEN_MODEL=hf.co/bartowski/Qwen2.5-Math-7B-Instruct-GGUF:Q4_K_M \
  bash evaluation/eval.sh --skip-sweep --self-consistency --tir
GEN_MODEL=qwen3:8b bash evaluation/eval.sh --skip-sweep --self-consistency --tir
python evaluation/self_consistency.py --model qwen3:8b --tools   # arm E
```

**Arm E is why `tir` and `tools` are independent fields in `kbm/llm_profiles.py`.** qwen3 has
both capabilities; the server picks one (`config.TIR_ENABLED = False if TOOLS_ENABLED …`),
and `KBM_TOOLS=0` is what keeps D runnable at all. D vs E is the only place the question
"is the text protocol or the native one better *for the same model*?" can be answered, and
it is worth answering before agent mode is made anyone's default.

⚠️ **Arm E binds `run_python` only, never `search_documents`** — unlike the server, which
binds all three. That is deliberate and is stated in `Generator`'s docstring: this benchmark
is closed-book by construction (§11.1b), and routing a question through retrieval would make
a wrong answer un-attributable — the model's error and the retriever's become one number.
Retrieval-path behaviour is §13.3's job, not this one's.

Each writes `results/self_consistency_baseline_<model>[_tir].json`, so the four coexist
and `--report-only` re-scores any of them for free.

**Run the college tier.** §11.1 is explicit about why, and §11.5 ends by saying the
easy-tier verdict is unsettled until it is done: deepseek scores 0.85 on `--easy`, where
four candidate models would land inside the noise of each other. The easy tier is the
regression tripwire — run it too, but never rank on it.

### 13.3 What the closed-book run cannot tell you

Self-consistency is **closed-book by construction** (§11), so it measures the generator
and says nothing about the thing that motivated half of this work. Instruction-following
in the RAG path needs its own run, per arm:

```bash
KBM_LLM_MODEL=<arm> python evaluation/eval.py --user <u> --all --answers --judge-model <non-qwen>
```

⚠️ The default judge is `qwen2:7b` (`eval.py:JUDGE_MODEL`). Judging three Qwen arms with a
Qwen judge is a same-family self-preference risk — not self-grading, but not clean either.
Use `--judge-model` with a non-Qwen instruct model for the bake-off, or report both and say
which is which.

And three behaviours that no aggregate score will surface, each a probe worth running by
hand and recording here:

0. **Does it search when it should, and stop when it should not?** Agent mode only, and it
   has no closed-book proxy. Three questions, three expected decisions: something the corpus
   demonstrably covers *and* the upfront retrieval already found (expect **no** search — the
   context is there); something in the corpus the upfront query misses because the student's
   wording differs from the book's (expect a search, with a **rewritten** query); pure
   arithmetic (expect `run_python`, or a correct answer without it). `done.searches`,
   `done.late_sources` and telemetry's `search_queries` make all three readable. The
   measured failure mode is over-eager calling — asked "why is the derivative of a constant
   zero?", `qwen2:7b` reached for `run_python` and wrote code that printed nothing.
1. **Does it write its own source list?** deepseek does not, which is why
   `chat.sources_footer` exists. A model that complies with the "do not cite" instruction
   is a win — but a model that cites *anyway* now produces **two** footers, one of them
   invented. Ask a grounded question and count the `Sources:` lines.
2. **Does it say when the context does not cover the question?** Ask something the corpus
   demonstrably lacks, in `grounded` mode. Silence here is the failure `RELEVANCE_FLOOR`
   only partly catches.
3. **Does it stop?** `LATENCY.md` measures deepseek filling any budget it is given (534
   tokens for a one-line conceptual question). An instruct model that stops when it is
   done is worth real latency, and shows up as `truncated=false` at a *lower*
   `KBM_NUM_PREDICT`.

### 13.4 What would make this a swap

Read in this order, and stop at the first one that fails:

1. **Tool use actually happened.** The `TOOL USE` block in the report. `greedy answers
   that ran a program: 0/30` means the arm measured plain CoT whatever the accuracy column
   says — the report says so itself rather than letting the number be read.
2. **College-tier accuracy**, A vs C. §11.4's thresholds apply: under 5 points is noise on
   30 questions.
3. **`BEYOND VOTING'S REACH` fell.** This is the diagnostic §11.6 cared about. The tool is
   supposed to convert *execution* failures into correct answers; if the count of
   stably-wrong questions does not move, the sandbox is not touching the failure mode it
   was built for, and the accuracy delta came from somewhere else.
4. **The easy tier did not regress.** Its whole job.
5. **The `--answers` run and the three probes above.** A generator that wins the closed-book
   set and stops grounding its answers is not an improvement to this product.

### 13.5 Cost and constraints known before the run

- **Qwen2.5-Math is a 4096-token model** (`max_position_embeddings`, config.json). Same as
  deepseek, but a TIR trace re-sends the whole transcript every round, and Ollama answers
  an overflow by shifting the window from the **left** — dropping the system prompt first,
  silently. `MAX_TOOL_ROUNDS=3` and `sandbox.MAX_OUTPUT_CHARS=400` are that budget, and
  `config.NUM_CTX` is what makes it explicit. Arm D exists partly because Qwen3 is not
  capped this way, which matters most in `grounded` mode where retrieved context is
  competing with the trace.
- **Latency.** `LATENCY.md`: generation is ~95% of query time. TIR multiplies generation
  passes by up to 4 and adds prefill for a growing transcript. A TIR answer will be several
  times slower than a CoT one — `DoneEvent.timings.tool_ms` separates sandbox time from
  decode time so the two are not confused. Measure before flipping the default, and
  consider gating TIR on question shape rather than every turn.
- **Contamination.** §10.5's warning stands and applies to Qwen too: these models publish
  MATH/GSM8K numbers. The college tier is hand-written and sympy-verified
  (`verify_reasoning_set.py`), which is exactly why it is the tier that decides.

### 13.6 Results

Not yet run. Record each arm's table, its `TOOL USE` block, and its
`BEYOND VOTING'S REACH` count here, then state the decision and the date — the same shape
as §11.5. **The default in `config.OLLAMA_MODEL` stays deepseek-math until this section
has numbers in it.**

⚠️ **Read this before running arm D or E.** The harness did not pass `reasoning=` to
`ChatOllama` until 2026-08-20, so qwen3 ran with **thinking mode on** — its default. The
`<think>` block never appears in `.text` (langchain-ollama routes it to
`additional_kwargs["reasoning_content"]`), and it consumes the entire `num_predict` budget,
so the answer is never reached and every sample scores *unparseable*. On a 3-question smoke
that was the difference between accuracy **0.50 with 50% unparseable** and **1.00 with 0%**,
and between 7 minutes and 9 seconds for one question. Any arm-D or arm-E number produced by
an older checkout is measuring thinking-mode overflow, not the model. Fixed in
`Generator.__init__` (`reasoning=profile_for(model).think`); see `ERRORS.md` 2026-08-20.

**Harness status.** Arm E (`--tools`) is implemented and verified end to end on a 3-question
smoke of the college tier: `greedy answers that ran a program: 3/3`, 3/3 correct, 0 sandbox
failures. That is a wiring check and **not a result** — three questions is not an exam, and
the arm is only worth reading against arm C and arm D on the full 30.
||||||| 1783c86
