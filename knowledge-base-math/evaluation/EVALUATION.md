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
finding (ROADMAP §3), not a retrieval one, and no reranker will fix it.

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
   400-char/80-overlap boundary, not a retrieval miss. This is a *chunking* finding (ROADMAP §3),
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
  redirects effort to the likelier win: **math-aware chunking** (ROADMAP §3).
- **`recall@pool` is low across the board** → stop tuning retrieval. The ceiling is upstream:
  extraction (ROADMAP §2) and chunking (§3).
- **`bm25` ≈ `hybrid`** → suspect the gold set (§3): the questions are probably parroting chunk
  vocabulary. Rewrite them before trusting anything else in the table.

The eval is also the training data for the embedding fine-tune (ROADMAP §6) — a contrastive
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

Candidates (ROADMAP §7): `deepseek-math-7b-rl:Q4` (current), the same at Q8,
`Qwen2.5-Math-7B-Instruct`, `Qwen3-8B`. A/B **the prompt** as a variable too — ROADMAP §7
notes it is unexamined and usually buys more than a model swap. When varying prompts,
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

`api/` now labels every answer `grounded` or `general` (`api/chat.py:decide_mode`), and in
`general` mode **the server prepends the marker itself** rather than asking the model to.
Measured: instructed to emit that line, `deepseek-math-7b-rl` ignored it and answered
directly — it is a solver, not an instruction-follower, exactly as §3 says.

This is what makes faithfulness measurable at all. Without a trustworthy label, an answer
grounded in documents and one confabulated from parametric memory are indistinguishable,
and any faithfulness number is computed over a mixture of the two.

**`KBM_RELEVANCE_FLOOR` is uncalibrated.** It is on a **sigmoid (0–1) scale**, not raw
logits — `sentence_transformers.CrossEncoder` applies the model's activation and
`bge-reranker-v2-m3` carries a Sigmoid. Observed: unrelated text ~1e-5–1.5e-3, weakly
on-topic ~0.19, correct chunk ~0.94. Default is 0.01, a guess from a handful of pairs.
**Sweep it against the `no_answer` slice and set it from data** before quoting any
abstention number.

### 10.9 Usage telemetry (already shipped)

`telemetry.py` writes one JSONL record per query (hashed user, mode, retrieved chunk ids
and scores, per-stage timings) plus `feedback` records keyed by `event_id`. It exists now
because it **cannot be backfilled**. Two payoffs:

1. **Real questions replace synthetic ones.** §8 names "LLM-generated questions are not
   how your family talks" as a limitation this protocol cannot fix from the inside.
   Logged questions are the fix — **gold set v4 should be drawn from them.**
2. **Fine-tuning data.** Thumbs-up `(question, retrieved chunk)` pairs are exactly the
   contrastive pairs the embedding fine-tune (ROADMAP §6) needs.

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
