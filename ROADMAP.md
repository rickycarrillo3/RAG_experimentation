# Roadmap: improvements after cross-encoder reranking

Where to take `knowledge-base-math` once the reranking stage from `plan.md` lands
(hybrid BM25 + dense → RRF top-20 → `bge-reranker-v2-m3` → top-5 → LLM).

Ordered roughly by expected quality-per-effort for math QA. Everything here stays within
the free / open-source constraint.

## 1. Build an evaluation set (do this first)

Today every change — reranking included — is judged by eyeballing whether the top-5
"looks different." That doesn't scale and can't tell you when a change makes things worse.

- 30–50 question/answer pairs drawn from the actual textbooks, each tagged with the chunk
  (or chunk id) that *should* be retrieved.
- An `eval.py` reporting **recall@5** and **MRR** for a given retrieval config.

A couple of hours of work, and it turns every item below from a guess into a measurement —
including the honest possibility that reranking didn't help and that `bge-reranker-v2-m3`
isn't worth its 2.2GB.

The eval set also doubles as training data for the embedding fine-tune in §6.

## 2. Fix extraction quality

No retrieval stage can recover a formula that was mangled into ASCII soup on the way in —
extraction bounds everything downstream.

- Finish the **Nougat → Marker** swap already in progress on the
  `worktree-marker-extraction` branch.
- Nougat is unmaintained, hallucinates on long pages, and falls back to `pymupdf4llm` on
  failure — which flattens LaTeX entirely.

## 3. Math-aware chunking

`RecursiveCharacterTextSplitter` at chunk_size=400 / overlap=80 splits on generic
separators. It will happily cut a proof in half, or sever a formula from the sentence that
defines its symbols.

- Split on **structural boundaries** in the `.mmd`: headings, theorem / definition /
  example blocks, display-math blocks.
- Keep each **worked example intact**.
- **Prepend the section heading** to every chunk so a chunk carries its own context.

In practice this usually matters more than swapping the embedding model.

## 4. Query rewriting / HyDE

The student asks "how do I do the one with the squiggly line," which matches nothing
lexically and embeds poorly. Have the LLM expand the question — or draft a hypothetical
answer (HyDE) — and retrieve against *that* instead of the raw query.

## 5. Small-to-big retrieval

Search over small, precise chunks but hand the LLM the surrounding **parent section**.
Fixes the "found the right formula, but the LLM lacks the setup" failure mode.

## 6. A math-tuned embedding model

`BAAI/bge-small-en-v1.5` was never trained on LaTeX. Replacing it with a math-tuned
embedder is the natural first target for the fine-tuning you eventually want: a contrastive
fine-tune of the embedder is far cheaper than touching the 7B generator, and the eval set
from §1 is the training data.

## 7. Generation side

- `t1c/deepseek-math-7b-rl:Q4` is a 2023-era model, and **Q4 quantization** costs accuracy
  on exactly the multi-step arithmetic it exists to do. A/B it against a current open math
  model (Qwen2.5-Math-7B and up) at **Q8 or full precision** — the GPU pod can take it.
- The **prompt itself is unexamined**. Asking for step-by-step reasoning and requiring
  citation of the retrieved chunk usually buys more than a model swap.

## 8. Housekeeping: extract `retrieval.py`

Retrieval logic is duplicated between `query.py` and `app.py` across three stages now
(search, RRF, rerank). Drift between them will silently make the CLI and the web UI
disagree. Extract a shared `retrieval.py` imported by both.

This is also a **prerequisite for a clean `eval.py`** — otherwise the pipeline gets written
a fourth time.

## Suggested order

Eval set → Marker → chunking, re-measuring after each. The rest is guided by what the
numbers say.
