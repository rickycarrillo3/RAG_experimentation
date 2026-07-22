"""
eval.py - Measure the retrieval pipeline instead of eyeballing it.

Runs a gold set (from make_evalset.py) against one or more retrieval configs and reports
recall / MRR / nDCG, per-stage latency, and peak VRAM. The question it exists to answer:
does the cross-encoder reranker actually earn the 2.2GB it costs?

    (run from knowledge-base-math/, so the indexes and gold set resolve)
    python evaluation/eval.py --user calctest --config hybrid+rerank
    python evaluation/eval.py --user calctest --all                  # sweep every config
    python evaluation/eval.py --user calctest --all --answers        # also judge answers (slow)

Every run prints a GOLD SET HEALTH header first, on purpose: a recall number is only as
trustworthy as the exam that produced it, and these questions are machine-generated.

Protocol, caveats, and how to read the output: EVALUATION.md
"""

import argparse
import json
import math
import os
import re
import statistics
import sys
import time

# This lives in evaluation/; the pipeline modules (retrieval, query, …) are one level up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from query import SYSTEM_PROMPT
from retrieval import (
    EMBED_MODEL,
    OLLAMA_MODEL,
    RERANK_TOP_C,
    TOP_K,
    chunk_id,
    format_context,
    load_bm25,
    load_chroma,
    load_embeddings,
    load_reranker,
    retrieve_detailed,
)

EVAL_DIR = "evaluation"                             # gold set + review live here
RESULTS_DIR = os.path.join(EVAL_DIR, "results")     # machine-specific run outputs
GOLDSET_PATH = os.path.join(EVAL_DIR, "goldset.jsonl")
JUDGE_MODEL = "qwen2:7b"  # NOT the generator — a model grading its own output is not evidence

# Each config is a set of kwargs for retrieve_detailed.
#   bm25/dense       - is hybrid even beating its own parts?
#   hybrid           - the system BEFORE reranking. The number to beat.
#   hybrid+rerank    - the system as it ships today.
#   dense+rerank     - does BM25 contribute anything once a reranker exists?
#   hybrid_bm25_lite - down-weight BM25 in the fusion rather than dropping it.
#   pool variants    - is a 20-candidate pool the right size?
# top_k caps the pool: RRF emits at most top_k*2 candidates, so raising rerank_top_c
# without raising top_k is a silent no-op.
CONFIGS: dict[str, dict] = {
    "bm25":             dict(use_bm25=True,  use_dense=False, use_rerank=False),
    "dense":            dict(use_bm25=False, use_dense=True,  use_rerank=False),
    "hybrid":           dict(use_bm25=True,  use_dense=True,  use_rerank=False),
    "hybrid+rerank":    dict(use_bm25=True,  use_dense=True,  use_rerank=True,  top_k=TOP_K, rerank_top_c=RERANK_TOP_C),
    "dense+rerank":     dict(use_bm25=False, use_dense=True,  use_rerank=True,  top_k=TOP_K, rerank_top_c=RERANK_TOP_C),
    "hybrid_bm25_lite": dict(use_bm25=True,  use_dense=True,  use_rerank=True,  rrf_weights=[0.3, 1.0]),
    "rerank_pool10":    dict(use_bm25=True,  use_dense=True,  use_rerank=True,  top_k=5,  rerank_top_c=10),
    "rerank_pool50":    dict(use_bm25=True,  use_dense=True,  use_rerank=True,  top_k=25, rerank_top_c=50),
}

JUDGE_PROMPT = """You are grading a math tutor's answer for factual correctness.

Question: {question}

Reference passage (the ground truth the answer should agree with):
---
{reference}
---

Tutor's answer:
---
{answer}
---

Grade the answer 1-5 on whether it is factually correct and supported by the reference:
5 = fully correct and grounded in the reference
4 = correct, minor imprecision
3 = partially correct, or correct but missing the key point
2 = mostly wrong, or unsupported by the reference
1 = wrong, or refuses/fails to answer

Output ONLY the single digit. No explanation."""


# ── Gold matching ─────────────────────────────────────────────────────────────

def neighbours(gold_id: str) -> set[str]:
    """The gold chunk plus its immediate neighbours (`<source>::<n>` ± 1).

    Chunking at 400 chars with 80 overlap means an answer routinely straddles two
    chunks. Under strict matching, a config that retrieves the adjacent half of the
    same worked example scores a total miss — which understates recall and can rank
    configs wrongly. `soft` credits the neighbours; `strict` does not. We report both,
    because a large gap between them is itself a finding — about chunking, not retrieval.
    """
    m = re.match(r"^(.*)::(\d+)$", gold_id)
    if not m:
        return {gold_id}
    source, n = m.group(1), int(m.group(2))
    ids = {gold_id}
    if n > 0:
        ids.add(f"{source}::{n - 1}")
    ids.add(f"{source}::{n + 1}")
    return ids


def rank_of(ranked: list, targets: set[str]) -> int | None:
    """1-indexed rank of the first chunk in `ranked` that is in `targets`."""
    for i, (doc, _) in enumerate(ranked, start=1):
        if chunk_id(doc) in targets:
            return i
    return None


# Overlap matching — needed because chunk ids are NOT comparable across chunking
# strategies. A retrieved chunk counts as the gold hit when it *contains* enough of
# the gold passage's tokens, so the same frozen gold set scores every chunking fairly.
OVERLAP_STRICT = 0.7   # the retrieved chunk holds ≥70% of the gold passage's tokens
OVERLAP_SOFT = 0.5     # partial: the answer straddles two chunks (the eqaware failure mode)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    """Lowercase alphanumeric token set. Drops LaTeX punctuation so raw `$\\theta$` and a
    normalized `θ`/`theta` both reduce to comparable content tokens."""
    return set(_TOKEN_RE.findall(text.lower()))


def overlap_containment(gold_text: str, chunk_text: str) -> float:
    """Fraction of the gold passage's tokens present in `chunk_text` (0..1)."""
    gold = _tokens(gold_text)
    if not gold:
        return 0.0
    return len(gold & _tokens(chunk_text)) / len(gold)


def rank_of_overlap(ranked: list, gold_text: str, threshold: float) -> int | None:
    """1-indexed rank of the first chunk whose token-containment of gold ≥ threshold."""
    for i, (doc, _) in enumerate(ranked, start=1):
        if overlap_containment(gold_text, doc.page_content) >= threshold:
            return i
    return None


def ndcg_at_k(rank: int | None, k: int = 5) -> float:
    """With exactly one relevant document, IDCG is 1, so nDCG@k = 1/log2(rank+1)."""
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def peak_vram_gb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    return None  # CPU / MPS: no comparable counter. Watch nvidia-smi on the pod.


# ── Evaluation ────────────────────────────────────────────────────────────────

def load_goldset(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No gold set at {path}. Run make_evalset.py first.")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def print_goldset_health(goldset: list[dict], path: str) -> None:
    """Never let a recall number be read without the quality of the exam beside it."""
    n = len(goldset)
    leaks = [g.get("leak_score") for g in goldset if g.get("leak_score") is not None]
    n_math = sum(1 for g in goldset if g.get("is_math"))

    print(f"\n{'═' * 64}")
    print(f"GOLD SET HEALTH — {path}")
    print(f"  questions: {n}   math-bearing: {n_math}")
    if leaks:
        mean_leak = sum(leaks) / len(leaks)
        high = sum(1 for x in leaks if x >= 0.6)
        print(f"  mean leak score: {mean_leak:.2f}   high-leak (>=0.60): {high}/{len(leaks)}")
        if mean_leak >= 0.4 or high > n * 0.15:
            print("  ⚠ LEAKY. Questions reuse their chunk's wording, which flatters BM25 and")
            print("    understates the reranker. Trust the DELTAS between configs, not the")
            print("    absolute scores. Clean evaluation/goldset_review.md.")
    else:
        print("  leak scores absent (gold set predates the leakage filter) — regenerate it.")
    print(f"{'═' * 64}")


def run_config(
    name: str,
    goldset: list[dict],
    user: str,
    embeddings,
    reranker,
    bm25_index,
    store,
    answer_chain=None,
    judge=None,
    match: str = "overlap",
) -> tuple[dict, list[dict]]:
    cfg = CONFIGS[name]
    strict_ranks: list[int | None] = []
    soft_ranks: list[int | None] = []
    in_pool: list[bool] = []
    pool_sizes: list[int] = []
    stage_times: dict[str, list[float]] = {}
    judge_scores: list[int] = []
    gen_times: list[float] = []
    failures: list[dict] = []

    for item in goldset:
        result = retrieve_detailed(
            item["q"],
            user=user,
            embeddings=embeddings,
            reranker=reranker,
            bm25_index=bm25_index,
            store=store,
            **cfg,
        )

        gold = item["gold_chunk_id"]
        if match == "overlap":
            gold_text = item["chunk_text"]
            s_rank = rank_of_overlap(result.ranked, gold_text, OVERLAP_STRICT)
            soft_rank = rank_of_overlap(result.ranked, gold_text, OVERLAP_SOFT)
            pooled = rank_of_overlap(result.candidates, gold_text, OVERLAP_STRICT) is not None
        else:
            s_rank = rank_of(result.ranked, {gold})
            soft_rank = rank_of(result.ranked, neighbours(gold))
            pooled = rank_of(result.candidates, {gold}) is not None
        strict_ranks.append(s_rank)
        soft_ranks.append(soft_rank)
        in_pool.append(pooled)
        pool_sizes.append(len(result.candidates))

        for stage, secs in result.timings.items():
            stage_times.setdefault(stage, []).append(secs)

        # Aggregates say THAT something is wrong; this says WHAT.
        if s_rank is None or s_rank > 5:
            failures.append({
                "q": item["q"],
                "gold_chunk_id": gold,
                "gold_rank": s_rank,
                "gold_rank_soft": soft_rank,
                "gold_in_pool": in_pool[-1],
                "leak_score": item.get("leak_score"),
                "retrieved_instead": [
                    {"chunk_id": chunk_id(d), "score": round(s, 4),
                     "preview": d.page_content[:120].replace("\n", " ")}
                    for d, s in result.final
                ],
            })

        if answer_chain is not None and judge is not None:
            t0 = time.perf_counter()
            answer = answer_chain.invoke({
                "context": format_context(result.final),
                "input": item["q"],
            })
            gen_times.append(time.perf_counter() - t0)

            raw = judge.invoke({
                "question": item["q"],
                "reference": item["chunk_text"],
                "answer": answer,
            })
            digits = [c for c in raw if c.isdigit()]
            if digits:
                judge_scores.append(max(1, min(5, int(digits[0]))))

    n = len(strict_ranks)
    found = [r for r in strict_ranks if r is not None]

    metrics = {
        "config": name,
        "match": match,
        "n": n,
        "pool_size": round(statistics.mean(pool_sizes), 1),
        "recall@1": sum(1 for r in strict_ranks if r == 1) / n,
        "recall@5": sum(1 for r in strict_ranks if r is not None and r <= 5) / n,
        "recall@5_soft": sum(1 for r in soft_ranks if r is not None and r <= 5) / n,
        "recall@pool": sum(in_pool) / n,
        "mrr": sum(1.0 / r for r in found) / n,
        "ndcg@5": sum(ndcg_at_k(r, 5) for r in strict_ranks) / n,
        "latency_ms": {
            stage: round(statistics.mean(times) * 1000, 2)
            for stage, times in stage_times.items()
        },
    }
    metrics["latency_ms"]["retrieval_total"] = round(sum(metrics["latency_ms"].values()), 2)

    if judge_scores:
        metrics["answer_score_1to5"] = round(statistics.mean(judge_scores), 2)
        metrics["answer_n_judged"] = len(judge_scores)
        metrics["latency_ms"]["generation"] = round(statistics.mean(gen_times) * 1000, 2)

    vram = peak_vram_gb()
    if vram is not None:
        metrics["peak_vram_gb"] = round(vram, 2)

    return metrics, failures


def print_table(results: list[dict]) -> None:
    has_answers = any("answer_score_1to5" in r for r in results)

    header = (f"{'config':<18} {'pool':>5} {'R@1':>6} {'R@5':>6} {'R@5soft':>8} "
              f"{'R@pool':>7} {'MRR':>6} {'nDCG@5':>7} {'retr ms':>8}")
    if has_answers:
        header += f" {'answer':>7}"
    print("\n" + header)
    print("─" * len(header))

    for r in results:
        line = (
            f"{r['config']:<18} {r['pool_size']:>5.0f} "
            f"{r['recall@1']:>6.2f} {r['recall@5']:>6.2f} {r['recall@5_soft']:>8.2f} "
            f"{r['recall@pool']:>7.2f} {r['mrr']:>6.3f} {r['ndcg@5']:>7.3f} "
            f"{r['latency_ms']['retrieval_total']:>8.1f}"
        )
        if has_answers:
            score = r.get("answer_score_1to5")
            line += f" {score:>7.2f}" if score is not None else f" {'—':>7}"
        print(line)

    print("\nR@5soft credits an adjacent chunk (gold ±1) — chunks overlap, so an answer often")
    print("  straddles two. A big gap vs R@5 is a CHUNKING problem, not a retrieval one.")
    print("R@pool = was the gold chunk in the candidate pool the reranker saw at all? If this")
    print("  is low, reranking cannot help — the miss happened upstream. Pool sizes differ by")
    print("  config (single-retriever configs have half the candidates), hence the pool column.")

    by = {r["config"]: r for r in results}

    base, rr = by.get("hybrid"), by.get("hybrid+rerank")
    if base and rr:
        d5 = rr["recall@5"] - base["recall@5"]
        dm = rr["mrr"] - base["mrr"]
        print(f"\nVERDICT — reranker vs plain hybrid:")
        print(f"  recall@5 {base['recall@5']:.2f} → {rr['recall@5']:.2f}  ({d5:+.2f})")
        print(f"  MRR      {base['mrr']:.3f} → {rr['mrr']:.3f}  ({dm:+.3f})")
        print(f"  cost     +{rr['latency_ms'].get('rerank', 0):.0f} ms/query, +2.2GB VRAM & disk")
        if d5 <= 0 and dm <= 0.01:
            print("  → Not paying for itself. Drop it, or fix retrieval first.")

    dr = by.get("dense+rerank")
    if rr and dr:
        d5 = rr["recall@5"] - dr["recall@5"]
        print(f"\nDOES BM25 EARN ITS PLACE? hybrid+rerank vs dense+rerank:")
        print(f"  recall@5 {dr['recall@5']:.2f} (dense only) → {rr['recall@5']:.2f} (with BM25)  ({d5:+.2f})")
        if d5 < 0:
            print("  → BM25 is HURTING. It is dragging the fusion below dense alone. Consider")
            print("    dropping it, or see hybrid_bm25_lite (down-weighted) below.")
        elif d5 == 0:
            print("  → BM25 adds nothing here. It is free to run, but it is not helping either.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG retrieval pipeline.")
    parser.add_argument("--user", required=True, help="Username whose index to evaluate against")
    parser.add_argument("--config", help=f"One of: {', '.join(CONFIGS)}")
    parser.add_argument("--all", action="store_true", help="Run every config")
    parser.add_argument("--answers", action="store_true", help="Also generate + judge answers (slow)")
    parser.add_argument("--goldset", default=GOLDSET_PATH)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--match", choices=["overlap", "id"], default="overlap",
                        help="How a retrieval counts as the gold hit. 'overlap' (default) is "
                             "chunking-invariant; 'id' matches exact chunk ids (same-chunking only).")
    parser.add_argument("--embed-model", default=EMBED_MODEL,
                        help="Embedding model the target index was built with (must match).")
    parser.add_argument("--normalize-latex", action="store_true",
                        help="Query-side LaTeX normalization; must match how the index was built.")
    args = parser.parse_args()

    if not args.all and not args.config:
        parser.error("Pass --config <name> or --all")
    names = list(CONFIGS) if args.all else [args.config]
    for name in names:
        if name not in CONFIGS:
            parser.error(f"Unknown config '{name}'. Choose from: {', '.join(CONFIGS)}")

    goldset = load_goldset(args.goldset)
    print_goldset_health(goldset, args.goldset)

    print(f"\nLoading embedding model '{args.embed_model}' (normalize_latex={args.normalize_latex})...")
    embeddings = load_embeddings(args.embed_model, normalize_latex=args.normalize_latex)
    print("Loading reranker...")
    reranker = load_reranker()

    # Load the indexes once: per-query reloading would swamp the latency numbers.
    bm25_index = load_bm25(args.user)
    store = load_chroma(args.user, embeddings)

    answer_chain = judge = None
    if args.answers:
        print(f"Loading generator '{OLLAMA_MODEL}' and judge '{args.judge_model}'...")
        answer_chain = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]) | ChatOllama(model=OLLAMA_MODEL, temperature=0, num_predict=1024) | StrOutputParser()
        judge = ChatPromptTemplate.from_template(JUDGE_PROMPT) | ChatOllama(
            model=args.judge_model, temperature=0, num_predict=5
        ) | StrOutputParser()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []
    for name in names:
        print(f"\nRunning config '{name}'...")
        t0 = time.perf_counter()
        metrics, failures = run_config(
            name, goldset, args.user, embeddings, reranker,
            bm25_index, store, answer_chain, judge, match=args.match,
        )
        metrics["wall_clock_s"] = round(time.perf_counter() - t0, 1)
        results.append(metrics)

        slug = name.replace("+", "_")
        with open(os.path.join(RESULTS_DIR, f"results_{slug}.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(RESULTS_DIR, f"failures_{slug}.json"), "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)
        print(f"  done in {metrics['wall_clock_s']}s → {RESULTS_DIR}/results_{slug}.json "
              f"({len(failures)} misses → {RESULTS_DIR}/failures_{slug}.json)")

    print_table(results)


if __name__ == "__main__":
    main()
