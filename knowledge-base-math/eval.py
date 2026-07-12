"""
eval.py - Measure the retrieval pipeline instead of eyeballing it.

Runs a gold set (from make_evalset.py) against one or more retrieval configs and reports
recall / MRR / nDCG, per-stage latency, and peak VRAM. The question it exists to answer:
does the cross-encoder reranker actually earn the 2.2GB it costs?

    python eval.py --user test --config hybrid+rerank
    python eval.py --user test --all                    # sweep every config, print table
    python eval.py --user test --all --answers          # also score end-to-end answers (slow)

Protocol, caveats, and how to read the output: EVALUATION.md
"""

import argparse
import json
import math
import os
import statistics
import time

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import retrieval
from query import SYSTEM_PROMPT
from retrieval import (
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

EVAL_DIR = "eval"
GOLDSET_PATH = os.path.join(EVAL_DIR, "goldset.jsonl")
JUDGE_MODEL = "qwen2:7b"  # NOT the generator — a model grading its own output is not evidence

# Each config is a set of kwargs for retrieve_detailed. Every one is a real question:
#   bm25/dense      - is hybrid even beating its own parts?
#   hybrid          - the system BEFORE reranking. The number to beat.
#   hybrid+rerank   - the system as it ships today.
#   pool variants   - is a 20-candidate pool the right size?
# Note top_k caps the pool: RRF emits at most top_k*2 candidates, so widening
# rerank_top_c without raising top_k does nothing at all.
CONFIGS: dict[str, dict] = {
    "bm25":            dict(use_bm25=True,  use_dense=False, use_rerank=False),
    "dense":           dict(use_bm25=False, use_dense=True,  use_rerank=False),
    "hybrid":          dict(use_bm25=True,  use_dense=True,  use_rerank=False),
    "hybrid+rerank":   dict(use_bm25=True,  use_dense=True,  use_rerank=True,  top_k=TOP_K, rerank_top_c=RERANK_TOP_C),
    "rerank_pool10":   dict(use_bm25=True,  use_dense=True,  use_rerank=True,  top_k=5,     rerank_top_c=10),
    "rerank_pool50":   dict(use_bm25=True,  use_dense=True,  use_rerank=True,  top_k=25,    rerank_top_c=50),
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


# ── Metrics ───────────────────────────────────────────────────────────────────

def gold_rank(ranked: list, gold_id: str) -> int | None:
    """1-indexed rank of the gold chunk in a ranked list, or None if absent."""
    for i, (doc, _) in enumerate(ranked, start=1):
        if chunk_id(doc) == gold_id:
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
) -> dict:
    cfg = CONFIGS[name]
    ranks: list[int | None] = []
    in_pool: list[bool] = []
    stage_times: dict[str, list[float]] = {}
    judge_scores: list[int] = []
    gen_times: list[float] = []

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
        ranks.append(gold_rank(result.ranked, gold))
        in_pool.append(gold_rank(result.candidates, gold) is not None)

        for stage, secs in result.timings.items():
            stage_times.setdefault(stage, []).append(secs)

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

    n = len(ranks)
    found = [r for r in ranks if r is not None]

    metrics = {
        "config": name,
        "n": n,
        "recall@1": sum(1 for r in ranks if r == 1) / n,
        "recall@5": sum(1 for r in ranks if r is not None and r <= 5) / n,
        "recall@pool": sum(in_pool) / n,
        "mrr": sum(1.0 / r for r in found) / n,
        "ndcg@5": sum(ndcg_at_k(r, 5) for r in ranks) / n,
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

    return metrics


def print_table(results: list[dict]) -> None:
    has_answers = any("answer_score_1to5" in r for r in results)

    header = f"{'config':<16} {'R@1':>6} {'R@5':>6} {'R@pool':>7} {'MRR':>6} {'nDCG@5':>7} {'retr ms':>8}"
    if has_answers:
        header += f" {'answer':>7}"
    print("\n" + header)
    print("─" * len(header))

    for r in results:
        line = (
            f"{r['config']:<16} "
            f"{r['recall@1']:>6.2f} {r['recall@5']:>6.2f} {r['recall@pool']:>7.2f} "
            f"{r['mrr']:>6.3f} {r['ndcg@5']:>7.3f} "
            f"{r['latency_ms']['retrieval_total']:>8.1f}"
        )
        if has_answers:
            score = r.get("answer_score_1to5")
            line += f" {score:>7.2f}" if score is not None else f" {'—':>7}"
        print(line)

    print("\nR@pool = was the gold chunk in the candidate pool the reranker saw at all?")
    print("If R@pool is low, the reranker cannot help — the miss happened upstream, in")
    print("retrieval or chunking. Reranking can only reorder what BM25/dense already found.")

    base = next((r for r in results if r["config"] == "hybrid"), None)
    rr = next((r for r in results if r["config"] == "hybrid+rerank"), None)
    if base and rr:
        d_r5 = rr["recall@5"] - base["recall@5"]
        d_mrr = rr["mrr"] - base["mrr"]
        cost = rr["latency_ms"].get("rerank", 0.0)
        print(f"\nVERDICT — reranker vs plain hybrid:")
        print(f"  recall@5 {base['recall@5']:.2f} → {rr['recall@5']:.2f}  ({d_r5:+.2f})")
        print(f"  MRR      {base['mrr']:.3f} → {rr['mrr']:.3f}  ({d_mrr:+.3f})")
        print(f"  cost     +{cost:.0f} ms/query, +2.2GB VRAM & disk")
        if d_r5 <= 0 and d_mrr <= 0.01:
            print("  → Not paying for itself on this gold set. Drop it, or fix retrieval first.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG retrieval pipeline.")
    parser.add_argument("--user", required=True, help="Username whose index to evaluate against")
    parser.add_argument("--config", help=f"One of: {', '.join(CONFIGS)}")
    parser.add_argument("--all", action="store_true", help="Run every config")
    parser.add_argument("--answers", action="store_true", help="Also generate + judge answers (slow)")
    parser.add_argument("--goldset", default=GOLDSET_PATH)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    args = parser.parse_args()

    if not args.all and not args.config:
        parser.error("Pass --config <name> or --all")
    names = list(CONFIGS) if args.all else [args.config]
    for name in names:
        if name not in CONFIGS:
            parser.error(f"Unknown config '{name}'. Choose from: {', '.join(CONFIGS)}")

    goldset = load_goldset(args.goldset)
    print(f"Gold set: {len(goldset)} questions from {args.goldset}")

    print("Loading embedding model...")
    embeddings = load_embeddings()
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

    os.makedirs(EVAL_DIR, exist_ok=True)
    results = []
    for name in names:
        print(f"\nRunning config '{name}'...")
        t0 = time.perf_counter()
        metrics = run_config(
            name, goldset, args.user, embeddings, reranker,
            bm25_index, store, answer_chain, judge,
        )
        metrics["wall_clock_s"] = round(time.perf_counter() - t0, 1)
        results.append(metrics)

        out_path = os.path.join(EVAL_DIR, f"results_{name.replace('+', '_')}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"  done in {metrics['wall_clock_s']}s → {out_path}")

    print_table(results)


if __name__ == "__main__":
    main()
