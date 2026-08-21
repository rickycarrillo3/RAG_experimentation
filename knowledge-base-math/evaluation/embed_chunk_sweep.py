"""
embed_chunk_sweep.py - Does a math-aware index beat the generic one? Measure it.

Sweeps 3 chunking strategies x 3 embedding configs (9 indexes), evaluating each at
dense-only (isolates the embedding effect — the hypothesis) and hybrid+rerank (what
ships). Reuses the frozen, hand-cleaned gold set via overlap matching, so the exam is
identical across every chunking. Reports retrieval quality AND latency, because a
bigger embedder that wins recall but triples query time is a real tradeoff, not a
free win.

    python evaluation/embed_chunk_sweep.py
    python evaluation/embed_chunk_sweep.py --doc docs/extracted/calculus_chainrule.mmd

Run from knowledge-base-math/ with the venv active. Writes evaluation/results/sweep_results.json.
"""

import argparse
import json
import os
import sys
import time

# This lives in evaluation/; the pipeline modules (chunking, ingest, retrieval) are one level up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import load_goldset, run_config
from langchain_chroma import Chroma

from ingest import BM25_DIR, CHROMA_DIR, ingest
from kbm.chunking import CHUNKERS
from kbm.retrieval import load_bm25, load_chroma, load_embeddings, load_reranker

# (key, model, normalize_latex) — key names the index/collection.
EMBED_CONFIGS = [
    ("bgesmall", "BAAI/bge-small-en-v1.5", False),
    ("bgem3", "BAAI/bge-m3", False),
    ("bgem3norm", "BAAI/bge-m3", True),
]
CHUNK_KEYS = list(CHUNKERS)                 # baseline, eqaware, eqaware_context
EVAL_CONFIGS = ["dense", "hybrid+rerank"]   # isolate embedding, then shipping reality

DEFAULT_DOC = "docs/extracted/calculus_chainrule.mmd"
GOLDSET_PATH = os.path.join("evaluation", "goldset.jsonl")
OUT_PATH = os.path.join("evaluation", "results", "sweep_results.json")


def reset_index(user: str, embeddings) -> None:
    """Drop any existing collection + BM25 pickle so a re-run doesn't double-ingest."""
    try:
        Chroma(collection_name=f"user_{user}", embedding_function=embeddings,
               persist_directory=CHROMA_DIR).delete_collection()
    except Exception:
        pass
    pkl = os.path.join(BM25_DIR, f"user_{user}.pkl")
    if os.path.exists(pkl):
        os.remove(pkl)


def bm25_kb(user: str) -> float:
    pkl = os.path.join(BM25_DIR, f"user_{user}.pkl")
    return round(os.path.getsize(pkl) / 1024, 1) if os.path.exists(pkl) else 0.0


def main():
    parser = argparse.ArgumentParser(description="Sweep chunking x embedding for equation retrieval.")
    parser.add_argument("--doc", default=DEFAULT_DOC, help="Source .mmd the gold set was built from")
    parser.add_argument("--goldset", default=GOLDSET_PATH)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.doc):
        parser.error(f"Source doc not found: {args.doc}")
    goldset = load_goldset(args.goldset)
    print(f"Gold set: {len(goldset)} questions ({args.goldset})")
    print(f"Source doc: {args.doc}")
    print(f"Sweep: {len(CHUNK_KEYS)} chunkers x {len(EMBED_CONFIGS)} embeds "
          f"x {len(EVAL_CONFIGS)} eval configs = "
          f"{len(CHUNK_KEYS) * len(EMBED_CONFIGS) * len(EVAL_CONFIGS)} runs\n")

    print("Loading reranker (once)...")
    reranker = load_reranker()

    rows = []
    for embed_key, model, norm in EMBED_CONFIGS:
        print(f"\n{'#' * 70}\n# embedding: {embed_key}  ({model}, normalize_latex={norm})\n{'#' * 70}")
        embeddings = load_embeddings(model, normalize_latex=norm)

        for chunk_key in CHUNK_KEYS:
            user = f"sweep_{chunk_key}_{embed_key}"
            print(f"\n--- ingest [{chunk_key} x {embed_key}] → user '{user}' ---")
            reset_index(user, embeddings)
            stats = ingest(user, [args.doc], chunker=chunk_key, embed_model=model, normalize_latex=norm)

            bm25_index = load_bm25(user)
            store = load_chroma(user, embeddings)

            for cfg in EVAL_CONFIGS:
                t0 = time.perf_counter()
                metrics, _ = run_config(cfg, goldset, user, embeddings, reranker,
                                        bm25_index, store, match="overlap")
                lat = metrics["latency_ms"]
                rows.append({
                    "chunker": chunk_key,
                    "embed": embed_key,
                    "config": cfg,
                    "n_chunks": stats["n_chunks"],
                    "recall@1": metrics["recall@1"],
                    "recall@5": metrics["recall@5"],
                    "recall@5_soft": metrics["recall@5_soft"],
                    "recall@pool": metrics["recall@pool"],
                    "mrr": metrics["mrr"],
                    "ndcg@5": metrics["ndcg@5"],
                    "dense_ms": lat.get("dense", 0.0),
                    "rerank_ms": lat.get("rerank", 0.0),
                    "retrieval_ms": lat["retrieval_total"],
                    "build_s": stats["build_seconds"],
                    "bm25_kb": bm25_kb(user),
                })
                print(f"    {cfg:14s} R@5={metrics['recall@5']:.2f} "
                      f"R@pool={metrics['recall@pool']:.2f} MRR={metrics['mrr']:.3f} "
                      f"dense={lat.get('dense', 0):.0f}ms in {time.perf_counter() - t0:.1f}s")

    payload = {
        "doc": args.doc,
        "goldset": args.goldset,
        "n_questions": len(goldset),
        "embed_configs": [{"key": k, "model": m, "normalize_latex": n} for k, m, n in EMBED_CONFIGS],
        "chunkers": CHUNK_KEYS,
        "eval_configs": EVAL_CONFIGS,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print_table(rows)
    print(f"\nFull results → {args.out}")


def print_table(rows: list[dict]) -> None:
    hdr = (f"{'chunker':<16} {'embed':<10} {'config':<14} {'chunks':>6} "
           f"{'R@1':>5} {'R@5':>5} {'R@5s':>5} {'R@pool':>7} {'MRR':>6} {'nDCG':>6} "
           f"{'dense':>7} {'retr':>7} {'build':>6}")
    print("\n" + hdr)
    print("─" * len(hdr))
    for r in rows:
        print(f"{r['chunker']:<16} {r['embed']:<10} {r['config']:<14} {r['n_chunks']:>6} "
              f"{r['recall@1']:>5.2f} {r['recall@5']:>5.2f} {r['recall@5_soft']:>5.2f} "
              f"{r['recall@pool']:>7.2f} {r['mrr']:>6.3f} {r['ndcg@5']:>6.3f} "
              f"{r['dense_ms']:>6.0f}m {r['retrieval_ms']:>6.0f}m {r['build_s']:>5.1f}s")
    print("\ndense = query-embed latency (bge-m3 is ~17x the params of bge-small — watch this).")
    print("Compare within a config column: dense-only isolates the embedding; hybrid+rerank is")
    print("what ships. Ids differ across chunkings, so hits are scored by token overlap (≥0.70).")


if __name__ == "__main__":
    main()
