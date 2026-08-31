"""
evaluation/rerank_cpu_bench.py - What does the cross-encoder cost on a CPU host?

Why this exists. `LATENCY.md §4` declines to swap the reranker on the grounds that it is
"850 ms of a 20 s problem — the wrong end." That reasoning holds only while generation
sits on the same box. Move generation to a serverless GPU and leave retrieval on a small
always-on CPU pod and the arithmetic inverts: the reranker becomes the largest thing left
on the critical path, and it runs on the weakest hardware in the system. In agent mode it
runs again mid-answer, once per `search_documents` round.

So this measures the two things that decide the swap:

  latency  — per query, scoring a realistic RRF pool of RERANK_TOP_C pairs, on CPU
  quality  — recall@5 on the gold set, plus top-5 agreement with the incumbent

Neither alone can decide it. A model 20x faster that drops the right chunk is not a win,
and the incumbent's own measured pairwise AUC on human math-relevance grades is **0.60**
(`evaluation/calibrate_floor.py`) — barely better than chance — so "the big one is
obviously better" is exactly the assumption worth testing rather than inheriting.

    python evaluation/rerank_cpu_bench.py --user apitest
    python evaluation/rerank_cpu_bench.py --user apitest --threads 4

**Each model is scored in a separate process.** Two reasons, one of them learned the hard
way: a native crash in one model would otherwise take the whole run with it and lose the
rows already measured (`cross-encoder/ms-marco-MiniLM-L-6-v2` SIGBUSes on an arm64 Mac —
reproducible across Python 3.12/3.14, torch 2.8/2.13, SDPA and eager attention, with the
safetensors blob verified intact, so it is an environment anomaly rather than a property
of the model, and it is expected to work on the Linux CPU host this is measuring for). The
second reason is cleanliness: several hundred MB of weights per model, freed for certain
between rows rather than at the whim of the garbage collector.

⚠️ TWO CAVEATS ON THE NUMBERS, both of which make them optimistic:

1. **This machine is not the target machine.** A development Mac's cores are considerably
   faster than 4 shared vCPUs on a cloud CPU pod. Treat the ratios between models as the
   transferable result and the absolute milliseconds as a floor. `--threads` brackets it:
   run at the vCPU count you intend to buy.
2. **The corpus is 66 chunks.** The RRF pool is 20 of them, so a third of the document is
   in every pool and recall is inflated against a real textbook. The *ranking* comparison
   between models is still meaningful; the absolute recall is not a deployment number.

⚠️ AND ONE THING THIS SCRIPT CANNOT TELL YOU. `KBM_RELEVANCE_FLOOR` (0.15) is calibrated
to bge-reranker-v2-m3's sigmoid output scale, and abstention depends on it. A different
cross-encoder has a different scale, so switching models **silently breaks abstention**
until `evaluation/calibrate_floor.py` is re-run. Read the score distributions this script
prints as evidence for that, not as a substitute for the calibration.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

# This lives in evaluation/; the pipeline modules (kbm, …) are one level up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import CrossEncoder  # noqa: E402

from kbm.retrieval import (  # noqa: E402
    RERANK_MODEL,
    RERANK_TOP_C,
    TOP_K,
    TOP_N,
    bm25_search,
    dense_search,
    load_bm25,
    load_chroma,
    load_embeddings,
    reciprocal_rank_fusion,
)

GOLDSET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldset.jsonl")

# The incumbent first — every other row is reported relative to it.
CANDIDATES = [
    (RERANK_MODEL, "incumbent, multilingual"),
    ("BAAI/bge-reranker-base", "same family, half the size"),
    ("mixedbread-ai/mxbai-rerank-xsmall-v1", "8x smaller"),
    ("jinaai/jina-reranker-v1-tiny-en", "17x smaller, English-only"),
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", "25x smaller, the classic MS MARCO baseline"),
]


def build_pools(user: str, questions: list[str]) -> list[list]:
    """The RRF candidate pool each reranker will be asked to score.

    Built once and shared, so every model is measured on identical input — the same rule
    self_consistency.py follows for sampling. Uses the pipeline's own functions rather
    than a local reimplementation (CLAUDE.md: an eval that measures a reimplementation
    drifts from what ships and quietly starts lying).
    """
    bm25, chunks = load_bm25(user)
    embeddings = load_embeddings()
    store = load_chroma(user, embeddings)

    pools = []
    for q in questions:
        fused = reciprocal_rank_fusion([
            bm25_search(q, bm25, chunks, TOP_K),
            dense_search(q, store, TOP_K),
        ])
        pools.append(fused[:RERANK_TOP_C])
    return pools


def gold_hit(docs, row) -> bool:
    """Is the gold chunk among these? Exact chunk_id when the chunking matches, else
    eval.py's token-containment >= 0.70 against the gold chunk_text."""
    gold_id = row.get("gold_chunk_id")
    for d in docs:
        if gold_id and d.metadata.get("chunk_id") == gold_id:
            return True
    gold_tokens = set(row.get("chunk_text", "").split())
    if not gold_tokens:
        return False
    for d in docs:
        doc_tokens = set(d.page_content.split())
        if len(gold_tokens & doc_tokens) / len(gold_tokens) >= 0.70:
            return True
    return False


def score_one(model_name: str, payload_path: str, threads: int, repeats: int,
              device: str = "cpu") -> dict:
    """Run ONE model in a child process. Returns its result dict, or {"crashed": ...}."""
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--worker", model_name,
         "--payload", payload_path, "--threads", str(threads), "--repeats", str(repeats),
         "--device", device],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return {"crashed": True, "returncode": proc.returncode,
                "stderr": proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    return {"crashed": True, "returncode": 0, "stderr": "worker produced no result line"}


def worker(model_name: str, payload_path: str, threads: int, repeats: int,
           device: str = "cpu") -> int:
    """Child process: score every pool with one model and print a JSON result line."""
    import torch
    if threads:
        torch.set_num_threads(threads)

    payload = json.load(open(payload_path))
    # device="cpu" explicitly, NOT load_reranker(): that calls resolve_device(), which
    # would quietly pick MPS on this Mac and CUDA on a pod, measuring the wrong thing.
    model = CrossEncoder(model_name, device=device)
    n_params = sum(p.numel() for p in model.model.parameters())

    # Warm up: the first predict() pays lazy init and would otherwise land in the p95.
    model.predict([(payload[0]["q"], payload[0]["texts"][0])])

    per_query_ms, top5_ids, all_scores = [], [], []
    for item in payload:
        pairs = [(item["q"], t) for t in item["texts"]]
        timings = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            scores = model.predict(pairs)
            timings.append((time.perf_counter() - t0) * 1000)
        per_query_ms.append(statistics.median(timings))

        ranked = sorted(zip(item["ids"], (float(x) for x in scores)),
                        key=lambda x: x[1], reverse=True)
        top5_ids.append([i for i, _ in ranked[:TOP_N]])
        all_scores.extend(s for _, s in ranked)

    print("__RESULT__" + json.dumps({
        "model": model_name,
        "params_m": n_params / 1e6,
        "p50_ms": statistics.median(per_query_ms),
        "p95_ms": sorted(per_query_ms)[max(0, int(len(per_query_ms) * 0.95) - 1)],
        "top5_ids": top5_ids,
        "score_min": min(all_scores), "score_max": max(all_scores),
        "score_median": statistics.median(all_scores),
    }))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="CPU cost and quality of the cross-encoder.")
    ap.add_argument("--user", help="Index to build candidate pools from")
    ap.add_argument("--goldset", default=GOLDSET_PATH)
    ap.add_argument("--threads", type=int, default=0,
                    help="torch CPU threads (0 = leave torch's default). Set this to the "
                         "vCPU count of the host you intend to buy.")
    ap.add_argument("--repeats", type=int, default=3, help="timed passes per query")
    ap.add_argument("--device", default="cpu",
                    help="cpu (the CPU-host question) | mps | cuda. The point of the "
                         "comparison: the cross-encoder is stateless, so unlike BM25 and "
                         "Chroma it is free to run anywhere.")
    ap.add_argument("--worker", help=argparse.SUPPRESS)   # internal: score one model
    ap.add_argument("--payload", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        return worker(args.worker, args.payload, args.threads, args.repeats, args.device)
    if not args.user:
        ap.error("--user is required")

    rows = [json.loads(l) for l in open(args.goldset) if l.strip()]
    questions = [r["q"] for r in rows]
    print(f"Gold set : {len(rows)} questions from {os.path.basename(args.goldset)}")
    print(f"Device   : {args.device}, torch threads = {args.threads or 'default'}")
    print(f"Pool     : RRF top-{RERANK_TOP_C} -> rerank -> top-{TOP_N}\n")

    print("Building candidate pools (shared across all models)...")
    pools = build_pools(args.user, questions)

    # One payload, reused by every child, so all models are scored on identical input.
    payload = []
    for row, pool in zip(rows, pools):
        payload.append({
            "q": row["q"],
            "ids": [d.metadata.get("chunk_id") or d.page_content[:120] for d, _ in pool],
            "texts": [d.page_content for d, _ in pool],
        })
    payload_path = os.path.join(os.path.dirname(__file__), ".rerank_bench_pools.json")
    json.dump(payload, open(payload_path, "w"))

    gold_ids = []
    for row, pool in zip(rows, pools):
        by_id = {}
        for d, _ in pool:
            by_id[d.metadata.get("chunk_id") or d.page_content[:120]] = d
        gold_ids.append({k for k, d in by_id.items() if gold_hit([d], row)})

    sizes = [len(p["texts"]) for p in payload]
    med_chars = statistics.median(len(t) for p in payload for t in p["texts"])
    print(f"  pools: {min(sizes)}-{max(sizes)} candidates, median chunk {med_chars:.0f} chars\n")

    results, baseline = [], None
    for model_name, note in CANDIDATES:
        print(f"── {model_name}  ({note})")
        r = score_one(model_name, payload_path, args.threads, args.repeats, args.device)
        if r.get("crashed"):
            print(f"   ✗ CRASHED (exit {r['returncode']}) — {r['stderr'][:90]}")
            print("     Reported, not fatal: the other rows below are unaffected.\n")
            results.append({"model": model_name, "crashed": True})
            continue

        hits = sum(bool(set(t5) & g) for t5, g in zip(r["top5_ids"], gold_ids))
        r["recall5"] = hits / len(rows)
        r["agreement"] = (
            statistics.mean(len(set(a) & set(b)) / TOP_N
                            for a, b in zip(baseline, r["top5_ids"]))
            if baseline else 1.0
        )
        if baseline is None:
            baseline = r["top5_ids"]
        results.append(r)
        print(f"   {r['params_m']:6.1f}M params   p50 {r['p50_ms']:7.1f} ms   "
              f"p95 {r['p95_ms']:7.1f} ms   recall@5 {r['recall5']:.2f}   "
              f"top5 agreement {r['agreement']:.2f}")
        print(f"   score range [{r['score_min']:.3f}, {r['score_max']:.3f}] "
              f"median {r['score_median']:.3f}\n")

    os.remove(payload_path)

    ok = [r for r in results if not r.get("crashed")]
    base = ok[0]
    print("=" * 84)
    print(f"{'model':44s} {'params':>8s} {'p50':>10s} {'vs base':>8s} {'recall@5':>9s}")
    print("-" * 84)
    for r in results:
        if r.get("crashed"):
            print(f"{r['model']:44s} {'—':>8s} {'CRASHED':>10s} {'—':>8s} {'—':>9s}")
            continue
        print(f"{r['model']:44s} {r['params_m']:7.1f}M {r['p50_ms']:9.1f}ms "
              f"{base['p50_ms']/r['p50_ms']:7.1f}x {r['recall5']:9.2f}")
    print("=" * 84)
    print(f"\nAt {args.threads or 'default'} threads. Paid per QUESTION — and again per")
    print("mid-answer search in agent mode, where MAX_SEARCH_ROUNDS allows two more.")
    print("\nThe score ranges are why KBM_RELEVANCE_FLOOR (0.15, calibrated to the")
    print("incumbent's sigmoid scale) must be re-derived by evaluation/calibrate_floor.py")
    print("before any of these ships: a different scale means abstention breaks silently.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
