"""
eval_followup.py - Does a follow-up turn still find the right chunk?

evaluation/eval.py measures retrieval on questions that stand on their own. Nobody talks
that way for more than one turn. "Explain that again" retrieves on a sentence made of
stopwords: BM25 matches noise, the dense embedder encodes nothing, every chunk falls
under KBM_RELEVANCE_FLOOR, and the answer silently loses its grounding — same document,
same corpus, no `Sources:` line. `goldset.jsonl` is single-turn and cannot see any of it.

    (run from knowledge-base-math/, so the indexes resolve)
    python evaluation/eval_followup.py --user calctest

Three arms, one gold set:

    turn1       retrieve on the standalone question   — the ceiling, eval.py's number
    raw         retrieve on the follow-up alone       — what ships with the flag off
    rewritten   retrieve on chat.retrieval_query(...) — what ships with it on

Then the number that actually decides it: the FALSE-POSITIVE COST. The rewrite is gated
by a heuristic (chat.is_follow_up), so it will fire on some standalone questions too.
That arm re-runs the ordinary gold set with a prior question from an UNRELATED row
attached, and asks whether recall@5 survives. A follow-up fix that quietly degrades
ordinary questions is a net loss, and nothing else in the harness would notice.

Everything measured here is imported — the matcher, the metrics, the pipeline, the
rewrite itself. eval.py's own warning applies twice over: a harness that measures a
reimplementation drifts from what ships and starts lying.

Protocol and how to read the output: EVALUATION.md
"""

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kbm.retrieval import (
    EMBED_MODEL,
    load_bm25,
    load_chroma,
    load_embeddings,
    load_reranker,
    retrieve_detailed,
)

# The pipeline's own matcher and metric, not a second copy of either.
from eval import (  # noqa: E402 - after the sys.path insert above
    CONFIGS,
    OVERLAP_STRICT,
    RESULTS_DIR,
    load_goldset,
    ndcg_at_k,
    rank_of_overlap,
)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
FOLLOWUP_PATH = os.path.join(EVAL_DIR, "followup_set.jsonl")
GOLDSET_PATH = os.path.join(EVAL_DIR, "goldset.jsonl")


def load_pairs(followup_path: str, goldset_path: str) -> list[dict]:
    """Join the follow-up phrasings onto the gold rows they were written against.

    followup_set.jsonl deliberately holds no `chunk_text`. The gold passage already
    exists in goldset.jsonl, hand-cleaned, and a second copy here would be a second
    thing to keep in step — the CHROMA_DIR mistake in miniature. The join is on
    gold_chunk_id, and a follow-up whose gold row has been deleted is a hard error
    rather than a silently shorter exam.
    """
    gold = {g["gold_chunk_id"]: g for g in load_goldset(goldset_path)}
    if not os.path.exists(followup_path):
        raise FileNotFoundError(f"No follow-up set at {followup_path}.")
    pairs = []
    with open(followup_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            g = gold.get(row["gold_chunk_id"])
            if g is None:
                raise KeyError(
                    f"{row['gold_chunk_id']} is in {os.path.basename(followup_path)} "
                    f"but not in {os.path.basename(goldset_path)}. The two sets are "
                    f"joined on that id — regenerate the follow-up set."
                )
            pairs.append({**row, "chunk_text": g["chunk_text"], "source": g["source"]})
    return pairs


def _history(prior_question: str):
    """The conversation a follow-up arrives in: one prior turn, student then tutor.

    The tutor's text is a placeholder on purpose. retrieval_query reads only the last
    STUDENT message, and putting a plausible answer here would invite the reader to
    believe the answer's wording was part of what was measured.
    """
    from api.schemas import Message, Role

    return [
        Message(role=Role.USER, content=prior_question),
        Message(role=Role.ASSISTANT, content="(the tutor's answer)"),
    ]


def score(queries: list[str], pairs: list[dict], cfg: dict, **kw) -> dict:
    """recall/MRR/nDCG for one query per gold row, in eval.py's own terms."""
    ranks: list[int | None] = []
    pooled: list[bool] = []
    for query, item in zip(queries, pairs):
        r = retrieve_detailed(query, **kw, **cfg)
        ranks.append(rank_of_overlap(r.ranked, item["chunk_text"], OVERLAP_STRICT))
        pooled.append(
            rank_of_overlap(r.candidates, item["chunk_text"], OVERLAP_STRICT) is not None
        )
    n = len(ranks)
    found = [r for r in ranks if r is not None]
    return {
        "n": n,
        "recall@1": sum(1 for r in ranks if r == 1) / n,
        "recall@5": sum(1 for r in ranks if r is not None and r <= 5) / n,
        "recall@pool": sum(pooled) / n,
        "mrr": sum(1.0 / r for r in found) / n,
        "ndcg@5": sum(ndcg_at_k(r, 5) for r in ranks) / n,
    }


def print_table(rows: list[tuple[str, dict]]) -> None:
    print(f"\n{'arm':<24} {'n':>3} {'R@1':>6} {'R@5':>6} {'R@pool':>7} {'MRR':>6} {'nDCG@5':>7}")
    print("─" * 62)
    for name, m in rows:
        print(f"{name:<24} {m['n']:>3} {m['recall@1']:>6.2f} {m['recall@5']:>6.2f} "
              f"{m['recall@pool']:>7.2f} {m['mrr']:>6.3f} {m['ndcg@5']:>7.3f}")


def main():
    p = argparse.ArgumentParser(description="Measure retrieval on follow-up turns.")
    p.add_argument("--user", required=True, help="Username whose index to evaluate against")
    p.add_argument("--config", default="hybrid+rerank", choices=list(CONFIGS),
                   help="Retrieval config to hold constant across the arms")
    p.add_argument("--followup-set", default=FOLLOWUP_PATH)
    p.add_argument("--goldset", default=GOLDSET_PATH)
    p.add_argument("--embed-model", default=EMBED_MODEL,
                   help="Must match how the target index was built")
    p.add_argument("--normalize-latex", action="store_true",
                   help="Must match how the target index was built")
    args = p.parse_args()

    from api import chat as chatmod

    pairs = load_pairs(args.followup_set, args.goldset)
    cfg = CONFIGS[args.config]

    print(f"\n{'═' * 62}")
    print(f"FOLLOW-UP RETRIEVAL — {len(pairs)} turns, config={args.config}")
    sources = sorted({item["source"] for item in pairs})
    print(f"  source documents: {', '.join(sources)}")
    if len(sources) == 1:
        # Worth saying out loud. With one document in the corpus every arm retrieves
        # from the same book, so this measures WITHIN-document ranking only. The failure
        # a follow-up causes in production — retrieving from the wrong document, or from
        # nothing at all — is bounded away by the corpus, and the numbers below are
        # therefore a floor on the win, not the whole of it.
        print("  ⚠ single-document corpus: this measures ranking within one book, not")
        print("    document selection. Treat the deltas as a lower bound.")
    print(f"{'═' * 62}")

    embeddings = load_embeddings(args.embed_model, normalize_latex=args.normalize_latex)
    reranker = load_reranker()
    kw = {
        "user": args.user,
        "embeddings": embeddings,
        "reranker": reranker,
        "bm25_index": load_bm25(args.user),
        "store": load_chroma(args.user, embeddings),
    }

    q1 = [item["q1"] for item in pairs]
    q2 = [item["q2"] for item in pairs]
    rewritten, fired = [], 0
    for item in pairs:
        q, did = chatmod.retrieval_query(item["q2"], _history(item["q1"]))
        rewritten.append(q)
        fired += did

    rows = [
        ("turn1 (standalone)", score(q1, pairs, cfg, **kw)),
        ("follow-up, raw", score(q2, pairs, cfg, **kw)),
        ("follow-up, rewritten", score(rewritten, pairs, cfg, **kw)),
    ]
    print_table(rows)
    print(f"\nthe rewrite fired on {fired}/{len(pairs)} follow-ups "
          f"({fired / len(pairs):.0%}) — anything below 100% is a gap in "
          f"chat.is_follow_up, not in retrieval")

    # ── False-positive cost ───────────────────────────────────────────────────
    # The ship/no-ship gate. Every question here stands on its own and needs no context;
    # the rewrite fires on some of them anyway (a short question is indistinguishable
    # from an elliptical one without reading it), and when it does it staples an
    # unrelated topic onto the query. If recall@5 falls here, the fix costs more than it
    # earns. The prior question is taken from the NEXT row, so every one is off-topic by
    # construction and the pairing is deterministic rather than sampled.
    gold_rows = load_goldset(args.goldset)
    gq = [g["q"] for g in gold_rows]
    contaminated, fp_fired = [], 0
    for i, g in enumerate(gold_rows):
        q, did = chatmod.retrieval_query(g["q"], _history(gq[(i + 1) % len(gq)]))
        contaminated.append(q)
        fp_fired += did

    fp_rows = [
        ("goldset, clean", score(gq, gold_rows, cfg, **kw)),
        ("goldset, unrelated prior", score(contaminated, gold_rows, cfg, **kw)),
    ]
    print(f"\n{'═' * 62}")
    print("FALSE-POSITIVE COST — standalone questions, unrelated prior turn attached")
    print(f"{'═' * 62}")
    print_table(fp_rows)
    delta = fp_rows[1][1]["recall@5"] - fp_rows[0][1]["recall@5"]
    delta1 = fp_rows[1][1]["recall@1"] - fp_rows[0][1]["recall@1"]
    print(f"\nthe rewrite fired on {fp_fired}/{len(gold_rows)} standalone questions "
          f"({fp_fired / len(gold_rows):.0%}) — that is the exposure")
    print(f"recall@5 delta: {delta:+.2f}    recall@1 delta: {delta1:+.2f}")
    # recall@5 is the gate and recall@1 is not, which is a claim about the SERVING path
    # rather than a convenient choice of metric: retrieval hands the prompt TOP_N=5
    # chunks and select_context filters them per chunk against the floor, so a gold
    # chunk demoted from rank 1 to rank 3 is still in front of the model. A gold chunk
    # pushed past rank 5 is gone. Report both; ship on the second.
    print("  (recall@5 is the gate: all TOP_N=5 chunks reach the prompt, so a demotion")
    print("   inside the top 5 costs the answer nothing. recall@1 is the honest cost.)")
    if delta < -0.05:
        print("⚠ THE HEURISTIC IS TOO EAGER. Tighten chat.FOLLOWUP_MAX_WORDS or the")
        print("  anaphor list before shipping — this arm is the gate, not the one above.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "followup_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": args.config,
            "user": args.user,
            "embed_model": args.embed_model,
            "normalize_latex": args.normalize_latex,
            "followup": {name: m for name, m in rows},
            "followup_fired": fired,
            "false_positive": {name: m for name, m in fp_rows},
            "false_positive_fired": fp_fired,
            "recall5_delta": round(delta, 4),
            "recall1_delta": round(delta1, 4),
            "queries": [
                {"q1": i["q1"], "q2": i["q2"], "rewritten": r}
                for i, r in zip(pairs, rewritten)
            ],
        }, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
