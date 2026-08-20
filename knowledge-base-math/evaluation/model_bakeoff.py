"""
model_bakeoff.py - Is another 7B generator better than deepseek-math-7b-rl on OUR questions?

EVALUATION.md §11.6 ended with a finding, not a verdict: deepseek-math's errors are
*execution* errors (right method, wrong long division) and *knowledge* errors, neither of
which sampling can fix. It named two levers — tool-integrated reasoning, and a better
generator. This script is the second lever, measured.

    (run from knowledge-base-math/)
    python evaluation/model_bakeoff.py --check          # are the models pulled? no generation
    python evaluation/model_bakeoff.py --dry-run        # plan + cost, generate nothing
    python evaluation/model_bakeoff.py                  # the run (~2h on a Mac, ~15 min on a pod)
    python evaluation/model_bakeoff.py --report-only    # re-score saved runs, generate nothing
    python evaluation/model_bakeoff.py --easy           # regression tier, AFTER a baseline win

Why this is a separate script and not `--model` used twice
----------------------------------------------------------
`self_consistency.py --model X` already runs any generator. What it cannot do is *rank* two
of them, and eyeballing two of its reports side by side is the wrong statistics: on 30
questions an unpaired accuracy gap needs to be roughly 18 points to clear 95% confidence,
so a real 10-point improvement looks like noise and a fluke looks like a win.

But the two models answer the *same 30 questions*, which makes this a paired design. The
paired test (McNemar) ignores the questions both models got right and both got wrong — they
carry no information about which is better — and asks only about the disagreements. On a
small exam that is dramatically more sensitive, and it is the honest test for this data.
So: generation and scoring come from self_consistency.py (imported, never reimplemented);
this script owns the orchestration and the comparison.

What is held constant
---------------------
Identical questions, identical prompt, identical extraction, identical scoring, identical
sample count, temperature and top_p, and Q4-class quantization for every arm. The only
variable is the model tag. That matters more than it sounds: deepseek-math is a Q4 GGUF, so
comparing it against an fp16 challenger would measure the quantization and call it the model.

The confound this script watches for
------------------------------------
**Parse rate, not accuracy, is where a bake-off goes wrong.** SC_PROMPT asks for
`\\boxed{}`. deepseek-math emits that natively (it is RL-trained on the format) and the Qwen
math models are trained on it too — but a general instruct model may answer correctly in
prose and score zero because nothing was extractable. That is a prompting failure being
reported as a reasoning failure. The report prints per-model unparseable rates first and
refuses to declare a winner when they differ materially, because at that point the exam is
measuring format compliance.

Published benchmarks are context, not evidence
----------------------------------------------
REFERENCE_BENCHMARKS below carries GSM8K/MATH numbers from the Qwen2.5-Math technical report
(arXiv:2409.12122, Table 3), which is the useful kind of reference: every row was scored by
one harness, so the rows are comparable to *each other*. They are not comparable to this
script's numbers — different questions, different prompt, unquantized weights, and a public
test set these models may have trained on. Use them to decide which models are worth the
GPU time, then let the 30 college questions decide what ships.
"""

import argparse
import json
import math
import os
import statistics
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OLLAMA_BASE_URL
from self_consistency import (
    DEFAULT_KS,
    DEFAULT_SAMPLES,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIER,
    DEFAULT_TOP_P,
    RESULTS_DIR,
    TIERS,
    gold_forms,
    model_slug,
    print_report,
    run_tier,
    unload,
)
from retrieval import OLLAMA_MODEL

# ── The arms ───────────────────────────────────────────────────────────────────
# Three questions, one per challenger:
#   1. Does a *newer* math specialist beat the incumbent specialist? (Qwen2.5-Math)
#   2. Does a strong *generalist* beat a math specialist at math? (Qwen2.5 Instruct) —
#      not a silly question: the specialist is a year older and 128K vs 4K context is the
#      difference between a model that can hold retrieved chunks and one that cannot.
#   3. Fallback specialist that lives in the official Ollama library, in case the HF GGUF
#      pull is unavailable on the pod. (Qwen2-Math)
#
# All Q4_K_M, to match the incumbent's Q4 and keep quantization out of the comparison.
# Qwen2.5-Math has no official Ollama library entry; Ollama pulls GGUFs from HF directly.
DEFAULT_MODELS = [
    OLLAMA_MODEL,                                                 # incumbent
    "hf.co/bartowski/Qwen2.5-Math-7B-Instruct-GGUF:Q4_K_M",       # math specialist, newer
    "qwen2.5:7b-instruct-q4_K_M",                                 # strong generalist
]

# Published numbers, for deciding what to test — never for deciding what ships.
# Source: Qwen2.5-Math Technical Report (arXiv:2409.12122) Table 3, few-shot CoT, English.
# Every row scored by the same harness, so the rows are comparable to each other.
REFERENCE_BENCHMARKS = {
    "t1c/deepseek-math-7b-rl:Q4": {
        "name": "DeepSeekMath-7B-RL", "gsm8k": 88.2, "math": 52.4, "ctx": 4096,
        "note": "incumbent; MATH 51.7 in its own paper, 52.4 when re-scored by Qwen's harness",
    },
    "hf.co/bartowski/Qwen2.5-Math-7B-Instruct-GGUF:Q4_K_M": {
        "name": "Qwen2.5-Math-7B-Instruct", "gsm8k": 95.2, "math": 83.6, "ctx": 4096,
        "note": "+31 MATH over the incumbent on paper; 4K context, same as incumbent",
    },
    "qwen2.5:7b-instruct-q4_K_M": {
        "name": "Qwen2.5-7B-Instruct", "gsm8k": 91.6, "math": 75.5, "ctx": 131072,
        "note": "generalist; +23 MATH over the incumbent on paper, and 128K context",
    },
    "qwen2-math:7b-instruct-q4_K_M": {
        "name": "Qwen2-Math-7B-Instruct", "gsm8k": 89.9, "math": 75.1, "ctx": 4096,
        "note": "in the official Ollama library; fallback if the HF GGUF pull fails",
    },
}


# ── Paired significance ────────────────────────────────────────────────────────

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b challenger-only wins and c incumbent-only wins.

    Under "the two models are equally good", each of the b+c questions they disagreed on is
    a fair coin. Concordant questions (both right, both wrong) are dropped: they say nothing
    about which model is better, and keeping them is exactly what makes the unpaired test so
    blunt on 30 questions.
    """
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def paired(challenger: list[dict], incumbent: list[dict], key: str) -> dict:
    """Per-question win/loss/tie for one correctness field, plus the exact p-value."""
    inc = {q["id"]: q for q in incumbent}
    wins, losses, ties = [], [], 0
    for q in challenger:
        base = inc.get(q["id"])
        if base is None:                      # different question sets: not comparable
            continue
        if q[key] and not base[key]:
            wins.append(q["id"])
        elif base[key] and not q[key]:
            losses.append(q["id"])
        else:
            ties += 1
    return {
        "wins": wins, "losses": losses, "ties": ties,
        "p": mcnemar_exact(len(wins), len(losses)),
    }


def beyond_reach(per_q: list[dict]) -> int:
    """Questions no amount of sampling can fix — the ceiling, per self_consistency.py."""
    return len({q["id"] for q in per_q
                if (not q["modal_correct"] and q["modal_share"] >= 0.8)
                or (q["pass_rate"] <= 0.1 and not q["modal_correct"])})


# ── Ollama inventory ───────────────────────────────────────────────────────────

def installed_models() -> set[str]:
    with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=15) as r:
        return {m["name"] for m in json.load(r).get("models", [])}


def check_models(models: list[str]) -> list[str]:
    """Report which arms are pulled. Returns the missing ones.

    Front-loaded on purpose: discovering a missing model 40 minutes into a bake-off costs
    the whole run, and `ollama pull` of a 4.5GB GGUF is not something to hit mid-loop.
    """
    try:
        have = installed_models()
    except Exception as e:
        print(f"[warn] cannot reach Ollama at {OLLAMA_BASE_URL}: {e}")
        return []
    # Ollama reports tags canonically (`qwen2.5:7b-instruct-q4_K_M`, `hf.co/...:Q4_K_M`),
    # and appends `:latest` to a bare name.
    have_norm = {h.removesuffix(":latest") for h in have} | have
    missing = []
    print("\nMODELS")
    for m in models:
        ok = m in have_norm or f"{m}:latest" in have_norm
        ref = REFERENCE_BENCHMARKS.get(m)
        tag = "✓ pulled" if ok else "✗ MISSING"
        extra = (f"  [published: GSM8K {ref['gsm8k']}, MATH {ref['math']}, "
                 f"ctx {ref['ctx']:,}]" if ref else "")
        print(f"  {tag:<10} {m}{extra}")
        if not ok:
            missing.append(m)
    if missing:
        print("\n  Pull them before running (each is ~4.5GB):")
        for m in missing:
            print(f"    ollama pull {m}")
    return missing


def print_reference_table(models: list[str]) -> None:
    print("\n" + "=" * 78)
    print("PUBLISHED BENCHMARKS — context for choosing arms, NOT evidence for choosing a model")
    print("Source: Qwen2.5-Math Technical Report (arXiv:2409.12122) Table 3, few-shot CoT,")
    print("English, unquantized weights. One harness scored every row, so rows are comparable")
    print("to each other — and to nothing in this file's output.")
    print("=" * 78)
    hdr = f"{'model':<28} {'GSM8K':>7} {'MATH':>7} {'context':>9}   note"
    print("\n" + hdr)
    print("─" * len(hdr))
    for m in models:
        ref = REFERENCE_BENCHMARKS.get(m)
        if not ref:
            print(f"{m[:28]:<28} {'—':>7} {'—':>7} {'—':>9}   no published numbers on file")
            continue
        print(f"{ref['name'][:28]:<28} {ref['gsm8k']:>7.1f} {ref['math']:>7.1f} "
              f"{ref['ctx']:>9,}   {ref['note']}")
    print("\nWhy these are not the answer:")
    print("  · GSM8K is grade-school word problems and MATH is competition problems; neither")
    print("    is the multivariable/linear-algebra/ODE distribution the college tier samples.")
    print("  · Both are public test sets these models may have trained on. The 30 hand-written")
    print("    college questions are not on the internet, which is the whole point of them.")
    print("  · Published rows are unquantized. Every arm here is Q4, and §11.6 flags 4-bit as a")
    print("    suspect in exactly the multi-digit-arithmetic failures this is meant to fix.")
    print("  · Table 3 is FEW-SHOT; SC_PROMPT is zero-shot. Exemplars mostly buy format")
    print("    compliance, so expect every arm to land below its published row.")
    print("  · Every row above is CoT, not TIR. Nothing here executes tool calls, so the arms")
    print("    were chosen on the CoT column on purpose. Adopting TIR means re-choosing them.")
    print("  · A +31 MATH gap on paper is a reason to spend the GPU time, not a result.")


# ── Comparison report ──────────────────────────────────────────────────────────

def print_bakeoff(runs: dict[str, dict], incumbent: str, ks: list[int]) -> None:
    n = runs[incumbent]["config"]["n_questions"]
    tier = runs[incumbent]["config"]["tier"]

    print("\n" + "=" * 78)
    print(f"MODEL BAKE-OFF — {tier} tier, {n} questions, closed-book")
    print(f"Incumbent: {incumbent}")
    print("=" * 78)

    # Parse rate first, because a bad parse rate invalidates every number under it.
    print("\nANSWER EXTRACTION (read this before the accuracy table)")
    parse = {}
    for m, r in runs.items():
        cfg = r["config"]
        unp = sum(q["unparseable"] for q in r["per_question"]) / (n * cfg["samples"])
        parse[m] = unp
        print(f"  {unp:>5.0%} unparseable   {m}")
    spread = max(parse.values()) - min(parse.values())
    if spread > 0.05:
        print(f"\n  ⚠️  {spread:.0%} spread in unparseable rate. A model that reasons correctly but")
        print("     does not emit \\boxed{} scores zero on those questions, so the table below is")
        print("     partly measuring format compliance, not math. Fix the prompt for the arm that")
        print("     is failing to parse and re-run before ranking anything.")
    else:
        print(f"\n  Spread {spread:.0%} — extraction is not confounding the comparison.")

    hdr = (f"{'model':<34} {'greedy':>7}" + "".join(f"{'k=' + str(k):>7}" for k in ks)
           + f"{'s/gen':>7}{'beyond':>8}")
    print("\n" + hdr)
    print("─" * len(hdr))
    for m, r in runs.items():
        per_q = r["per_question"]
        s_gen = statistics.mean(q["mean_sample_s"] for q in per_q)
        row = f"{m[-34:]:<34} {r['greedy_accuracy']:>7.2f}"
        row += "".join(f"{r['accuracy_at_k'][str(k)]:>7.2f}" for k in ks)
        row += f"{s_gen:>7.1f}{beyond_reach(per_q):>7}/{n}"
        print(row)
    print("\n'beyond' = questions sampling can never fix (stably wrong, or right answer never")
    print("  produced). A challenger that wins on accuracy but not on 'beyond' is winning at")
    print("  decoding luck; one that shrinks 'beyond' has actually raised the ceiling.")

    # The paired test — the part that decides.
    base = runs[incumbent]["per_question"]
    print("\nPAIRED vs INCUMBENT (McNemar exact, two-sided; concordant questions dropped)")
    for m, r in runs.items():
        if m == incumbent:
            continue
        print(f"\n  {m}")
        for label, key in (("greedy", "greedy_correct"), ("vote (modal)", "modal_correct")):
            d = paired(r["per_question"], base, key)
            w, l = len(d["wins"]), len(d["losses"])
            verdict = ("significant at p<0.05" if d["p"] < 0.05
                       else "NOT significant — inside the noise of this exam")
            print(f"    {label:<13} +{w} / −{l} / ={d['ties']}   p = {d['p']:.3f}   {verdict}")
            if d["wins"]:
                print(f"      fixes:  {', '.join(d['wins'][:8])}")
            if d["losses"]:
                print(f"      breaks: {', '.join(d['losses'][:8])}")

    # Verdict.
    print("\n" + "─" * 78)
    print("VERDICT")
    best = max(runs, key=lambda m: runs[m]["greedy_accuracy"])
    inc_acc = runs[incumbent]["greedy_accuracy"]
    if best == incumbent:
        print(f"  Incumbent still leads on greedy accuracy ({inc_acc:.2f}). Keep it.")
    else:
        d = paired(runs[best]["per_question"], base, "greedy_correct")
        gap = runs[best]["greedy_accuracy"] - inc_acc
        print(f"  Best greedy: {best} at {runs[best]['greedy_accuracy']:.2f} ({gap:+.2f} vs incumbent),")
        print(f"  paired p = {d['p']:.3f} on {len(d['wins']) + len(d['losses'])} disagreements.")
        if d["p"] < 0.05 and spread <= 0.05:
            print("  → SWITCH is supported by this exam. Before shipping, check: context length")
            print("    (the RAG path sends retrieved chunks + history, and a 4K model will truncate),")
            print("    the \\boxed{}/prose output style against api/chat.py's prompt, and the --easy")
            print("    regression tier. A generator that wins here and breaks arithmetic is a loss.")
        elif spread > 0.05:
            print("  → NOT DECIDABLE yet: the extraction spread above means the arms were not")
            print("    scored on equal terms. Fix that first.")
        else:
            print("  → NOT SUPPORTED at p<0.05. The gap is real in the table and indistinguishable")
            print("    from luck on 30 questions. Either accept it as a tie or grow the exam;")
            print("    do not switch the generator on this evidence.")
    print(f"\n  This is {n} questions. The paired test is what makes that survivable, and it still")
    print("  cannot rescue a 2-question difference. Treat a p above 0.05 as 'no result'.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Rank generators on the closed-book reasoning tiers.")
    tier = p.add_mutually_exclusive_group()
    tier.add_argument("--baseline", dest="tier", action="store_const", const="baseline",
                      help="College tier (default) — the tier that decides model choice.")
    tier.add_argument("--easy", dest="tier", action="store_const", const="easy",
                      help="Grade-school regression tier. Run it AFTER a baseline win, as a "
                           "tripwire — never as the ranking itself; it sits at ceiling.")
    p.set_defaults(tier=DEFAULT_TIER)
    p.add_argument("--questions", help="Explicit question file, overriding the tier flags.")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help="Ollama tags to compare. The incumbent must be one of them.")
    p.add_argument("--incumbent", default=OLLAMA_MODEL,
                   help="The arm every challenger is paired against.")
    p.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    p.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--limit", type=int, help="Only the first N questions (smoke run)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Parallel samples per question; >1 needs OLLAMA_NUM_PARALLEL set")
    p.add_argument("--seed", type=int, default=0, help="Seed for the resampling, not the model")
    p.add_argument("--check", action="store_true",
                   help="Print the published-benchmark table and which arms are pulled, then stop.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate questions and print the plan and cost. Generates nothing.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Reuse an arm's saved results JSON instead of regenerating it. This is "
                        "how you add a fourth model without paying for the first three again.")
    p.add_argument("--report-only", action="store_true",
                   help="Compare the saved runs for these models and generate nothing.")
    p.add_argument("--per-model-report", action="store_true",
                   help="Also print self_consistency.py's full single-model report for each arm.")
    args = p.parse_args()

    ks = sorted(set(args.ks))
    if max(ks) > args.samples:
        p.error(f"--ks max ({max(ks)}) exceeds --samples ({args.samples}).")
    if args.incumbent not in args.models:
        p.error(f"--incumbent {args.incumbent} is not in --models; nothing to pair against.")
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be >= 1 (0 would silently run the whole tier).")

    questions_path = args.questions or TIERS[args.tier]
    tier_label = "custom" if args.questions else args.tier

    def out_for(model: str) -> str:
        return os.path.join(RESULTS_DIR,
                            f"self_consistency_{tier_label}_{model_slug(model)}.json")

    if args.check:
        print_reference_table(args.models)
        check_models(args.models)
        return

    with open(questions_path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        items = items[:args.limit]
    if not items:
        p.error(f"No questions in {questions_path}")

    if args.report_only:
        runs = {}
        for m in args.models:
            path = out_for(m)
            if not os.path.exists(path):
                p.error(f"No saved run for {m} at {path}. Run without --report-only first.")
            with open(path, encoding="utf-8") as f:
                runs[m] = json.load(f)
        print_bakeoff(runs, args.incumbent, ks)
        return

    print_reference_table(args.models)
    missing = check_models(args.models)

    gens = len(items) * (args.samples + 1)
    print(f"\nPLAN — {tier_label} tier, {len(items)} questions, {len(args.models)} models")
    print(f"  {gens} generations per model, {gens * len(args.models)} total.")
    print("  At ~8 s/generation (Mac, Metal) that is "
          f"~{gens * len(args.models) * 8 / 3600:.1f} h; a 4090-class pod is roughly 10× faster.")
    print("  Models are unloaded between arms so they do not all sit in VRAM at once.")
    if args.dry_run:
        bad = [it["id"] for it in items if not gold_forms(it)]
        print("\nDry run: questions load and "
              + ("all golds normalize." if not bad else f"BAD GOLDS: {bad}"))
        for m in args.models:
            print(f"  would write {out_for(m)}")
        return
    if missing:
        p.error("Pull the missing models above first — a bake-off that drops an arm mid-run "
                "wastes the arms that already ran.")

    runs = {}
    for i, model in enumerate(args.models, 1):
        path = out_for(model)
        if args.skip_existing and os.path.exists(path):
            print(f"\n[{i}/{len(args.models)}] {model} — reusing {path}")
            with open(path, encoding="utf-8") as f:
                runs[model] = json.load(f)
            continue
        print(f"\n[{i}/{len(args.models)}] {model} — {gens} generations")
        runs[model] = run_tier(items, model, args.samples, ks, args.temperature, args.top_p,
                               args.concurrency, args.seed, tier_label, questions_path, path)
        print(f"  wrote {path} ({runs[model]['config']['wall_clock_s'] / 60:.1f} min)")
        unload(model)

    if args.per_model_report:
        for model, r in runs.items():
            print_report(r["per_question"],
                         {int(k): v for k, v in r["accuracy_at_k"].items()}, ks, r["config"])

    print_bakeoff(runs, args.incumbent, ks)

    summary = os.path.join(RESULTS_DIR, f"model_bakeoff_{tier_label}.json")
    with open(summary, "w", encoding="utf-8") as f:
        json.dump({
            "tier": tier_label,
            "questions": questions_path,
            "n_questions": len(items),
            "incumbent": args.incumbent,
            "models": {
                m: {
                    "greedy_accuracy": r["greedy_accuracy"],
                    "accuracy_at_k": r["accuracy_at_k"],
                    "beyond_reach": beyond_reach(r["per_question"]),
                    "unparseable_rate": sum(q["unparseable"] for q in r["per_question"])
                                        / (len(items) * r["config"]["samples"]),
                    "wall_clock_s": r["config"]["wall_clock_s"],
                    "results_file": out_for(m),
                    "reference": REFERENCE_BENCHMARKS.get(m),
                } for m, r in runs.items()
            },
            "paired_vs_incumbent": {
                m: {
                    "greedy": paired(r["per_question"],
                                     runs[args.incumbent]["per_question"], "greedy_correct"),
                    "vote": paired(r["per_question"],
                                   runs[args.incumbent]["per_question"], "modal_correct"),
                } for m, r in runs.items() if m != args.incumbent
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {summary}")


if __name__ == "__main__":
    main()
