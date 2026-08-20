"""
self_consistency.py - Is majority voting over k sampled answers worth k times the cost?

deepseek-math-7b is a solver, and solvers get arithmetic wrong in ways that are not
systematic: re-sample the same question at a non-zero temperature and the wrong answers
scatter while the right one repeats. Self-consistency (Wang et al., 2022) exploits that —
sample k chains of thought, take the majority final answer — and it is the cheapest
accuracy fix available to us because it needs no new model, no fine-tune, and no data.

It is also a k-times latency and GPU-cost multiplier on a pipeline where generation is
already ~95% of query time (LATENCY.md). So the question is not "does it help" — it is
"does it help ENOUGH here to be worth 5-10x the decode". This script answers that with
numbers instead of vibes.

    (run from knowledge-base-math/)
    python evaluation/self_consistency.py                    # baseline (college) tier, 10 samples
    python evaluation/self_consistency.py --easy             # grade-school regression tier
    python evaluation/self_consistency.py --limit 5          # quick smoke run
    python evaluation/self_consistency.py --concurrency 4    # needs OLLAMA_NUM_PARALLEL>1
    python evaluation/self_consistency.py --model qwen2.5:7b-instruct-q4_K_M   # another generator
    python evaluation/self_consistency.py --report-only \
        evaluation/results/self_consistency_baseline_deepseek-math-7b-rl_Q4.json

To rank generators rather than measure one, use evaluation/model_bakeoff.py, which drives this
script's own functions over several models and adds the paired significance test that a
30-question exam needs. Do not eyeball two separate runs of this script side by side: on n=30
an unpaired 10-point gap is inside the noise, and the questions are the same both times, so the
paired test is both the honest and the more sensitive one.

What it measures
----------------
CLOSED-BOOK, on purpose. The questions are hand-written with verifiable short answers and no
document context, because the thing under test is the model's *reasoning stability*, not
retrieval. Mixing retrieval in would mean a wrong answer could be the retriever's fault, and
the whole point is to isolate the generator. Retrieval quality is eval.py's job.

Two tiers, and the distinction decides what a number means:
  --baseline (default)  evaluation/reasoning_set_college.jsonl — college-level. THE set that
                        decides model choice and whether voting pays, because a benchmark
                        only discriminates near the incumbent's ~50% mark.
  --easy                evaluation/reasoning_set_easy.jsonl — grade-school. A REGRESSION
                        tier: it catches a change that wins on hard questions while breaking
                        basic arithmetic. It sits near ceiling by design, so a high score
                        there is not evidence of quality.

Both answer keys are recomputed independently by evaluation/verify_reasoning_set.py — a
wrong key marks a right model wrong and corrupts everything downstream.

How k=1/5/10 are compared fairly
--------------------------------
Naively you would generate 1, then 5, then 10 answers — 16 generations per question, and a
k=1 number estimated from a single sample (huge variance on 20 questions). Instead this
draws N samples ONCE per question and estimates majority-vote accuracy at each k by
resampling k of those N without replacement, many times. Same generations, far tighter
estimates, and every k is scored against the identical pool of model outputs — so a
difference between k=1 and k=10 is the voting, not sampling luck.

A greedy (temperature=0) run is included separately as the honest baseline, because that is
what ships today: today's system is greedy k=1, not sampled k=1.

Verdict, not a table dump: the script ends by stating whether the accuracy gained per extra
generation clears a bar worth paying for, and where the remaining errors live. Voting can
only reorder the sample pool, so an answer the model never produces — or produces and
agrees is something else — is outside its reach no matter how large k gets.
"""

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# This lives in evaluation/; the pipeline modules (retrieval, config, …) are one level up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import OLLAMA_BASE_URL
from retrieval import OLLAMA_MODEL

EVAL_DIR = "evaluation"
RESULTS_DIR = os.path.join(EVAL_DIR, "results")


def model_slug(model: str) -> str:
    """Filesystem-safe short name for an Ollama model tag.

    Result files are named per (tier, model). A run is only meaningful against the run it
    is compared to, so two models writing the same filename would mean the challenger
    silently destroying the incumbent baseline — the same failure the per-tier naming
    already guards against, one axis up. `hf.co/bartowski/Qwen2.5-Math-7B-Instruct-GGUF:Q4_K_M`
    becomes `Qwen2.5-Math-7B-Instruct-GGUF_Q4_K_M`.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.rsplit("/", 1)[-1])

# Two tiers, because they answer different questions and one cannot replace the other.
#   baseline - the real target distribution: college-level (multivariable, linear algebra,
#              ODEs, analysis, probability). This is the set that DECIDES things — model
#              choice, whether self-consistency pays — because a benchmark only
#              discriminates near the incumbent's ~50% mark, not at its ceiling.
#   easy     - the original grade-school/early-undergrad set. Kept as a REGRESSION tier:
#              a change that lifts the hard set while breaking basic arithmetic is a bad
#              change, and only the easy tier can see that. Near-ceiling by design, so it
#              is useless for ranking models — do not read it as a quality score.
TIERS = {
    "baseline": os.path.join(EVAL_DIR, "reasoning_set_college.jsonl"),
    "easy": os.path.join(EVAL_DIR, "reasoning_set_easy.jsonl"),
}
DEFAULT_TIER = "baseline"

DEFAULT_SAMPLES = 10
DEFAULT_KS = [1, 5, 10]
DEFAULT_TEMPERATURE = 0.8   # SC needs diversity; at temperature 0 every sample is identical
DEFAULT_TOP_P = 0.95
NUM_PREDICT = 512           # room to reason, but these are short problems
KEEP_ALIVE = "30m"          # keep the model resident across ~200 generations
RESAMPLE_TRIALS = 400       # bootstrap draws per (question, k)

# The answer must be machine-extractable or nothing downstream is measurable. deepseek-math
# emits \boxed{} natively (it is RL-trained on exactly that format), so we ask for it and
# fall back through progressively looser patterns rather than dropping the sample.
SC_PROMPT = """Solve the problem. Reason step by step, then give the final answer on its own last line in the form:

Final answer: \\boxed{{<answer>}}

Give a single number or a single expression inside the box — no units, no words, no explanation inside the box.

Problem: {question}"""


# ── Answer extraction ──────────────────────────────────────────────────────────
# Voting is only as good as the extraction: if two samples say "5/16" and "0.3125" and the
# normalizer treats them as different strings, the vote splits and self-consistency looks
# worse than it is. So parse to a NUMBER whenever the answer is numeric, and only fall back
# to string comparison for genuinely symbolic answers.

_BOXED = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_FINAL_LINE = re.compile(r"final answer\s*[:=]?\s*(.+)", re.IGNORECASE)
_FRAC = re.compile(r"\\d?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*(?:\s*/\s*-?\d+\.?\d*)?")


def extract_answer(text: str) -> str | None:
    """Pull the final answer out of one generated solution, or None if there is none.

    Order matters: \\boxed{} first (what we asked for), then a "Final answer:" line, then
    the last number anywhere in the text. The last fallback is deliberately permissive —
    an unparseable sample is a wasted generation, and on a 7B model there are enough of
    those already without the parser adding its own.
    """
    if not text:
        return None

    boxed = _BOXED.findall(text)
    if boxed:
        return normalize_answer(boxed[-1])

    for line in reversed(text.strip().splitlines()):
        m = _FINAL_LINE.search(line)
        if m and m.group(1).strip():
            return normalize_answer(m.group(1))

    # Last resort: the final number the model wrote. Wrong often enough to matter, which is
    # why unparseable/fallback rates are reported alongside accuracy.
    tail = text.strip().splitlines()[-1] if text.strip() else ""
    nums = _NUMBER.findall(tail) or _NUMBER.findall(text)
    if nums:
        return normalize_answer(nums[-1])
    return None


def normalize_answer(raw: str) -> str | None:
    """Canonical form of an answer string: a number rendered to 4dp, or a cleaned string.

    Numeric canonicalization is what makes "5/16", "0.3125" and "$0.3125$" one vote instead
    of three. 4dp is chosen to match how the gold aliases are written (0.1667 for 1/6) —
    tighter and repeating decimals stop agreeing with themselves.
    """
    if raw is None:
        return None
    s = raw.strip()
    s = s.replace("$", "").replace("\\!", "").replace("\\,", "").replace("\\ ", " ")
    s = s.replace("\\%", "%").replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = _FRAC.sub(r"(\1)/(\2)", s)          # \frac{a}{b} and \dfrac{a}{b} -> (a)/(b)
    s = re.sub(r"\\frac(\d)(\d)", r"(\1)/(\2)", s)   # \frac12, the brace-less form
    # A stray backslash before a plain number ("\0.05" — a mangled \$ escape) must not make
    # a correct answer count as wrong. Seen in the 2026-08-19 run on r20.
    s = re.sub(r"\\+(?=[\d.])", "", s)
    s = s.strip().strip(".").strip()
    s = re.sub(r"\s+", " ", s)
    if not s:
        return None

    val = to_number(s)
    if val is not None:
        # -0.0 and 0.0 must not be different votes.
        return f"{val + 0.0:.4f}"
    return s.lower()


_CONSTS = {"pi": math.pi, "π": math.pi, "e": math.e}


def to_number(s: str) -> float | None:
    """Best-effort numeric value of an answer string; None if it is not a number.

    Handles plain numbers, thousands separators, percentages, a/b fractions, and short
    products/quotients of numbers and the constants π and e — because college answers come
    back as "\\pi/2" or "2\\pi" at least as often as "1.5708", and scoring those wrong would
    make the harness measure its own parser instead of the model (the r20 lesson).

    Deliberately does NOT eval() — the factors are parsed and multiplied by hand. eval() on
    model output is a code-execution hole, and no benchmark is worth one.
    """
    t = s.replace(",", "").replace(" ", "").rstrip("%")
    is_pct = s.strip().endswith("%")
    try:
        val = float(t)
        return val / 100 if is_pct else val
    except ValueError:
        pass

    m = re.fullmatch(r"\(?(-?\d+\.?\d*)\)?/\(?(-?\d+\.?\d*)\)?", t)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den == 0:
            return None
        val = num / den
        return val / 100 if is_pct else val

    val = _product_of_factors(t)
    if val is not None:
        return val / 100 if is_pct else val
    return None


def _product_of_factors(t: str) -> float | None:
    """Value of a chain like "2*pi/3", "\\pi/2" or "2pi", or None if it is not one.

    Only numbers and the constants π/e are accepted as factors; anything else (a variable,
    a function call, a root) returns None and the answer stays a string, which is the safe
    outcome — string voting still works, it just will not unify with a decimal.
    """
    t = t.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    t = re.sub(r"(?<=[\d)])\s*(?=(?:pi|π))", "*", t)      # implicit "2pi" -> "2*pi"
    t = t.replace("(", "").replace(")", "")
    if not re.fullmatch(r"-?[\d.]*(?:pi|π|e)?(?:[*/][\d.]*(?:pi|π|e)?)*", t) or not t:
        return None

    tokens = re.split(r"([*/])", t)
    if not tokens or tokens[0] == "":
        return None
    try:
        value = _factor(tokens[0])
        for op, raw in zip(tokens[1::2], tokens[2::2]):
            f = _factor(raw)
            if f is None or value is None:
                return None
            if op == "*":
                value *= f
            else:
                if f == 0:
                    return None
                value /= f
        return value
    except (ValueError, TypeError):
        return None


def _factor(raw: str) -> float | None:
    """One factor: a number, a constant, or a number glued to a constant ("-2pi")."""
    if raw in _CONSTS:
        return _CONSTS[raw]
    m = re.fullmatch(r"(-?[\d.]*)(pi|π|e)?", raw)
    if not m:
        return None
    coeff, const = m.group(1), m.group(2)
    if coeff in ("", "-", "+"):
        if const is None:
            return None
        base = 1.0 if coeff != "-" else -1.0
    else:
        base = float(coeff)
    return base * _CONSTS[const] if const else base


def gold_forms(item: dict) -> set[str]:
    """Every normalized string that counts as correct for one question."""
    forms = {normalize_answer(item["answer"])}
    for alias in item.get("aliases", []):
        forms.add(normalize_answer(alias))
    return {f for f in forms if f is not None}


def is_correct(candidate: str | None, gold: set[str]) -> bool:
    if candidate is None:
        return False
    if candidate in gold:
        return True
    # Numeric near-match: 0.1667 vs 0.1666… should not be a miss.
    cval = to_number(candidate)
    if cval is None:
        return False
    for g in gold:
        gval = to_number(g)
        if gval is not None and abs(cval - gval) <= max(1e-3, abs(gval) * 1e-3):
            return True
    return False


# ── Voting ─────────────────────────────────────────────────────────────────────

def majority_vote(answers: list[str | None]) -> str | None:
    """The winning answer of one vote, or None if nothing was parseable.

    Unparseable samples do not vote (a model that failed to state an answer has not cast
    one). Ties break toward the answer that appeared first, which is what a real
    implementation would do with a stream of samples.
    """
    votes = [a for a in answers if a is not None]
    if not votes:
        return None
    counts = Counter(votes)
    best = max(counts.values())
    for a in votes:                       # first-appearance tie-break
        if counts[a] == best:
            return a
    return None


def vote_accuracy_at_k(answers: list[str | None], gold: set[str], k: int,
                       rng: random.Random, trials: int) -> float:
    """P(majority vote over k of these samples is correct), by resampling without replacement.

    k >= len(answers) is exact (one possible draw), so no resampling is done there.
    """
    n = len(answers)
    if k >= n:
        return 1.0 if is_correct(majority_vote(answers), gold) else 0.0
    hits = 0
    for _ in range(trials):
        subset = rng.sample(answers, k)
        if is_correct(majority_vote(subset), gold):
            hits += 1
    return hits / trials


# ── Generation ─────────────────────────────────────────────────────────────────

def build_chain(temperature: float, top_p: float, seed: int | None = None,
                model: str = OLLAMA_MODEL):
    llm = ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        top_p=top_p,
        num_predict=NUM_PREDICT,
        keep_alive=KEEP_ALIVE,
        **({"seed": seed} if seed is not None else {}),
    )
    return ChatPromptTemplate.from_template(SC_PROMPT) | llm | StrOutputParser()


def generate(chain, question: str) -> tuple[str, float]:
    """One generation, returning (text, seconds). A failed call is an empty sample, not a crash."""
    t0 = time.perf_counter()
    try:
        text = chain.invoke({"question": question})
    except Exception as e:                       # a dropped Ollama call must not lose the run
        text = ""
        print(f"    [warn] generation failed: {e}")
    return text, time.perf_counter() - t0


def run_question(item: dict, greedy_chain, sample_chain, n_samples: int,
                 concurrency: int) -> dict:
    gold = gold_forms(item)

    greedy_text, greedy_s = generate(greedy_chain, item["question"])
    greedy_ans = extract_answer(greedy_text)

    def one(_):
        return generate(sample_chain, item["question"])

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            outputs = list(pool.map(one, range(n_samples)))
    else:
        outputs = [one(i) for i in range(n_samples)]

    texts = [t for t, _ in outputs]
    times = [s for _, s in outputs]
    answers = [extract_answer(t) for t in texts]

    counts = Counter(a for a in answers if a is not None)
    top_answer, top_count = (counts.most_common(1)[0] if counts else (None, 0))

    return {
        "id": item["id"],
        "topic": item.get("topic", ""),
        "question": item["question"],
        "gold": sorted(gold),
        "greedy_answer": greedy_ans,
        "greedy_correct": is_correct(greedy_ans, gold),
        "greedy_s": round(greedy_s, 2),
        "sampled_answers": answers,
        "sample_correct": [is_correct(a, gold) for a in answers],
        "pass_rate": sum(is_correct(a, gold) for a in answers) / len(answers) if answers else 0.0,
        "unparseable": sum(a is None for a in answers),
        "modal_answer": top_answer,
        "modal_share": (top_count / len(answers)) if answers else 0.0,
        "modal_correct": is_correct(top_answer, gold),
        "distinct_answers": len(counts),
        "mean_sample_s": round(statistics.mean(times), 2) if times else 0.0,
    }


# ── One run ────────────────────────────────────────────────────────────────────

def run_tier(items: list[dict], model: str, samples: int, ks: list[int],
             temperature: float, top_p: float, concurrency: int, seed: int,
             tier_label: str, questions_path: str, out_path: str) -> dict:
    """Generate and score one (model, tier) run, write its JSON, and return the record.

    Both the single-model CLI and evaluation/model_bakeoff.py call this. A bake-off that
    reimplemented the loop would drift from the thing it claims to be comparing — the same
    trap CLAUDE.md records for retrieval and the index paths.
    """
    greedy_chain = build_chain(0.0, 1.0, model=model)
    sample_chain = build_chain(temperature, top_p, model=model)

    t0 = time.perf_counter()
    per_q = []
    for i, item in enumerate(items, 1):
        print(f"  [{i}/{len(items)}] {item['id']} ({item.get('topic', '')})…", flush=True)
        rec = run_question(item, greedy_chain, sample_chain, samples, concurrency)
        per_q.append(rec)
        print(f"      greedy {'✓' if rec['greedy_correct'] else '✗'} ({rec['greedy_answer']})  "
              f"sampled pass-rate {rec['pass_rate']:.0%}  "
              f"modal {rec['modal_answer']} ×{rec['modal_share']:.0%} "
              f"{'✓' if rec['modal_correct'] else '✗'}")
    wall = time.perf_counter() - t0

    rng = random.Random(seed)
    acc_at_k = {
        k: statistics.mean(
            vote_accuracy_at_k(q["sampled_answers"], set(q["gold"]), k, rng, RESAMPLE_TRIALS)
            for q in per_q
        )
        for k in ks
    }

    cfg = {
        "model": model,
        "samples": samples,
        "ks": ks,
        "temperature": temperature,
        "top_p": top_p,
        "trials": RESAMPLE_TRIALS,
        "num_predict": NUM_PREDICT,
        "questions": questions_path,
        "tier": tier_label,
        "n_questions": len(items),
        "wall_clock_s": round(wall, 1),
    }
    record = {
        "config": cfg,
        "greedy_accuracy": sum(q["greedy_correct"] for q in per_q) / len(per_q),
        "accuracy_at_k": {str(k): acc_at_k[k] for k in ks},
        "per_question": per_q,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return record


def unload(model: str) -> None:
    """Ask Ollama to drop a model's weights now (keep_alive=0).

    KEEP_ALIVE is 30m so one run does not pay a reload per generation — but a bake-off runs
    several 4-5GB models back to back, and without this all of them stay resident at once.
    Best-effort: failing to free VRAM must not lose a run that already cost 40 minutes.
    """
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/generate",
            data=json.dumps({"model": model, "keep_alive": 0}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=30).read()
    except Exception as e:
        print(f"    [warn] could not unload {model}: {e}")


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_report(per_q: list[dict], acc_at_k: dict[int, float], ks: list[int],
                 cfg: dict) -> None:
    n = len(per_q)
    greedy_acc = sum(q["greedy_correct"] for q in per_q) / n
    mean_sample_s = statistics.mean(q["mean_sample_s"] for q in per_q)

    print("\n" + "=" * 74)
    print(f"SELF-CONSISTENCY — {cfg['model']}, closed-book, {n} questions")
    print(f"  {cfg['samples']} samples/question at temperature={cfg['temperature']}, "
          f"top_p={cfg['top_p']}; k estimated by {cfg['trials']} resamples")
    print("=" * 74)

    header = f"{'setting':<22} {'accuracy':>9} {'vs greedy':>10} {'gens/q':>7} {'est. s/query':>13}"
    print("\n" + header)
    print("─" * len(header))
    mean_greedy_s = statistics.mean(q["greedy_s"] for q in per_q)
    print(f"{'greedy (ships today)':<22} {greedy_acc:>9.2f} {'—':>10} {1:>7} "
          f"{mean_greedy_s:>13.1f}")
    for k in ks:
        acc = acc_at_k[k]
        print(f"{'majority vote k=' + str(k):<22} {acc:>9.2f} {acc - greedy_acc:>+10.2f} "
              f"{k:>7} {mean_sample_s * k:>13.1f}")

    print("\nk=1 is a SAMPLED single answer, not greedy — it is the honest control for the")
    print("  voting itself (same temperature, no vote). Greedy is what the system does now.")
    print("est. s/query assumes samples run serially, which is the pod default; with")
    print(f"  OLLAMA_NUM_PARALLEL>1 the wall clock falls but the GPU cost does not.")

    # Where the errors live. This is the part that decides the answer, not the table.
    unanimous_wrong = [q for q in per_q if not q["modal_correct"] and q["modal_share"] >= 0.8]
    recoverable = [q for q in per_q if not q["greedy_correct"] and q["modal_correct"]]
    broken = [q for q in per_q if q["greedy_correct"] and not q["modal_correct"]]
    # The other way voting cannot help: the samples scatter across many WRONG answers and the
    # right one is essentially absent from the pool. Voting reorders a pool; it cannot add to
    # it. These look nothing like "confidently wrong" (low modal share, high answer diversity)
    # but are just as far out of reach, so counting only unanimity understates the ceiling.
    unreachable = [q for q in per_q if q["pass_rate"] <= 0.1 and not q["modal_correct"]]
    beyond = {q["id"]: q for q in unanimous_wrong + unreachable}
    mean_distinct = statistics.mean(q["distinct_answers"] for q in per_q)
    unparse = sum(q["unparseable"] for q in per_q) / (n * cfg["samples"])

    print(f"\nERROR STRUCTURE")
    print(f"  fixed by voting (greedy wrong → vote right):   {len(recoverable):>3}/{n}")
    print(f"  broken by voting (greedy right → vote wrong):  {len(broken):>3}/{n}")
    print(f"  confidently wrong (≥80% of samples agree on a wrong answer): {len(unanimous_wrong):>3}/{n}")
    print(f"  never right (≤10% of samples correct — right answer not in the pool): "
          f"{len(unreachable):>3}/{n}")
    print(f"  mean distinct answers per question: {mean_distinct:.1f}   unparseable samples: {unparse:.0%}")
    if beyond:
        print(f"\n  BEYOND VOTING'S REACH: {len(beyond)}/{n}. More sampling cannot fix these — the")
        print("  model either agrees on the wrong answer or never produces the right one. Only a")
        print("  better prompt, a better model, or retrieval can move them. This count, not the")
        print("  accuracy delta, is the ceiling on what self-consistency can ever deliver here.")
        for q in list(beyond.values())[:5]:
            kind = "stably wrong" if q["modal_share"] >= 0.8 else "scattered"
            print(f"      {q['id']} ({q['topic']}, {kind}): said {q['modal_answer']} "
                  f"({q['modal_share']:.0%} of samples), gold {q['gold'][0]}, "
                  f"{q['pass_rate']:.0%} of samples correct")

    # Verdict.
    best_k = max(ks, key=lambda k: acc_at_k[k])
    gain = acc_at_k[best_k] - greedy_acc
    print(f"\nVERDICT — is self-consistency worth implementing here?")
    print(f"  best k = {best_k}: accuracy {greedy_acc:.2f} → {acc_at_k[best_k]:.2f} ({gain:+.2f}) "
          f"for {best_k}x the generations")
    if gain <= 0.0:
        print("  → NO. Voting does not beat greedy on this set. Spend the GPU elsewhere.")
    elif gain < 0.05:
        print("  → NO (not yet). Under 5 points for a multi-x latency bill on a pipeline where")
        print("    generation is already ~95% of query time (LATENCY.md). Revisit if the set")
        print("    grows or the model changes.")
    elif gain < 0.10:
        print("  → MARGINAL. Worth it only where correctness beats latency — consider making it")
        print("    opt-in (a 'check my work' button) rather than the default path.")
    else:
        print("  → YES. A double-digit accuracy gain from sampling alone, no new model, no")
        print("    fine-tune. Implement it, and look at whether a smaller k captures most of it:")
        for k in ks:
            print(f"      k={k}: {acc_at_k[k]:.2f} ({acc_at_k[k] - greedy_acc:+.2f})")
    print(f"\n  Caveat: {n} questions is a small exam. A ±0.05 difference is inside the noise —")
    print("  treat this as a go/no-go signal, not a precise effect size.")


def main():
    p = argparse.ArgumentParser(description="Measure whether self-consistency voting helps.")
    tier = p.add_mutually_exclusive_group()
    tier.add_argument("--baseline", dest="tier", action="store_const", const="baseline",
                      help="College-level set (default) — the tier that decides things.")
    tier.add_argument("--easy", dest="tier", action="store_const", const="easy",
                      help="Grade-school regression tier. Near ceiling; not for ranking models.")
    p.set_defaults(tier=DEFAULT_TIER)
    p.add_argument("--questions", help="Explicit question file, overriding the tier flags.")
    p.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                   help="Samples generated per question (must be >= max k)")
    p.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS,
                   help="Vote sizes to score, e.g. --ks 1 5 10")
    p.add_argument("--model", default=OLLAMA_MODEL,
                   help=f"Ollama tag of the generator under test (default: {OLLAMA_MODEL}). "
                        "Everything else — prompt, questions, extraction, scoring — is held "
                        "identical, so the only difference between two runs is the model.")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    # A positive int, so that `--limit 0` is a loud error rather than silently falling
    # through to a full 30-question, ~50-minute run.
    p.add_argument("--limit", type=int, help="Only the first N questions (smoke run)")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve the tier, load and validate the questions, then stop "
                        "without generating anything.")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Parallel samples per question; >1 needs OLLAMA_NUM_PARALLEL set")
    p.add_argument("--seed", type=int, default=0, help="Seed for the resampling, not the model")
    p.add_argument("--out", help="Results JSON (default: per-tier, so tiers never clobber "
                                 "each other's runs).")
    p.add_argument("--report-only", metavar="RESULTS_JSON",
                   help="Re-print the report from a previous run's JSON, generating nothing. "
                        "The samples are the expensive part (~30 min); re-scoring them at "
                        "different k, or after changing the analysis, must not need a re-run.")
    args = p.parse_args()

    ks = sorted(set(args.ks))
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be >= 1 (0 would silently run the whole tier).")
    questions_path = args.questions or TIERS[args.tier]
    tier_label = "custom" if args.questions else args.tier
    # Per-tier AND per-model default output: one shared filename would mean an easy run
    # silently destroying the baseline run it is supposed to be compared against — and,
    # once models are swappable, a challenger destroying the incumbent it is measured against.
    out_path = args.out or os.path.join(
        RESULTS_DIR, f"self_consistency_{tier_label}_{model_slug(args.model)}.json")

    if args.report_only:
        with open(args.report_only, encoding="utf-8") as f:
            saved = json.load(f)
        per_q = saved["per_question"]
        cfg = saved["config"]
        n_samples = cfg["samples"]
        if max(ks) > n_samples:
            p.error(f"--ks max ({max(ks)}) exceeds the {n_samples} samples in that run.")
        rng = random.Random(args.seed)
        acc_at_k = {
            k: statistics.mean(
                vote_accuracy_at_k(q["sampled_answers"], set(q["gold"]), k, rng, RESAMPLE_TRIALS)
                for q in per_q
            )
            for k in ks
        }
        print_report(per_q, acc_at_k, ks, cfg)
        return
    if max(ks) > args.samples:
        p.error(f"--ks max ({max(ks)}) exceeds --samples ({args.samples}); "
                "raise --samples or lower --ks.")

    with open(questions_path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        items = items[:args.limit]
    if not items:
        p.error(f"No questions in {questions_path}")

    print(f"Loaded {len(items)} questions from {questions_path}  [tier: {tier_label}]")
    print(f"Generator: {args.model} @ {OLLAMA_BASE_URL}")
    print(f"Plan: {len(items)} × (1 greedy + {args.samples} sampled) = "
          f"{len(items) * (args.samples + 1)} generations. This takes a while.")
    if args.dry_run:
        bad = [it["id"] for it in items if not gold_forms(it)]
        print("Dry run: questions load and "
              + ("all golds normalize." if not bad else f"BAD GOLDS: {bad}"))
        print(f"Would write {out_path}")
        return

    record = run_tier(items, args.model, args.samples, ks, args.temperature, args.top_p,
                      args.concurrency, args.seed, tier_label, questions_path, out_path)
    acc_at_k = {int(k): v for k, v in record["accuracy_at_k"].items()}
    print_report(record["per_question"], acc_at_k, ks, record["config"])
    print(f"\nWrote {out_path}  (wall clock {record['config']['wall_clock_s'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
