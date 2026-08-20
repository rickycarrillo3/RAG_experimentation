# The generator: what deepseek-math is good at, what it is bad at, and why

The system's generator is `t1c/deepseek-math-7b-rl:Q4` via Ollama. `LATENCY.md` establishes
that generation is ~95% of query time, and `EVALUATION.md §11` establishes that its errors on
the reasoning tiers are not the kind sampling can fix. This document answers the question those
two leave open: **what exactly is wrong with it, and which lever matches the failure.**

The short version: it is not bad at mathematics. It is bad at *executing arithmetic*, and the
distinction is not pedantic — it determines which of four available levers is worth spending
GPU time on. Three of them are aimed at the wrong failure.

Companion documents: `EVALUATION.md §11` (self-consistency, where the failures were first
measured), `EVALUATION.md §12` (the bake-off that tests a replacement), `LATENCY.md` (why the
generator dominates cost at all).

---

## 1. Scores

### 1.1 Published benchmarks

From the **Qwen2.5-Math Technical Report** ([arXiv:2409.12122](https://arxiv.org/abs/2409.12122),
Table 3) — few-shot chain-of-thought, English, unquantized weights. This table is the useful
kind of reference because *one harness scored every row*, so the rows are comparable to each
other, which a table assembled from each model's own README would not be.

| model | GSM8K | MATH | context |
|---|---|---|---|
| **DeepSeekMath-7B-RL** (incumbent) | 88.2 | 52.4 | 4,096 |
| **Qwen2.5-Math-7B-Instruct** | **95.2** | **83.6** | 4,096 |
| Qwen2.5-7B-Instruct | 91.6 | 75.5 | 131,072 |
| Qwen2-Math-7B-Instruct | 89.9 | 75.1 | 4,096 |
| Mathstral-7B-v0.1 | 84.9 | 56.6 | 32,768 |
| Llama-3.1-8B-Instruct | 76.6 | 47.2 | 131,072 |

DeepSeekMath's own paper ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)) reports
**88.2 / 51.7**; Qwen's independent re-scoring gives 52.4. The sub-point difference between a
model's self-report and an outside harness is itself a calibration on how much scoring
methodology moves these numbers.

Qwen2.5-Math-7B-Instruct also reports **94.6 / 85.2 under TIR** (tool-integrated reasoning —
the model writes and executes Python). Hold that number; §4 explains why it is the one that
matters most here.

**Every row above is CoT.** The TIR numbers in the previous paragraph are quoted for §4.3's
argument and are deliberately *not* what the bake-off arms were chosen on — nothing in this
system executes tool calls, so selecting a model on its TIR column would be choosing on a
capability we have not built.

**These numbers do not decide what ships**, for the reasons in `EVALUATION.md §12.2`: wrong
question distribution, public test sets with known contamination, unquantized weights against
our Q4, and **few-shot prompting against our zero-shot `SC_PROMPT`**. That last one cuts
predictably rather than randomly: few-shot exemplars mostly buy format compliance, so expect
our absolute numbers to sit below every published row for every arm. They are a reason to
spend two hours of GPU time on a bake-off, not a result.

### 1.2 Measured on our questions

`EVALUATION.md §11.5`, easy tier (20 grade-school questions), Mac/Metal, Q4:

| setting | accuracy |
|---|---|
| greedy (ships today) | 0.85 |
| majority vote k=5 | 0.88 |
| majority vote k=10 | 0.90 |

Three greedy errors. Voting fixed one and could never fix the other two. The college tier
(30 questions — the tier that actually discriminates) has not been run yet; it is the first
arm of `model_bakeoff.py`.

### 1.3 Reading the gap

88.2 on GSM8K is not a bad score. GSM8K is arithmetic-*light* word problems with small
numbers, where memorization covers most of the work. The gap opens on MATH — 52.4 against
Qwen2.5-Math's 83.6 — which has larger intermediate values and more steps to slip on. **The
benchmark spread itself points at execution rather than knowledge**, before we look at a
single trace.

---

## 2. The failure, from the traces

`EVALUATION.md §11.6` records the two errors that voting could not touch. Both are worth
reading rather than counting.

### 2.1 r07 — remainder of 7¹⁰⁰ mod 13 (gold: 9)

The greedy trace:

> find the smallest k with 7^k ≡ 1 (mod 13) … k = 12 … divide 100 by 12, quotient 8
> remainder 4 … so 7^100 ≡ 7^4 … **7^4 = 2401 ≡ 3 (mod 13)**

Everything up to the last clause is correct. The order of 7 mod 13 is 12; 100 mod 12 = 4;
7⁴ = 2401 — that is correct four-digit arithmetic. Then it wrote `2401 ≡ 3 (mod 13)` as a
**single atomic assertion**. 13 × 184 = 2392, so the answer is 9.

The important part is what is *absent* from the trace: there is no division. No "13 × 184 =
2392", no subtraction, nothing. The model did not compute the modulus and get it wrong — it
never attempted to compute it.

Ten samples returned 1×3, 3×3, 7×3, 9×1: every one a member of the cycle of 7 mod 13. The
model reliably knows which neighbourhood the answer lives in and is guessing the position
within it. **That is associative recall, not arithmetic.**

### 2.2 r03 — sum of integers 1..100 divisible by 3 or 5 (gold: 2418)

Samples included 2385 and 2413. Inclusion–exclusion was set up correctly every time; the
arithmetic slipped. One sample returned 285 — the sum of the multiples of 5 alone, a run that
stopped before completing the union.

Same shape: method right, execution wrong, and the wrong answers scatter instead of clustering
on a single systematic mistake.

---

## 3. Why

### 3.1 It is not tokenization

This is the standard explanation and the first place to look: LLMs are said to be bad at
arithmetic because number tokenization destroys place value — `2401` becomes `240` + `1`,
digits no longer align, carries cannot be learned. **It does not apply to this model.**

From `deepseek-math-7b-rl`'s `tokenizer.json`:

```
pre_tokenizer[5]  Digits: individual_digits = True
multi-digit tokens in vocab: 0
single-digit tokens:        10
```

Every digit is its own token and the vocabulary contains no multi-digit numerals at all. That
is current best practice, and Qwen2.5-Math does exactly the same thing by a different route
(its pre-tokenizer regex splits on `\p{N}`, one number character per token). Reproduce with:

```bash
curl -s https://huggingface.co/deepseek-ai/deepseek-math-7b-rl/raw/main/tokenizer.json \
  | python3 -c "import json,re,sys; d=json.load(sys.stdin); \
      print([p for p in d['pre_tokenizer']['pretokenizers'] if p['type']=='Digits']); \
      print('multi-digit vocab entries:', sum(bool(re.fullmatch(r'\d{2,}',k)) for k in d['model']['vocab']))"
```

Recorded here because it is the obvious hypothesis, it is wrong, and the next person to ask
this question should not spend an afternoon on it. Tokenization is not what separates 52.4
from 83.6.

### 3.2 Fixed depth: an algorithm attempted in one forward pass

A transformer has a bounded number of layers per token and no loops. Long division is an
iterative algorithm — repeated comparison, multiplication and subtraction with carries — so a
model has exactly two ways to run one:

1. **Spread it across output tokens.** This is what chain-of-thought *is for*: the token
   stream is scratch memory, and each step gets its own forward pass.
2. **Do it inside a single forward pass.** Bounded by depth. An algorithm that needs more
   serial steps than the model has layers cannot fit, and what emerges instead is the network's
   best single-shot guess.

deepseek took the second path. `2401 ≡ 3 (mod 13)` in one step is option 2, and option 2 for
a modulus this size is not a computation — it is a lookup of a fact the model does not have
memorized. Small products and common moduli *are* memorized from training data; `2401 mod 13`
is a rare string, so the model produces something from the right residue class-ish
neighbourhood and moves on.

### 3.3 A strategy error underneath the execution error

The more interesting failure is that deepseek chose the harder path. Reducing at every step —
7² ≡ 10, then 7⁴ ≡ 10² = 100 ≡ 9 — never produces a value above 100 and reaches the right
answer using arithmetic the model demonstrably can do. A mathematician reduces early
*precisely* to avoid carrying 2401 around.

deepseek computed the large intermediate and then needed a division it could not perform. So
the model is not only failing to execute; it is selecting methods without regard to whether it
can execute them. That is a distinct defect, and unlike the others it is potentially
addressable in the prompt (§4.4).

### 3.4 Outcome-only RL trains method, not execution

DeepSeekMath-RL was trained with GRPO on **outcome** reward: the signal is whether the final
answer matched. That is very efficient at teaching *method selection* — which is exactly what
the traces show it doing well, every time — and applies almost no targeted pressure to
execution. There is no per-step credit assignment, so nothing specifically penalizes "you
skipped the division and asserted the result." Method errors dominate the loss signal early in
training; arithmetic slips are a long tail the outcome reward barely separates from noise.

Process supervision — rewarding correct *steps* rather than correct answers — is the technique
aimed at this failure, and it is not what this model received.

There is corroboration in DeepSeekMath's own paper: it reports a large gap between
chain-of-thought and tool-integrated reasoning on MATH. The authors are documenting that
arithmetic execution, not mathematical knowledge, is their model's binding constraint.

### 3.5 Q4 quantization — a live, untested suspect

The mechanism is specific to arithmetic. Prose is redundant: many next tokens are acceptable,
the output distribution is broad, and small weight perturbations rarely flip the model to
something *wrong*. Arithmetic has exactly one correct token, and the logit gap between
competing digits is frequently small — 4-bit noise flips near-ties. The error also does not
self-correct: a wrong digit at step 3 propagates deterministically through everything after
it, where a slightly-off word choice in prose gets absorbed.

**This has not been tested.** Re-running the easy tier at Q8 or fp16 separates "the model
can't" from "the quantization can't," and it is now a one-flag change:

```bash
python evaluation/self_consistency.py --easy --model <same-model-at-Q8>
```

Until that runs, every claim in this document about the *model's* arithmetic carries an
asterisk.

### 3.6 Scale and age

7B parameters, 2024-era. Arithmetic reliability scales steeply with model size, and multi-step
execution is where small models degrade first. This is context rather than a lever — we are
not running a 72B model on a single pod within the cost model in `ops/idle_stop.py`.

---

## 4. Which lever matches

Four levers, and the analysis above rules out two of them.

### 4.1 More sampling — ✗ ruled out, measured

`EVALUATION.md §11.5`: +5 points at k=10 for 10× the decode, on the stage that already owns
~95% of query time. The verdict was no, and §2 explains *why* it had to be no rather than
merely why it was.

Voting reorders a pool; it cannot add to it. Self-consistency repairs **random** errors when
the correct answer is reliably present among the samples. When the model guesses rather than
computes, the pool contains four plausible residues and the right one appears once — more
samples produce more guesses. This is a structural mismatch between the fix and the failure,
not a tuning problem, and no value of k changes it.

### 4.2 A better generator — being tested

`evaluation/model_bakeoff.py`, `EVALUATION.md §12`. Qwen2.5-Math-7B-Instruct is +31 MATH on
paper against a same-size, same-quantization, same-context-window incumbent, which is the most
promising single swap available. Qwen2.5-7B-Instruct is the generalist arm and brings a
128K context against the incumbent's 4K, which matters for the RAG path even at parity on math.

Not yet run. See `EVALUATION.md §12.5` for what a win has to mean before `OLLAMA_MODEL` changes.

### 4.3 Tool-integrated reasoning — ✓ the lever aimed at this failure

Let the model emit Python for the arithmetic step and execute it, so `2401 % 13` is *computed*
rather than recalled. This targets §3.2 directly: it removes the requirement that an iterative
algorithm fit inside a fixed-depth forward pass, by moving the algorithm somewhere that has
loops.

Qwen2.5-Math gains +1.6 MATH from TIR on top of an already-strong CoT score; DeepSeekMath's
own paper shows a substantially larger jump, precisely because it has more execution error to
recover. If the bake-off only partly closes the gap, this is the next experiment.

The cost is not free: it means a sandboxed execution path in `api/`, which is a real security
and operational surface for a family-facing service, and it interacts with the SSE streaming
contract in `api/routes.py` the same way voting does — the answer does not exist until the
tool call returns.

### 4.4 Prompt-level: reduce intermediates early — cheap, untested

§3.3 identifies a strategy error that no model swap is needed to test. Instructing the
generator to reduce intermediate values at every step rather than compute large values and
reduce at the end targets the method-selection defect without touching the model, the
quantization, or the serving path.

It is nearly free to try and it is falsifiable: if it moves r07, that is information the
bake-off cannot produce. Note the constraint from `LATENCY.md` — `SYSTEM_PROMPT` order is
load-bearing for KV-cache reuse, so this text belongs in the static prefix, not near the
question.

---

## 5. What to conclude

1. **The model knows the mathematics.** In every failure examined, method selection was
   correct. Retrieval supplying "the method" — the thing this RAG system is built to do — is
   not what is failing on these questions.
2. **It cannot reliably execute multi-digit arithmetic**, and it does not reliably *notice*
   that it is about to need to.
3. **The failures are guess-shaped, not slip-shaped.** That is what makes sampling useless
   here and what makes tool use the matched fix.
4. **Two cheap experiments are outstanding** before any of the expensive ones: Q8 (§3.5) and
   the reduce-early prompt (§4.4). Both are one flag or one string.
5. **Every measured number in this document is from the easy tier**, which `EVALUATION.md
   §11.1` is explicit is a regression tripwire and not a quality score. The college tier is
   what decides, and it has not been run.
