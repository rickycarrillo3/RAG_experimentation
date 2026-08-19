# Latency

Where query time actually goes, what was changed to reduce it, and what is left.

All numbers were measured on the **Mac (MPS, no CUDA)** against `t1c/deepseek-math-7b-rl:Q4`
via Ollama, with a realistic prompt (5 retrieved chunks + conversation history). The RunPod GPU
pod is roughly 10× faster; the *shape* of the breakdown holds on both, but the absolute figures
below are the Mac's.

---

## 1. The baseline budget

| Stage | Cost | Share |
|---|---|---|
| BM25 + dense + RRF | ~20 ms | ~0% |
| Cross-encoder rerank (20 pairs) | ~850 ms | ~4% |
| **Prompt prefill (1.9k–3.4k tokens)** | **~7 s** | **~30%** |
| **Decode (316–534 tokens @ ~26 tok/s)** | **12–20 s** | **~60%** |
| Cold model load (after 5 min idle) | +4.4 s | one-off |
| **Total** | **~20–25 s** | |

**Retrieval is not the problem.** The reranker, which `EVALUATION.md` scrutinises hardest, is ~4%
of query time — its real cost is 2.2 GB of VRAM, not milliseconds. Generation is ~95%, and two
thirds of that was avoidable.

---

## 2. What changed

### Fix 1 — Stream the answer (`app.py`, `query.py`, `test_chat.py`)

`handle_chat` was a blocking `chain.invoke`: nothing appeared until the entire answer had decoded.
It is now a generator that yields per token, and Gradio repaints on each yield. `query.py` and
`test_chat.py` use `chain.stream()` and print tokens as they arrive.

This changes *perceived* latency, not total work — but that is the number a user feels. A
placeholder is painted before retrieval so the UI reacts immediately.

Measured end-to-end through the real `app.py`: **145 incremental yields, first token at 4.4 s,
total 9.8 s** (versus a ~20–25 s silent wait before).

### Fix 2 — Cap `num_predict` at 350 (was 1024)

Decode is ~60% of query latency and deepseek-math-7b-rl is a chain-of-thought solver: it fills
whatever budget it is given. It produced **534 tokens on a one-line conceptual question** — ~20 s
of decode. The system prompt's "Always show full working step by step" was actively buying that
tail; it now asks for length proportionate to the question.

Same query at `num_predict=350`: **11.5 s → 6.9 s total.**

### Fix 3 — Reorder the prompt so the KV cache can hit

**This is the non-obvious one.** Ollama/llama.cpp already caches the KV of the longest common
*prefix* between consecutive prompts — it is on by default. You do not enable it; you earn it by
prompt ordering. Verified directly:

| scenario | prefill |
|---|---|
| identical prompt repeated | 6.24 s → **0.05 s** |
| text appended at the **end** | **0.17 s** (hit) |
| text changed near the **start** | 5.69 s (full recompute) |

`SYSTEM_PROMPT` used to interpolate `{context}` (fresh every query) *before* `{history}`. That put
a guaranteed change at token ~150, so everything after it — the whole 3.4k-token prompt — was
re-prefilled every single turn.

The order is now **static text → history → context → question**, with `context` moved out of the
system message into the human message. The static block never changes and history only grows by
appending, so both stay cached; the per-query context sits last, where its churn costs least.

Prefill over a 6-turn conversation:

| turn | before | after |
|---|---|---|
| 0 | 6.32 s | 5.55 s |
| 2 | 8.40 s | 7.40 s |
| 5 | **13.43 s** | **7.40 s** |
| **total** | **57.0 s** | **41.6 s** |

The old layout's prefill *grew with every turn*; the new one goes flat at ~7.2 s, which is just the
uncacheable retrieved context. The saving grows with conversation length — ~0.8 s at turn 0, ~6 s
by turn 5.

### Fix 4 — `keep_alive="30m"`

Nothing set `keep_alive`, so Ollama unloaded the model after 5 minutes idle and the next question
paid a **4.4 s** cold load. For intermittent family use that is *every* question. Set in all three
entry points.

### Supporting change — block-trimmed history window

Found while verifying fix 3. `_format_history` used `history[-6:]`, a per-turn sliding window. Once
full, it drops the oldest message every turn, which shifts the *start* of the history block — the
one thing a prefix cache cannot survive. Measured: sliding locks at a flat **9.0 s** prefill from
turn 4 on and never gets a hit again.

`_history_window()` now trims to a fixed grid instead:

```python
start = max(0, ((len(history) - HISTORY_KEEP) // HISTORY_BLOCK) * HISTORY_BLOCK)
```

with `HISTORY_KEEP = HISTORY_BLOCK = 8` messages (a turn is two: student + tutor). The window start
only moves in steps of 8, so between moves history grows purely by appending and stays cached:

```
msgs  window        size
  14  [ 0.. 14)      14      grows freely, cached
  16  [ 8.. 16)       8   <- anchor: start jumps, one full re-prefill
  22  [ 8.. 22)      14      grows again, cached
  24  [16.. 24)       8   <- anchor
```

Anchors land every 8 messages (4 turns), evenly spaced.

**Be honest about what this buys.** In steady state it is roughly a wash on latency — block
averages ~8.9 s/turn against sliding's ~9.05 s, because the re-anchor turn is expensive (13.4 s)
and pays back the cheap turns. The actual win is that you get **~2× the conversation context for
the same latency**: at turn 8 the block window prefilled 4079 tokens in 6.87 s while sliding
prefilled 2399 tokens in 9.07 s. More tokens, less time — that is the cache working.

There is deliberately **no** `HISTORY_MAX`. Until history reaches `KEEP + BLOCK` messages the
integer division is 0, so the window starts at 0 and keeps everything — "grow freely at first"
falls out of the arithmetic. An earlier version had a separate max that disagreed with the grid,
which made the first trim lurch twice in consecutive turns.

---

## 3. Net effect

Same one-line conceptual question, cold:

- **before:** ~20–25 s, nothing on screen until the end
- **after:** first token at ~4.4 s, complete at ~9.8 s

In a long conversation the gap widens, because the old prompt layout's prefill grew every turn
while the new one is flat.

---

## 4. Not done, and why

**Swapping the reranker.** 850 ms of a 20 s problem — the wrong end. If it ever matters, measured
alternatives on this Mac for 20 pairs: `bge-reranker-v2-m3` 935 ms (mps+fp16), `bge-reranker-base`
350 ms, `ms-marco-MiniLM-L-6-v2` 59 ms. The MiniLM is 16× faster but small and English-only against
a LaTeX corpus — run `evaluation/eval.py` for recall@5 before trusting it. On the GPU pod this
stage is ~50–150 ms and the question is moot.

**A smaller generation model.** Decode is memory-bandwidth-bound, so a 1.5–3B math model would be
~3–4× faster. That is a real quality trade; decide it with the eval harness, not by feel.

**Running on the GPU pod.** The single largest available win (~10×). Fixes 2–4 are prompt/config
bugs that waste time on *both* machines, which is why they were worth doing regardless.

---

## 5. Tuning notes

- `NUM_PREDICT` (350) is the direct lever on decode time — the dominant cost; every extra token
  is ~38 ms on the Mac. **Raising it is no longer the answer to truncation.** Since 2026-08-19
  the server detects `done_reason == "length"` and resumes the generation up to
  `KBM_MAX_CONTINUATIONS` times (default 2), labelling the answer only if it is still cut off.
  Raising the cap moves the cliff; continuing removes it. Lower `MAX_CONTINUATIONS` to bound
  worst-case answer latency, not `NUM_PREDICT`.
- Continuation is cheap because it **appends** to the prompt, which is the same prefix rule as
  fix 3. Measured on a 1821-token prompt: 7358 ms cold, **47 ms** repeated identically,
  **160 ms** with a 350-token partial answer appended — against ~13 s to decode those tokens.
- ⚠️ **`prompt_eval_count` does not tell you whether the cache hit.** It reports *total* prompt
  tokens, not the ones actually computed: all three rows above report 1821. Only
  `prompt_eval_duration` distinguishes 7358 ms from 47 ms. Reading the count alone would tell
  you prefix caching is not working when it is.
- `HISTORY_KEEP` / `HISTORY_BLOCK` (8/8) trade context for prompt size. They now hold 8–14 messages
  versus the old fixed 6, so the model sees more of the conversation. Lower both to reduce prompt
  tokens; keep them equal and a multiple of 2 for evenly spaced anchors.
- **Never put per-query content before stable content in a prompt.** That is the rule fix 3
  encodes, and it is easy to undo by accident when editing `SYSTEM_PROMPT`.

## 6. Reproducing

Per-stage retrieval timings come from `retrieval.retrieve_detailed()` (`timings` dict) and are
reported by `evaluation/eval.py`. Generation timings above were taken from Ollama's own
`prompt_eval_count` / `prompt_eval_duration` / `eval_count` fields on `/api/chat` with
`stream: true` — `prompt_eval_duration` is the prefill number, and it dropping to ~0.05 s is how
you confirm a prefix-cache hit.
