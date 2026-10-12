# DEPLOYMENT_OPTIONS.md

Where the pipeline could run, what each layout costs, and which of the fears about
moving work to CPU survive measurement.

`DEPLOYMENT.md` describes the layout that **ships today** (two RunPod pods, §1–§2) and is
the authority on it. This file is the comparison *around* that choice: it exists because
"split the stateful half onto a cheap always-on CPU host and buy GPU by the message"
sounds obviously cheaper, and the two objections to it — *retrieval will be slow and
memory-heavy on CPU*, and *nobody knows what per-message inference costs* — are both
answerable with numbers rather than intuition.

Date of the pricing and free-tier survey: **2026-10**. Rates move; re-check §7 before
acting on §5.

---

## 1. The short version

| | Layout | Est. $/mo | Verdict |
|---|---|---|---|
| **A** | Two pods: GPU (API+Ollama) + always-on CPU (UI). **Ships today.** | **~$28** | Works; no headroom under the $30 target |
| **B** | Always-on CPU host (UI + API + retrieval + indexes) + **serverless GPU per message** | **~$7–15**, possibly ~$0–8 on free credits | **The interesting one.** See §5 |
| **C** | Everything on CPU, generation included | ~$0–7 | **Dead.** Prefill, not decode — see §3 |
| **D** | Free hosted inference for generation (Groq / Cerebras / Mistral free tiers) | **$0** | Cost floor, not a backbone — see §5.4 |

The one-sentence finding: **retrieval on CPU is cheap and fine; generation on CPU is
impossible, and it is impossible for a reason specific to RAG.**

---

## 2. What was measured, and on what

Measured 2026-10 on an **Apple M1 Pro, 8 cores, 16 GB**, CPU-only, through this repo's
own loaders (`retrieval.load_embeddings`, `sentence_transformers.CrossEncoder`) and
against the real models — `BAAI/bge-small-en-v1.5`, `BAAI/bge-reranker-v2-m3`,
`t1c/deepseek-math-7b-rl:Q4`. Generation was forced onto CPU with Ollama's
`options.num_gpu = 0`.

| Stage | GPU (A5000 — `LATENCY.md`) | CPU (M1 Pro — measured) |
|---|---|---|
| BM25 + dense + RRF | ~20 ms | already CPU |
| Query embed (bge-small) | few ms | **17.6 ms** |
| Cross-encoder rerank, pool=20 | ~850 ms | **1,990 ms** |
| …pool=10 / pool=5 | — | 1,072 ms / 589 ms |
| Prompt prefill, 1.9k–3.4k tok | ~7 s — **47 ms** warm-cache | **~40–71 s** @ 47.7 tok/s |
| Decode, 316–534 tok | 12–20 s @ 26 tok/s | 15–25 s @ **21.7 tok/s** |
| Embedder + reranker resident | 2.7 GB **VRAM** | **1.66 GB RSS** |

Reranker thread scaling, pool=20:

| threads | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| ms | 4,280 | 2,265 | 2,243 | 1,990 |

It flattens after ~2 cores: the cross-encoder is memory-bandwidth-bound, not core-bound.
Useful, because the cheap always-on hosts in §5.3 all have 2 cores.

⚠️ **The M1 Pro is the best case for CPU, not a typical one.** Unified memory gives it
bandwidth a 2-vCPU cloud box does not have. Every CPU figure here should be read as a
*floor*; §5.1 projects the penalty on real hardware, and those projections are
extrapolations, **not measurements on target hardware**.

⚠️ **Method note.** A first attempt at the thread sweep called
`torch.set_num_threads()` *after* importing torch and reported ~12 s at every thread
count — contradicting the 1,990 ms baseline for identical work. The fix is to set
`OMP_NUM_THREADS` / `MKL_NUM_THREADS` in the environment **before** torch is imported.
If these numbers are ever re-measured, set the env vars, not the Python API.

---

## 3. Why CPU generation is dead, and decode is not the reason

The obvious guess is that a 7B would decode too slowly on a CPU. On this hardware that
guess is wrong: **21.7 tok/s on CPU against 26 tok/s on the A5000** — an 18% penalty, not
a wall.

The wall is **prefill: 47.7 tok/s.** `LATENCY.md` records real prompts at 1.9k–3.4k
tokens, because the prompt *is* the retrieved context. So first-token latency on CPU is
**40–71 seconds**, before a single output token exists, and that is on the fast CPU.

This is the one place the system's own design works against it. A chat app with a
200-token prompt would prefill in ~4 s and CPU generation would be arguable. **RAG spends
its whole context budget on retrieved chunks, so it is the worst possible shape for CPU
inference.** Option C is closed, and it stays closed on a bigger CPU box — prefill is
compute-bound, so it scales with cores far better than the reranker does but nowhere near
far enough.

Corollary worth keeping: **the thing to protect when moving generation is the prefill,
not the decode.** See §4.

---

## 4. ⚠️ The trap in per-message inference: KV cache, not cold starts

`LATENCY.md` fix 3 is worth more than any other number in this document. On a
1,821-token prompt: **7,358 ms cold, 47 ms when the prefix repeats identically**, 160 ms
with a 350-token partial answer appended. That is why `api/chat.py`'s prompt order is
load-bearing (static text → history → context → question) and why the tool loops
*append* rather than insert.

**That optimisation is a property of a resident process.** A serverless worker that
scales to zero between messages starts every message with a cold cache, so:

- every message pays ~7 s of prefill instead of ~47 ms, and
- the **billed** GPU seconds per message rise by roughly 30%.

So the cost table in §5.2 is the *pessimistic* one by construction — it already assumes
no cache hit. The optimistic case is not available unless the worker stays warm, and
keeping it warm is billed:

> RunPod bills the configured idle period before scale-down. A 5-minute keep-warm costs
> ~300 s × $0.000192/s ≈ **$0.058 per conversation tail** — about twelve messages' worth
> of decode. On a three-message conversation the idle tail costs more than the answers.

This is the real trade in Option B, and it is a dial, not a yes/no: **short idle timeout
→ cheap, but every message eats 7 s of prefill; long idle timeout → fast follow-ups, and
the tail dominates the bill on short conversations.** Pick it from the family's actual
session shape in `kbm/telemetry.py`, which is already logging the data needed to decide.

### Cold starts are the lesser problem

Counter-intuitively, serverless cold starts would *improve* on what ships today.

| Path | Click → first token |
|---|---|
| Today's pod wake (`DEPLOYMENT.md`, cold path — component estimate) | **~1–2 min** |
| Modal, 7B+ | 15–30 s |
| RunPod FlashBoot (CRIU-style snapshot) | sub-200 ms on ~48% of requests; 30–90 s worst case |

Weight loading is physics and no platform escapes it: 4.2 GB of Q4 at ~500 MB/s over NFS
is ~8 s; on local NVMe at 3.5 GB/s it is ~1 s. Modal's memory snapshots skip the init
path entirely (a vLLM cold start 460 s → ~70 s). The conclusion stands either way —
**`ensure_awake()` in `app.py` exists to paper over a 1–2 minute gap that serverless would
shrink, not widen.**

---

## 5. The options in full

### 5.1 Option B — CPU host + serverless GPU per message

Split three ways rather than two:

| Half | Holds | Why there |
|---|---|---|
| Always-on **CPU host** | Gradio, FastAPI, BM25, Chroma, reranker, sandbox, telemetry, **the indexes** | Stateful. §2 says it fits in 1.66 GB |
| **Serverless GPU**, per message | Generation only | ~95% of query time, ~0% of the state |
| **Serverless GPU**, per upload | Marker / Surya extraction | A minutes-long batch job — the ideal serverless shape |

**Why the code is already most of the way there.** `OLLAMA_BASE_URL` is a single config
point (`kbm/config.py:46`), so generation is already addressable remotely. More
importantly, the tool loop lives in `api/routes.py` **on the API host**, which means agent
mode's `search_documents` keeps hitting local indexes and never needs a round trip back to
the GPU. That was the expected blocker and it is not one.

**What it costs in code** — none of it deep, all of it real:

1. `api/deps.py:56` (`/api/show` capability probe) and `api/routes.py:898` (`/api/ps`) are
   Ollama-specific endpoints. Either run Ollama inside the serverless container or replace
   both.
2. The continuation loop keys on `done_reason == "length"`. OpenAI-compatible endpoints
   give `finish_reason == "length"` — equivalent meaning, but a change threaded through
   `routes.py`, and ⚠️ **`PrefillEcho` behaviour is per-model and must be re-verified on
   whatever the new endpoint serves** (`ERRORS.md` 2026-08-20: getting this wrong emits
   nothing, silently).
3. `KBM_KEEP_ALIVE` stops meaning anything; the platform owns residency.
4. Rerank moves to CPU: **+1.2 s measured here, projected +3–5 s on a 2-core cloud box.**
   Levers in order of preference: drop `RERANK_TOP_C` 20→10 (halves it — but it also
   halves the pool the reranker can reorder, so re-read `EVALUATION.md`'s recall@pool
   before doing it), then ONNX int8 (2–4×).
   ⚠️ **Quantising the reranker invalidates `KBM_RELEVANCE_FLOOR`.** The 0.15 floor is
   calibrated against *that model's* sigmoid output scale; `evaluation/calibrate_floor.py`
   must be re-run or abstention breaks silently.

**What it saves.** The $16.80 GPU-pod line and most of the $3.50 network volume — indexes
move to the CPU host's disk, weights become the platform's cache problem.

### 5.2 What per-message inference actually costs

One answer ≈ **22 GPU-seconds** (7 s prefill + 12–20 s decode, per `LATENCY.md`), assuming
no warm cache (§4).

| Platform | Rate | Per message | 20/day | 50/day | 100/day |
|---|---|---|---|---|---|
| RunPod flex, 24 GB (A5000/L4) | $0.69/hr | $0.0042 | $2.53 | $6.33 | $12.66 |
| Modal L4 | $0.000222/s | $0.0049 | $2.93 | $7.32 | $14.64 |
| Modal A10G | $0.000306/s | $0.0067 | $4.04 | $10.10 | $20.19 |

Add the always-on CPU host: **$7.30** (RunPod CPU pod, today's line), **~$0** (Oracle
Always Free), or ~$4–5 (cheap ARM VPS).

Two structural points matter more than the rates:

- **Today's bill pays for availability, not use.** `DEPLOYMENT.md §2` bills 3.5 hr/day of
  wall clock — 105 hr/mo — to deliver maybe 10–30% of that in actual compute. Serverless
  bills only the used part, which is why a bursty family workload is the case where it
  wins.
- **Modal includes $30/month of free credits** (~27 A10G-hours). At 100 msg/day the real
  requirement is ~17 GPU-hours/mo, so this workload plausibly lands *inside* the free
  tier. Treat that as a reason to prototype there, not as a permanent price.

### 5.3 The always-on CPU host

⚠️ **Oracle's Always Free tier was quietly halved to 2 OCPU / 12 GB on 2026-06-15** (no
announcement; 1,500 OCPU-hours + 9,000 GB-hours/mo). 12 GB is ample against §2's 1.66 GB.
2 cores is the number that matters, and §2's thread sweep says 2 cores costs ~2.3 s on
M1-class hardware — call it 3–5 s on Ampere A1.

Also: Oracle reclaims idle Always Free instances, and ARM capacity is frequently
unavailable by region. That is survivable **only** because `ops/backup_indexes.py` already
exists and the indexes are the sole copy of the family's library (`ARCHITECTURE.md §5`).
Do not put the only copy of anything on a free tier that can reclaim it.

### 5.4 Option D — free hosted inference

Real, free, and fast: Groq (~14,400 req/day on Llama 3.1 8B, 6k tok/min), Cerebras
(5 req/min, 30k tok/min, 1M tok/day), Mistral (~1B tok/mo on open weights). All serve
**open weights**, so this does not breach the no-paid-API constraint in `CLAUDE.md`.

It is still not the backbone, for three reasons that are about this project rather than
about the vendors:

1. **No `deepseek-math-7b-rl`, and no arbitrary tag.** `EVALUATION.md §12`'s bake-off and
   `kbm/llm_profiles.py` are both built on naming an Ollama model.
2. **It closes the fine-tuning door**, which `CLAUDE.md` lists as a standing goal.
3. Rate limits are per-provider and volatile — Cerebras' free catalogue collapsed from
   ~a dozen models to two during 2026.

Worth knowing as the **cost floor** any other option is measured against, and worth
keeping as a fallback generator if the GPU half is ever unavailable.

### 5.5 Option A — what staying costs

$28/mo, documented in `DEPLOYMENT.md §2`, zero migration, and the known weakness is the
1–2 minute wake (§4). The reason not to dismiss it: every number in Option B's favour is
an estimate, and ⚠️ **`DEPLOYMENT.md` itself flags that the cold-path table is assembled
from component measurements rather than one end-to-end timing.** Option A is the only row
here backed by a running system.

---

## 6. Recommended sequence

Ordered so each step is independently useful and none of them is a one-way door:

1. **Time a real wake** and replace `DEPLOYMENT.md`'s cold-path estimate with the measured
   number. Everything in §4 compares against that figure; right now it is an estimate
   being used as a baseline.
2. **Read the session shape out of `kbm/telemetry.py`** — messages per conversation and
   gaps between them. That is exactly the data §4's idle-timeout dial needs, and it is
   already being collected.
3. **Move Marker/Surya ingest to a serverless GPU function.** The cleanest win and the
   most isolated change: extraction is already a job behind `POST /upload` +
   `GET /jobs/{id}`, it is the reason the card must be 24 GB at all (upload peak
   12.7–15.2 GB vs 8.2–10.7 GB steady, `ARCHITECTURE.md §4`), and at 0.3–1 s/page a
   textbook costs $0.02–0.07.
4. **Then** generation, behind `OLLAMA_BASE_URL`, with the §5.1 checklist.

Steps 1–2 are measurements, 3 shrinks the GPU requirement, and only 4 touches the serving
path.

⚠️ **Exit criterion.** If step 1 shows the current wake is already acceptable to the family
and telemetry in step 2 shows long conversations (where the warm KV cache dominates),
**Option A is the right answer** and this document's job was to rule the others out. Do not
migrate for the table in §5.2 alone.

---

## 7. Sources

Pricing and platform behaviour, surveyed 2026-10. Rates move.

- RunPod — [pricing](https://www.runpod.io/pricing) ·
  [2026 cost guide](https://flexprice.io/blog/runprod-pricing-guide-with-gpu-costs) ·
  [serverless & storage rates](https://www.hivenet.com/post/runpod-pricing-complete-guide-to-gpu-cloud-costs) ·
  [per-GPU rates](https://gpuperhour.com/providers/runpod)
- Modal — [GPU pricing](https://computeprices.com/providers/modal) ·
  [pricing analysis](https://www.beam.cloud/blog/modal-pricing-explained) ·
  [serverless GPU platform comparison](https://modal.com/resources/best-serverless-gpu-platforms-inference)
- Cold starts — [RunPod FlashBoot](https://www.runpod.io/blog/serverless-gpu-cold-starts-flashboot) ·
  [cold starts are a weight-loading problem](https://dreaming.press/posts/2026-06-27-scale-to-zero-llm-inference-gpu-cold-starts.html) ·
  [four fixes that work](https://www.spheron.network/blog/gpu-cold-start-llm-inference-2026/) ·
  [scale-to-zero compared](https://www.buildmvpfast.com/blog/scale-to-zero-serverless-gpu-modal-runpod-ai-hosting-2026)
- Free tiers — [Oracle halves Always Free](https://www.infoq.com/news/2026/07/oracle-cloud-free-tier-limits/) ·
  [what the new limits are](https://terminalbytes.com/oracle-cloud-free-tier-changes-2026/) ·
  [free LLM APIs compared](https://openrouter.ai/blog/tutorials/free-llm-apis-compared/)
- Reranker on CPU — [bge-reranker-v2-m3 ONNX int8](https://huggingface.co/tss-deposium/bge-reranker-v2-m3-onnx-int8) ·
  [cross-encoder ONNX latency issue](https://github.com/huggingface/sentence-transformers/issues/2470)

Internal: `DEPLOYMENT.md` (shipping layout, cost, cold path), `ARCHITECTURE.md` (residency
map, VRAM budget), `LATENCY.md` (where query time goes, the KV-cache measurement),
`evaluation/EVALUATION.md` (recall@pool, the floor calibration), `ERRORS.md`
(`PrefillEcho`, 2026-08-20).
