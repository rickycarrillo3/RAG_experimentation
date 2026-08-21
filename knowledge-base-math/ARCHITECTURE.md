# Architecture — where the bytes live

What each stage of a question does, and which piece of hardware actually holds it.

`DEPLOYMENT.md` covers hosting and cost; `LATENCY.md` covers where time goes;
`evaluation/EVALUATION.md` covers retrieval quality. This file answers a different
question: **GPU, CPU, disk, or your laptop?**

The short version: the GPU runs neural-network forward passes and nothing else. Four
models qualify. Everything else in the pipeline — including two thirds of retrieval — is
ordinary CPU work.

---

## 1. The tiers

**GPU and CPU are two components of one machine, not two machines.** A RunPod pod is a
container with vCPUs, system RAM, a GPU on the PCIe bus, and a network volume mounted at
`/workspace`. One `uvicorn` process runs the whole pipeline and dispatches to the GPU the
way it dispatches to a disk read.

```
┌─ ONE RunPod pod ─────────────────────────────┐
│                                              │
│   vCPUs + system RAM ──── "CPU" below        │
│         │                                    │
│         │ PCIe                               │
│         ▼                                    │
│   RTX A5000, 24 GB VRAM ── "GPU" below       │
│                                              │
└──────────────┬───────────────────────────────┘
               │ network mount
               ▼
     /workspace ── "Disk" below (not in the box)
```

| Tier | Holds | Survives a pod restart? |
|---|---|---|
| **GPU / VRAM** | Model weights + the matrix math over them. Four models, never more. | No |
| **CPU / RAM** | HTTP, tokenizing, sorting, ranking, merging, logging — all the bookkeeping. | No |
| **Disk `/workspace`** | Documents, indexes, model caches, telemetry. Network-attached. | **Yes — only this** |
| **Your Mac** | Once deployed: a browser tab. Nothing persisted. | n/a |

The volume is the only tier that is not physically in the box, which is why it survives
pod destruction and why it is billed separately ($0.07/GB/mo).

---

## 2. The query path

Percentages are each stage's share of wall-clock time, measured on the Mac
(`LATENCY.md`). The pod is roughly 10× faster in absolute terms; the *shape* holds.

| # | Stage | Runs on | Time |
|---|---|---|---|
| 1 | Question typed, sent over HTTPS | Your Mac | — |
| 2 | `POST /chat` — auth, username normalize, open SSE | CPU | — |
| 3a | BM25 top-10 | CPU | ~10 ms |
| 3b | Embed the query (`bge-small`) | **GPU** | few ms |
| 3c | Chroma HNSW search | CPU | ~10 ms |
| 4 | RRF fusion → candidate pool of ≤20 | CPU | <1 ms |
| 5 | Cross-encoder rerank → top 5 | **GPU** | ~850 ms · 4% |
| 6 | Prompt prefill (1.9k–3.4k tokens) | **GPU** | ~7 s · 30% |
| 7 | Decode (≤350 tokens) | **GPU** | 12–20 s · 60% |
| 7a | *(tools only)* Run the model's Python | CPU, **separate process** | ~0.05–0.5 s per program |
| 7b | *(agent only)* Search again — **re-enters rows 3a–5** | CPU + **GPU** | ~0.9–3 s per search |
| 7c | *(tools only)* Re-prefill the grown transcript and decode on | **GPU** | ×1–3 of rows 6–7 |
| 8 | SSE frames out + one telemetry line | CPU | — |
| 9 | LaTeX renders in the tab | Your Mac | — |

Two things this table is for:

**"Dense retrieval" is not the GPU half of hybrid search.** It splits across both tiers:
the GPU turns the question into 384 floats, and then Chroma's HNSW index — a graph walk,
not a matrix op — does the actual searching on CPU. Of the six things in the retrieval
pipeline, exactly one touches the GPU.

**Retrieval is ~4% of a query; generation is ~90%.** Retrieval is scrutinised so hard
not because it is slow, but because it decides *what the model is allowed to know*.
Optimising it for speed is optimising the wrong 4%.

**Tool use (rows 7a–7c) is the only thing that loops.** Every other row runs once. With a
tool protocol on, the model can stop mid-answer, hand work to the CPU, and continue from
the result. The tools themselves are cheap and live off the GPU; what costs is that each
round re-enters rows 6–7 with a longer transcript.

**Row 7b is new, and it is the structural change.** Until agent mode, retrieval ran exactly
once per request, before generation — rows 3a–5 were strictly upstream of row 6. With
`KBM_TOOLS` on, `search_documents` puts them *inside* the generation loop: the model can
decide the retrieval the server already did was not the one the question needed, rewrite the
query, and send the embedder and the cross-encoder round again mid-answer. That is why
`_add_retrieval_timings` accumulates rather than assigns — `bm25_ms`/`dense_ms`/`rerank_ms`
are now per-request totals across every search, not the cost of one.

It also means the **rerank is the expensive part of a re-search**: ~850 ms of the ~0.9–3 s
in row 7b is the cross-encoder, on the GPU, competing with nothing else because decode is
paused. `agent.MAX_SEARCH_ROUNDS = 2` is that budget as much as it is a context budget.

Three timing fields keep these apart, and they must stay apart: `tool_ms` is sandbox only,
`search_ms` is model-initiated retrieval, `generate_ms` is everything. Folding retrieval
into `tool_ms` would make every previously logged `tool_ms` incomparable. Both tools run in
a thread (`anyio.to_thread`), like the upfront retrieval — a `subprocess.communicate` or a
cross-encoder forward pass on the event loop would stall every other client's token stream.

---

## 3. The upload path

A different shape: GPU-heavy at the front, then a CPU tail ending in the only bytes here
that cannot be regenerated. Runs once per document and takes minutes, which is why
`POST /upload` returns a job handle instead of blocking.

| # | Stage | Runs on | Note |
|---|---|---|---|
| 1 | PDF uploaded | Your Mac | → `POST /upload`, returns a job id |
| 2 | Marker/Surya read the pages → Markdown+LaTeX | **GPU** | ~0.3–1 s/page on GPU vs. ~minutes/page on Mac CPU |
| 3 | Split into chunks (equations kept atomic) | CPU | `kbm/chunking.py` — cheap and deterministic |
| 4 | Embed every chunk | **GPU** | same loader as query-time, so both land in one vector space |
| 5 | Write PDF + `.mmd` + Chroma + BM25 | Disk | under `$DATA_DIR`, namespaced per user |

This is the reason the pod needs a GPU for more than generation — and the reason ingest
is serialised to one worker (§4).

---

## 4. VRAM budget on a 24 GB card

| Model | Role | When | VRAM |
|---|---|---|---|
| `deepseek-math-7b-rl:Q4` | generation | query | ~5.5 GB incl. KV cache |
| `bge-reranker-v2-m3` | cross-encoder | query | ~2.2 GB fp32 |
| `bge-small-en-v1.5` | dense embed | query + ingest | ~0.5 GB |
| | | **steady state** | **~8.2 GB** |
| Marker / Surya | PDF → LaTeX | ingest only | ~3–5 GB peak |
| | | **upload peak** | **~12.7 GB** |

The gap between those two rows is why 24 GB was the right buy rather than 12 GB. It is
also why **ingest is serialised** (`_ingest_pool`, one worker): two extractions at once
would each want ~4.5 GB on top of the 8.2 GB the query models never give back. One
worker turns a possible OOM into a queue.

Numbers from `evaluation/EVALUATION.md §6`.

---

## 5. Every component, by tier

| Component | Tier | Why there |
|---|---|---|
| The generator (`KBM_LLM_MODEL`) | GPU | 7–8B params of matrix multiply per token |
| `bge-reranker-v2-m3` | GPU | 568M-param transformer, 20 pairs per query |
| `bge-small-en-v1.5` | GPU | 33M-param encoder, query + ingest |
| Marker / Surya | GPU | vision models over page images; ingest only |
| BM25Okapi scoring | CPU | pure-Python word counting — no tensors exist |
| Chroma HNSW search | CPU | graph traversal, not a matrix op |
| RRF fusion | CPU | arithmetic on ~20 rank positions |
| FastAPI + SSE framing | CPU | HTTP, auth, streaming |
| Chunk splitting | CPU | string operations over the `.mmd` |
| Python sandbox (`kbm/tools/sandbox.py`) | CPU, own process | executes model-written code, reached from either tool protocol; a forked interpreter with rlimits and a scrubbed env, never a thread and never the GPU. See `DEPLOYMENT.md §8`. |
| Tool protocols (`kbm/tools/tir.py`, `kbm/tools/agent.py`) | CPU, negligible | pure string and schema handling — deciding what the transcript looks like. The work they trigger is the sandbox (CPU) and, for `agent.search_documents`, a full re-entry of the retrieval rows above (CPU + GPU). See `AGENT.md`. |
| Telemetry writer | CPU | one JSON line appended per answer |
| Idle-stop watchdog | CPU | a timer — the cheapest thing here, and the entire cost model |
| Uploaded PDFs | — | **not retained**; held in a temp dir for the length of the ingest |
| Extracted `.mmd` | — | **not retained**; deleted with the temp dir |
| Chroma vectors | Disk | **the only copy** — nothing on the volume rebuilds these |
| BM25 pickle | Disk | **the only copy**; also holds the chunk *text*, so re-embedding is still possible |
| Model weights | Disk | on the volume, or ~15 GB re-downloads every wake |
| Telemetry log | Disk | append-only; future gold set + fine-tune data |
| Browser tab | Your Mac | renders the token stream, writes nothing |

Back up `chroma_db/` and `bm25_indexes/`. That advice is the inverse of what it used to
be, and the reason matters: uploaded documents are no longer kept, so the indexes are no
longer derived data — they are the only copy the pod holds. Recovery depends on the family
still having their own PDFs, which is an assumption rather than a guarantee.

Re-*embedding* survives this: `build_bm25` pickles the chunk text alongside the index, so
a different embedding model can be applied offline. Re-*chunking* does not — the chunk
boundaries are frozen at whatever the upload produced, and changing them needs the document
uploaded again.

---

## 6. What this means for local storage

Measured on the development Mac, 2026-08-17:

| | |
|---|---|
| HuggingFace model cache | 37 GB |
| Ollama models | 8.1 GB |
| Python venv | 2.1 GB |
| Chroma vectors | 4.2 MB |
| Extracted docs + PDFs | 1.0 MB |
| BM25 index | 0.4 MB |
| **Total** | **~47 GB** |

**The documents are 5.6 MB. The models are 45 GB.** Moving the corpus off the laptop
reclaims about as much space as a few photos; moving the *models* reclaims the drive.
Both happen in the same step, so the deployment plan is unchanged — but the payoff
arrives when the pipeline stops running locally at all, not when the indexes relocate.

Keep the local copy as long as you are actively developing against it. The 60 GB volume
is sized to hold the whole thing (`DEPLOYMENT.md §2`).

---

## 7. When these tiers become separate machines

Not yet. Today one pod holds all three.

Because retrieval runs fine on CPU and generation is ~90% of query time, the natural next
architecture is a cheap always-on CPU host running BM25 + Chroma + RRF, with the GPU pod
holding only Ollama and Marker. That is the moment the CPU and GPU rows above become two
hosts with a network hop between them.

`DEPLOYMENT.md §1` explains why it is deferred: it adds a second always-on bill to a
budget that already works. **The FastAPI boundary is what keeps it cheap to do later** —
retrieval moves, and no client changes.
