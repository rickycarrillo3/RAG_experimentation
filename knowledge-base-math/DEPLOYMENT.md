# Deployment

How this runs somewhere other than a laptop, and what it costs.

Target: reachable by 1–5 family members, 2–4 hrs/day, **under $30/month**, on free/open
models throughout.

---

## 1. Shape

```
                    ┌─ RunPod pod (on-demand, stopped when idle) ────────┐
  browser ──HTTPS──▶│  FastAPI (uvicorn)  api/main.py                    │
                    │    /chat  SSE stream   /upload   /jobs   /feedback │
                    │    ├─ retrieval.py  (BM25 + Chroma + reranker)     │
                    │    ├─ extract.py    (Marker, GPU)                  │
                    │    └─ Ollama        (generator)                    │
                    │  idle watchdog ─── no /chat for N min ─▶ stop self │
                    └─ /workspace network volume ───────────────────────┘
                         docs/  chroma_db/  bm25_indexes/  telemetry/
                         ollama-models/  .cache/huggingface/
```

The frontend is currently `app.py` (Gradio), talking to the API over HTTP like any
other client. A TypeScript frontend replaces it against the same endpoints.

### Why one pod

Retrieval (BM25 + bge-small + the cross-encoder) runs perfectly well on CPU — a few
hundred milliseconds, versus the ~95% of query time that generation takes
(`LATENCY.md`). So the *natural* split is a cheap always-on CPU host for retrieval plus
an on-demand GPU for generation.

We are not doing that yet, because it adds a second always-on bill to a budget that
already works, and because a cold start is acceptable here. **The FastAPI boundary is
what makes the split cheap to do later**: when it's worth it, retrieval moves to a small
CPU host and the pod keeps only Ollama + Marker, with no change to any client.

---

## 2. Cost

| Line | Rate | Monthly |
|---|---|---|
| RTX A5000 24GB, **community** cloud | $0.16/hr × 3.5 hr/day × 30 | **$16.80** |
| Network volume, 50GB | $0.07/GB/mo | **$3.50** |
| | | **≈ $20/mo** |

Three things are load-bearing:

- **The pod must actually stop.** An always-on 24GB card is ~$115/mo. `ops/idle_stop.py`
  is what keeps this honest; without it there is no budget, only an intention.
- **Card choice.** RTX 4090 community is $0.34/hr → $35.70/mo on compute alone, over
  budget before storage. The A5000's 24GB covers the ~8GB steady-state query footprint
  (`evaluation/EVALUATION.md §6`) with room for a Q8 generator, so the cheap card is
  also the sufficient one. **If the generator benchmark picks a model needing >24GB,
  this table has to be redone** — treat that as an exit criterion of the benchmark, not
  a detail.
- **Network volume, not container disk.** RunPod bills container/volume disk at
  **$0.20/GB/mo while the pod is stopped**, versus $0.07/GB/mo for a network volume.
  50GB of model weights on container disk would cost more sitting idle than the GPU
  costs running.

---

## 3. Environment

Deployment knobs shared with the CLI and the Gradio client live in **`config.py`**
(`DATA_DIR`, `CHROMA_DIR`, `BM25_DIR`, `OLLAMA_BASE_URL`, `APP_HOST`, `APP_PORT`,
`APP_AUTH`, `REQUIRE_GPU`). `api/settings.py` holds only what is specific to the HTTP
service. Do not add a second name for the same thing in both — see `SETUP.md §3`.

| Variable | Purpose |
|---|---|
| `KBM_API_TOKEN` | **Required in any deployment.** Bearer secret for the API; fully open without it. |
| `APP_AUTH` | **Required in any deployment.** `user:pass` pairs gating the Gradio login page. A *different* lock from `KBM_API_TOKEN` — see §4. |
| `DATA_DIR` | Root for `chroma_db/`, `bm25_indexes/`, `telemetry/`. Set to `/workspace/kb-data` on the pod. |
| `REQUIRE_GPU` | `1` turns a silent CPU fallback — which looks exactly like success, only 10–40× slower — into a startup error. |
| `KBM_RELEVANCE_FLOOR` | Cross-encoder score below which we answer in `general` mode. Sigmoid scale (0–1). |
| `KBM_IDLE_STOP_MINUTES` | Minutes idle before the pod stops itself. `0` disables. |
| `RUNPOD_API_KEY`, `RUNPOD_POD_ID` | Needed for idle-stop to work; without them it warns and does nothing. |
| `KBM_NUM_PREDICT`, `KBM_KEEP_ALIVE` | Decode cap and Ollama residency. Keep `KEEP_ALIVE` shorter than the idle-stop window. |
| `KBM_TELEMETRY_SALT` | Salt for hashing usernames in the event log. Set it per deployment. |
| `OLLAMA_MODELS`, `HF_HOME` | Must point at the volume, or ~15GB of weights re-download on every wake (`SETUP.md`). |

Run:

```bash
cd knowledge-base-math
uvicorn api.main:app --host 0.0.0.0 --port 8000
python app.py                      # optional Gradio client, port 7860
```

---

## 4. Auth — read this before exposing a port

Per-user isolation is a **lowercased username string**, with no verification
(`CLAUDE.md`). It separates one family member's documents from another's; it does not
authenticate anyone. On a public URL with no token, anyone who finds the host can read
every uploaded document by guessing a name.

There are **two locks on two different doors**, and both must be set:

- **`APP_AUTH`** gates the Gradio login page (`config.app_auth()`).
- **`KBM_API_TOKEN`** gates the API itself, on port 8000.

Setting only `APP_AUTH` protects the page while leaving the API that serves it wide
open on another port — anyone hitting `:8000` directly bypasses the login entirely.
Setting only `KBM_API_TOKEN` leaves the UI open to anyone, though it could not talk to
the API. Neither is multi-user auth: they are the difference between "private" and
"crawlable". Real per-user auth is separate work, needed before this is shared beyond
people who already trust each other with one password.

Also tighten `allow_origins` in `api/main.py` before going public: a wildcard plus a
bearer token means any page a family member visits can spend that token.

---

## 5. Wake / sleep

**Sleep** is automatic via `ops/idle_stop.py`: no `/chat` for `KBM_IDLE_STOP_MINUTES`
and the pod calls RunPod's REST API to stop *itself*. It defers while an ingest job is
running — stopping mid-Marker would lose the work and leave a half-built index. Note it
stops rather than terminates, so the volume and its model weights survive.

**Wake** is manual (or scriptable):

```bash
curl -X POST https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/start \
     -H "Authorization: Bearer $RUNPOD_API_KEY"
```

The cold path is roughly: pod start ~30s → Ollama loads the model ~10–20s from the
volume → first token. `GET /healthz` reports `model_loaded`, which is the flag a client
should poll — after a cold start the API answers HTTP 200 well before the model is
resident, so a client that only checks for 200 will fire its first question into a model
load and look broken.

---

## 6. Known gaps

- **Documents still land wherever  `DATA_DIR` points.** On the pod that's the network
  volume, which solves the "not on my laptop" problem. It does **not** solve durability:
  a deleted pod volume is a deleted corpus. Moving `chroma_db/` and the uploaded PDFs to
  a managed store (S3-compatible object storage, or a hosted vector DB) is the next step
  — it also decouples retrieval from the GPU pod entirely, which is the same refactor
  as the CPU/GPU split in §1. Worth doing once the corpus is real textbooks rather than
  test files.
- **No per-user auth** (§4).
- **No backups.** The BM25 pickle and Chroma directory are rebuildable from the source
  PDFs, so back up `docs/raw/` first.
- **`KBM_RELEVANCE_FLOOR` is uncalibrated.** See `api/settings.py`; the `no_answer` slice
  of gold set v3 is what settles it.
