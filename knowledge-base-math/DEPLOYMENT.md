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

For which stage runs on which piece of hardware — and why only four things ever touch
the GPU — see **`ARCHITECTURE.md`**.

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
| `KBM_NUM_PREDICT`, `KBM_KEEP_ALIVE` | Decode cap, and how long Ollama holds the weights in VRAM. Keep `KEEP_ALIVE` **≥** the idle-stop window — see §5. |
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

### Which ports to expose

**HTTP: `7860` only.** `startup.sh` sets `KBM_API_URL=http://127.0.0.1:8000`, so the
Gradio client reaches the API over loopback *inside* the container — nothing outside the
pod needs to route to 8000. Exposing it adds a second public door guarded by one
credential, where 7860 has `APP_AUTH` in front and the token behind it. Add TCP `22` only
if you want `scp` for the eval corpus.

**Never expose `11434`.** Ollama's API is unauthenticated: a public port there lets anyone
run inference on the GPU you are paying for.

When the TypeScript frontend needs 8000 reachable, two things change together — expose the
port, *and* add the proxy origin to `allow_origins` in `api/main.py`. It is currently
`["http://localhost:5173", "http://localhost:3000", "http://localhost:7860"]`, so a browser
calling from `*.proxy.runpod.net` is blocked by CORS until that list is updated. Keep it a
list; a wildcard plus a bearer token means any page a family member visits can spend that
token.

---

## 5. Wake / sleep

**Sleep** is automatic via `ops/idle_stop.py`: no `/chat` for `KBM_IDLE_STOP_MINUTES`
and the pod calls RunPod's REST API to stop *itself*. It defers while an ingest job is
running — stopping mid-Marker would lose the work and leave a half-built index. Note it
stops rather than terminates, so the volume and its model weights survive.

### keep-alive vs idle-stop — two timers, one of them free

They sit at different layers and are easy to conflate:

| | `KBM_KEEP_ALIVE` | `KBM_IDLE_STOP_MINUTES` |
|---|---|---|
| Controls | Ollama holding weights in VRAM | whether the pod exists |
| On expiry | model unloads; next query reloads it (~10-20s) | **pod stops; billing stops** |
| Affects the bill | no — the pod bills regardless | **yes, entirely** |

**Idle-stop is the outer bound.** The model cannot stay resident longer than the pod
lives, so a keep-alive longer than the idle window is simply never reached — harmless.

Setting keep-alive *shorter* than the idle window is the real mistake. With keep-alive
5m and idle-stop 10m, the model unloads at minute 5 while the pod keeps billing until
minute 10; a question at minute 6 pays a reload for no saving at all. **Keep it ≥ the
idle window** — the `30m` default is fine with idle-stop at 10.

The one reason to lower it is VRAM: upload peaks at ~12-13GB with the generator resident
(`evaluation/EVALUATION.md §6`), so drop it toward 0 during ingest if it ever OOMs.

### Wake is two things, and neither is automatic

Sleep is one action; wake is two, and it is easy to plan for only the first:

1. **The pod must be started.** A stopped pod has no container and no network — the
   `proxy.runpod.net` URL fails outright. Opening the link does not wake anything.
2. **The processes must be started.** A pod start recreates the container from the image,
   so nothing is running: no Ollama, no uvicorn, no Gradio. `/workspace` survives; running
   processes do not. `startup.sh` is what brings them back.

Starting the pod:

```bash
curl -X POST https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/start \
     -H "Authorization: Bearer $RUNPOD_API_KEY"
```

### Automate step 2 with the container start command

RunPod's **Container Start Command** runs on every pod start. Point it at the script and
a wake brings the app up with no terminal involved:

```
bash -lc 'cd /workspace/RAG_experimentation/knowledge-base-math && bash startup.sh --no-prefetch'
```

Output lands in the pod's log tab. Use `--no-prefetch` on this path: `prefetch_models.py`
loads the embedder and reranker to verify them, and the API loads them again thirty
seconds later — on a warm volume that is a duplicated ~30 s on every single wake. Keep the
full `bash startup.sh` for manual runs, where a cold cache or a fresh `git pull` makes the
check worth paying for.

### What the cold path actually costs

| Step | Cost |
|---|---|
| Pod start (container created, volume mounted) | ~30 s |
| `startup.sh`: torch import + CUDA check, Ollama start | ~10–20 s |
| API startup: embedder + 2.2 GB reranker loaded | ~30 s |
| First query: Ollama loads the generator from the volume | ~10–20 s |
| **Click → first token** | **~1–2 min** |

⚠️ **This is an estimate assembled from component measurements, not an end-to-end
timing.** The figure the deployment was planned against was "30–60 s", which counted only
pod start plus the model load and missed the two model-loading stages in between. Time a
real wake and replace this table with the measured number — and if it lands materially
above 2 minutes, that is a finding, not a detail.

`GET /healthz` reports `model_loaded`, which is the flag a client should poll — after a
cold start the API answers HTTP 200 well before the model is resident, so a client that
only checks for 200 fires its first question into a model load and looks broken.

### The wake gap

Step 1 needs `RUNPOD_API_KEY`, which controls the whole RunPod account — start, stop,
terminate, spend. It cannot go in a bookmark handed to a family member, so today waking
the pod is something **you** do on request. Two ways out, neither built:

- A small always-on free-tier function holding the key and exposing a single "wake"
  button, with no other capability.
- A wake-on-request page that starts the pod and then polls `/healthz` until
  `model_loaded`, so the 1–2 minutes reads as a progress bar rather than a broken link.

Worth building when "text me and I'll turn it on" gets annoying — not before.

---

## 6. Known gaps

- **Documents still land wherever `DATA_DIR` points.** On the pod that's the network
  volume, which solves the "not on my laptop" problem. It does **not** solve durability:
  a deleted volume is a deleted corpus, and RunPod may terminate a volume whose storage
  charges go unpaid. Moving `chroma_db/` and the uploaded PDFs to a managed store
  (S3-compatible object storage, or a hosted vector DB) is the next step — it also
  decouples retrieval from the GPU pod entirely, which is the same refactor as the
  CPU/GPU split in §1. Worth doing once the corpus is real textbooks rather than test
  files.
- **No quota on uploads.** Nothing limits how many PDFs a user can add or how large they
  are. `/upload` streams to disk without a size check, and each accepted PDF is kept
  under `$DATA_DIR/docs/raw/<user>/` alongside its `.mmd`. That is deliberate (§7), but
  it means a family member uploading a shelf of textbooks grows the volume until it is
  full — and a full network volume fails writes rather than auto-expanding. Watch
  `du -sh $DATA_DIR` for now; a per-user cap is the obvious next guard.
- **No per-user auth** (§4).
- **No unattended wake path** (§5). Sleep is automatic; wake requires the RunPod account
  credential, so a family member who opens the link on a stopped pod gets a dead URL and
  no explanation. This is the largest remaining gap in the day-to-day experience.
- **`ops/idle_stop.py` has never run against a live pod.** It is the difference between
  ~$20/mo and ~$115/mo and carries zero evidence. Verify with
  `KBM_IDLE_STOP_MINUTES=2` before relying on it — a silent failure here surfaces on the
  invoice, not in a log.
- **No backups.** The BM25 pickle and Chroma directory are rebuildable from the source
  PDFs, so back up `docs/raw/` first.
- **`KBM_RELEVANCE_FLOOR` is uncalibrated.** See `api/settings.py`; the `no_answer` slice
  of gold set v3 is what settles it.

---

## 7. What an upload actually costs you

`POST /upload` → Marker extraction → chunk → BM25 + Chroma. What persists per document,
under `$DATA_DIR`:

| Artifact | Path | Size (measured) |
|---|---|---|
| Source PDF | `docs/raw/<user>/` | as uploaded — an OpenStax textbook is ~50–100 MB |
| Extracted `.mmd` | `docs/extracted/<user>/` | ~1.8 KB per page (18 KB for 10 pages) |
| Chroma vectors | `chroma_db/` | ~64 KB per chunk with `bge-small` (384-dim) |
| BM25 pickle | `bm25_indexes/user_<u>.pkl` | ~5.8 KB per chunk |

So a 1000-page textbook ≈ 6–7k chunks ≈ **~450 MB of index + ~100 MB of PDF**. Five of
them is roughly 3 GB — small next to the ~25 GB of model weights.

Two behaviours worth knowing:

**Re-uploading the same document replaces it, rather than duplicating it.** Chunks are
keyed by `chunk_id` (`<source>::<n>`) and merged with `ingest.merge_chunks`, so the
second upload of `calculus.pdf` overwrites its chunks in place. Before this was fixed,
each upload re-added the user's *entire* accumulated corpus to Chroma — 66 → 136 → 210
entries for uploads of 66, 4 and 4 chunks. Quadratic growth, and it degraded retrieval:
the dense top-k filled with copies of one chunk, so fewer distinct candidates reached the
reranker than `TOP_K` implied.

**Filenames are the identity.** Two different documents both named `notes.pdf`, uploaded
by the same user, collide — the second silently replaces the first, because chunk ids are
derived from the basename. Uploading under distinct filenames avoids it; a content hash
in the chunk id would fix it properly.

**Ingest is serialised** (`_ingest_pool`, one worker). Marker peaks at ~12–13 GB VRAM on
top of a resident generator and reranker (`evaluation/EVALUATION.md §6`), so two
concurrent extractions would race for memory on a 24 GB card. Several uploads at once
queue rather than fail; `GET /jobs/{id}` reports each one's position by status.
