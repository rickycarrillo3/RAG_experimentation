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
                    │    ├─ kbm/retrieval.py  (BM25 + Chroma + reranker)     │
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

Deployment knobs shared with the CLI and the Gradio client live in **`kbm/config.py`**
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
| `KBM_LLM_MODEL` | The generator's Ollama tag. Selects a profile in `kbm/llm_profiles.py` that supplies the window size, decode budget and whether the Python sandbox is enabled — so naming a model configures it. Default: `t1c/deepseek-math-7b-rl:Q4`. |
| `KBM_NUM_PREDICT`, `KBM_KEEP_ALIVE` | Decode cap, and how long Ollama holds the weights in VRAM. Keep `KEEP_ALIVE` **≥** the idle-stop window — see §5. `NUM_PREDICT` now defaults per model (350 for deepseek, 1024 for a TIR model, which needs room to reason, write a program, and reason again). |
| `KBM_NUM_CTX` | Context window sent to Ollama. Defaults to the model's real window. **Do not raise it above what the model was trained for** — Ollama will not refuse, it will degrade. Ollama also shifts an overflowing context from the *left*, which eats the system prompt first and silently. |
| `KBM_TIR` | `1`/`0` forces tool-integrated reasoning on or off, overriding the model's profile. On, the generator may write Python and have it executed between passes (§8). |
| `KBM_TOOLS` | `1`/`0` forces **native tool calling** on or off, overriding the model's profile. On, the generator gets `search_documents`, `run_python` and `list_documents` as JSON tools — it can compute, and it can search the family's documents again mid-answer (`AGENT.md`, §8). **Takes precedence over `KBM_TIR`**, which is forced off whenever this is on: a model must never be handed both protocols. Forced onto a model with no tools template, the server logs an error and downgrades rather than failing every answer. |
| `KBM_SANDBOX_TIMEOUT` | Wall-clock seconds for one sandboxed program. Default `10`. |
| `KBM_SHOW_TOOL_CODE` | `1` shows the student the raw ```python block instead of just the computed result. Off by default — the same escape hatch as `KBM_SHOW_DIAGNOSTICS`, for the same reason. |
| `KBM_MAX_CONTINUATIONS` | How many times the server may resume an answer cut off at `KBM_NUM_PREDICT`. `0` = label it truncated instead. |
| `KBM_SHOW_DIAGNOSTICS` | `1` appends the technical cause to upload status messages. Off by default — the family sees plain sentences, the log keeps the exception. |
| `KBM_ENABLE_DOCS` | `1` serves `/docs`, `/redoc`, `/openapi.json`. Defaults **on** with no `KBM_API_TOKEN` (laptop) and **off** once one is set — those routes belong to the app, not the router, so the token never covered them. |
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

`/healthz` sits **behind `KBM_API_TOKEN`** like every other endpoint: every caller today
(`startup.sh`, the Gradio client, you over SSH) already holds the token, and port 8000 is
not publicly exposed, so an unauthenticated probe would buy nothing. Note the consequence
for the wake page in §6 — it would need a credential to poll readiness, which is an
argument for giving that page its own narrow endpoint rather than for opening this one.
Poll the **status code**, not merely whether the request completed: a 401 and a 500 are
both "not ready", and treating any response as success is how a broken API reads as
healthy.

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
  are — `/upload` streams to disk without a size check. Since 2026-08-19 the PDF itself is
  not retained (§7), so the volume grows by index size only: ~450 MB per 1000-page
  textbook rather than ~550 MB. That is a smaller slope, not a bound. A family member
  uploading a shelf of textbooks still fills the volume, and a full network volume fails
  writes rather than auto-expanding. Watch `du -sh $DATA_DIR`; a per-user cap is the
  obvious next guard.
- **No per-user auth** (§4).
- **No unattended wake path** (§5). Sleep is automatic; wake requires the RunPod account
  credential, so a family member who opens the link on a stopped pod gets a dead URL and
  no explanation. This is the largest remaining gap in the day-to-day experience.
- **`ops/idle_stop.py` has never run against a live pod.** It is the difference between
  ~$20/mo and ~$115/mo and carries zero evidence. Verify with
  `KBM_IDLE_STOP_MINUTES=2` before relying on it — a silent failure here surfaces on the
  invoice, not in a log.
- **No backups, and now nothing to rebuild from.** Uploaded PDFs are no longer kept, so
  the BM25 pickle and Chroma directory are not derived data any more — they are the only
  copy on the pod. Back *those* up. Losing them means asking the family to re-upload every
  document and paying for Marker again.
- **`KBM_RELEVANCE_FLOOR` is calibrated but on a thin sample.** 0.15, set from
  `evaluation/calibrate_floor.py`; see `api/settings.py` for the measurements. The
  deployment-shaped half of that calibration is 19 on-topic and 18 off-topic questions
  against a single chapter — wide separation, small n. Re-run it against real logged
  questions once telemetry has collected them (`EVALUATION.md §10.9`), and re-run it
  after any change of reranker, since the number is a property of that model's output
  scale and does not survive swapping it.

---

## 7. What an upload actually costs you

`POST /upload` → Marker extraction → chunk → BM25 + Chroma. What persists per document,
under `$DATA_DIR`:

| Artifact | Path | Size (measured) |
|---|---|---|
| Source PDF | *(not retained)* | temp dir only, deleted when the job ends |
| Extracted `.mmd` | *(not retained)* | temp dir only, deleted when the job ends |
| Chroma vectors | `chroma_db/` | ~64 KB per chunk with `bge-small` (384-dim) |
| BM25 pickle | `bm25_indexes/user_<u>.pkl` | ~5.8 KB per chunk |

So a 1000-page textbook ≈ 6–7k chunks ≈ **~450 MB of index**, with the PDF costing nothing
after the job ends. Five of them is roughly 2.3 GB — small next to the ~25 GB of model
weights.

**The documents themselves are not kept.** The PDF and its `.mmd` live in the request's
temp directory and are removed when the job finishes, successfully or not. What this buys
is that the pod holds no copy of the family's library; what it costs is that the indexes
become irreplaceable and a chunking change cannot be evaluated against real user documents
without a re-upload. `ARCHITECTURE.md §5` carries the full trade.

Two behaviours worth knowing:

**Re-uploading the same document replaces it, rather than duplicating it.** Chunks are
keyed by `chunk_id` (`<source>::<n>`) and merged with `ingest.merge_chunks`, so the
second upload of `calculus.pdf` overwrites its chunks in place. This still holds now that
extraction happens in a temp directory, and the reason is worth knowing before anyone
refactors that path: `assign_chunk_ids` takes `os.path.basename`, and `mkdtemp()`
randomises the *directory* while the uploaded *file name* is preserved. Randomise the file
name — a uuid prefix, `NamedTemporaryFile` — and every re-upload becomes a duplicate. Before this was fixed,
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

---

## 8. The Python sandbox — what it is, and what it is not

With `KBM_TIR` **or `KBM_TOOLS`** on, the generator writes Python and the server executes
it. Two protocols reach the same sandbox — `kbm/tools/tir.py`'s text blocks and `kbm/tools/agent.py`'s native
`run_python` tool — and `kbm/config.py` guarantees a model gets exactly one of them, never
both. That is the point — `evaluation/EVALUATION.md §11.6`
traced the model's errors to arithmetic *execution*, and a model that can call Python does
not compute `2401 mod 13` as 3. It also means **a language model now decides what code
this pod runs**, so be exact about what protects it.

**What is in place.** Three independent layers, in order:

1. An **AST allow-list**, checked before anything runs. Imports are whitelisted (`math`,
   `sympy`, `numpy`, `fractions`, … — the list is short on purpose); `open`, `eval`,
   `exec`, `compile`, `__import__` and any dunder attribute are rejected. Blocking dunder
   access closes the `().__class__.__bases__[0].__subclasses__()` route back to arbitrary
   code without having to enumerate the escapes.
2. A **separate short-lived process** — never a thread — in its own session, its own empty
   temp cwd, and a **scrubbed environment**. The pod's real environment carries HF tokens,
   a `RUNPOD_API_KEY` with the power to stop the pod, and the paths to the family's
   indexes; none of it is visible to generated code.
3. **rlimits** (CPU, address space, file size) plus a wall-clock kill of the whole process
   group, so a program that ignores signals still burns a bounded number of CPU seconds.

**What is not in place.** There is no namespace, no seccomp filter, no container, and
**no network isolation** — outbound sockets are prevented by the import allow-list, which
is a policy check and not a kernel boundary. This is a gate, not a jail.

**The threat it is actually sized for** is a 7B math model emitting a runaway loop or a
typo. There is a second, narrower path worth naming rather than leaving implied: the model
writes code after reading text the family **uploaded**, so a PDF containing instructions
aimed at the model is a prompt-injection route to the sandbox. The layers above bound what
that could achieve on a single-tenant family pod. They would not be sufficient if this were
ever exposed to untrusted users — at that point the execution belongs in a container with
no network, not in this process.

**Agent mode shortens that path, and adds a second thing worth protecting.** With
`KBM_TOOLS` on the model can call `search_documents`, so within a single turn it can read
uploaded text, choose what to search for next, and choose what to execute — the retrieval
and the execution are no longer separated by a request boundary. Two consequences:

- The sandbox's exposure is unchanged in *kind* but reached more readily. The three layers
  above apply identically to both protocols; there is one gate, not two.
- **The second asset is now the other family members' indexes.** `search_documents` takes a
  `query` and nothing else: `user` is closed over from the HTTP request and is not a tool
  parameter, so a poisoned PDF cannot instruct the model to read someone else's documents —
  there is no argument in which to say so. This is the control, it is structural rather than
  a validation check, and `agent.TOOL_SCHEMAS` must never gain a `user` field. Isolation
  here is directory naming, not auth (`CLAUDE.md`), so nothing downstream would catch it.

`KBM_TIR=0` **and `KBM_TOOLS=0`** turns the whole thing off and costs only the accuracy the
tools were buying. `GET /healthz` reports which arm is actually live as `protocol`.
