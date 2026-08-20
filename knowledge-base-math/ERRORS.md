# Errors and fixes

A log of failures this project actually hit, what caused them, and what fixed them.

**Why keep it.** Most of these looked like something other than what they were — a wrong
CUDA build that reports "no GPU", a merge that passes review and breaks every entry point,
an upload path that silently triples the index. The cause is rarely near the symptom, and
the second person to hit one of these (including a future agent) should not have to
re-derive it.

**Format.** Newest first. Each entry: what you see, what it actually is, the fix, and —
where there is one — the general lesson. If a fix lives in another doc, link there rather
than duplicating it.

**Add to this file when a bug takes more than a few minutes to diagnose**, especially when
the symptom pointed somewhere misleading. Routine typos don't belong here.

---

## 2026-08-19 · `pip install` hung for minutes, blaming `sentence-transformers`

**Symptom.** On a fresh pod, `pip install -r requirements.txt` printed a wall of
`Downloading sentence_transformers-X.Y.Z...metadata` lines, walking backwards from 6.0.0
through 5.7.0, 5.6.1, 5.5.1, … 5.1.1, with pip's own advice to "provide the dependency
resolver with stricter constraints". It reads as a slow network or a
`sentence-transformers` problem. It is neither.

**What it actually was.** Two pins in `requirements.txt` had become unsatisfiable
together, through a chain neither one mentions:

```
marker-pdf<2.0  ->  marker-pdf 1.10.2  ->  transformers<5.0.0  ->  huggingface-hub<1.0
gradio==6.18.0                                                 ->  huggingface-hub>=1.2.0
```

Gradio raised its `huggingface-hub` floor from `>=0.33.5` to `>=1.2.0` in **6.18.0** — the
exact version that had been pinned to fix an unrelated `Blocks`/`launch()` crash. Nothing
can satisfy `hub<1.0` and `hub>=1.2.0` at once, so pip backtracks looking for an escape,
and `sentence-transformers` is simply the package it chose to pivot on. It has ~30
releases and its own `huggingface-hub` floor (6.0.0 needs `>=1.3.0`), so it looks like the
constrained package while contributing nothing to the conflict.

The `marker-pdf<2.0` pin is not the one to loosen: it exists because marker-pdf 2.0 needs
a Docker daemon a RunPod pod does not have (see the 2026-08-18 entry below).

**Fix.** Pin Gradio to the last release that still accepts `hub>=0.33.5`, and pin
`sentence-transformers` so the resolver stops exploring at all:

```
gradio==6.17.3
sentence-transformers==5.5.1
```

Still Gradio 6, so `app.py`'s Gradio-6 API usage is unaffected. To unblock an install
without editing the file, append the pins on the command line:

```bash
pip install -r requirements.txt gradio==6.17.3 sentence-transformers==5.5.1
```

**Lesson.** When pip backtracks, the package it names is almost never the cause — it is
whichever package has the most versions to walk. Read the *constraints*, not the log:
`pip index versions` and each candidate's `requires_dist` find the real pair in a minute,
while watching the download log finds nothing. And a version pin added to fix an API
crash is still a dependency edge; this one silently re-opened a conflict on a transitive
package neither pin names.

---

## 2026-08-19 · Long answers stopped mid-sentence, and the API called it a success

**Symptom.** Answers to derivation-style questions ended abruptly, mid-word or mid-step.
Nothing in the UI, the `done` frame, or the telemetry log distinguished them from answers
that had finished.

**Cause.** `KBM_NUM_PREDICT` caps decode at 350 tokens, and deepseek-math-7b-rl is a
chain-of-thought solver that fills whatever budget it is given — `LATENCY.md` records it
emitting 534 tokens on a *one-line conceptual* question. Ollama reports this honestly as
`done_reason: "length"` (versus `"stop"`). Nothing in this repo had ever read that field.

**Why it was invisible.** `api/routes.py` built the chain as
`prompt | models.llm | StrOutputParser()`. `StrOutputParser` maps each `AIMessageChunk` to
its `.content` and drops `response_metadata` — which is exactly where `done_reason` lives.
The convenience parser threw away the failure signal along with the metadata, three layers
before anything could act on it: `DoneEvent` had no field for it and `app.py` read only
`event_id` from the `done` frame.

**Fix.** Stream `AIMessageChunk`s directly, and when `done_reason == "length"`, resume by
sending the partial answer back as an `AIMessage` and continuing — bounded by
`KBM_MAX_CONTINUATIONS` (default 2). If it is still cut off after that, the server appends
`TRUNCATION_MARKER` itself, the same way it already prepends `GENERAL_MODE_MARKER`, and
for the same measured reason: the model ignores instructions to emit such lines.

**The trap inside the fix.** Ollama treats a trailing assistant message as a *prefill* —
which is what makes seamless continuation work at all — but it **echoes the entire prefill
back at the head of the continuation stream** before emitting anything new. Forwarded
unfiltered, the student sees the answer twice. `chat.PrefillEcho` strips it, and on any
mismatch emits nothing and abandons the continuation, degrading to a shorter honest answer.
Verified on ollama 0.32.6: pass 0 emitted 147 chars; pass 1 sent 224 and emitted 77 (147
stripped); pass 2 sent 350 and emitted 126 (224 stripped). Exact, every time.

**Latency.** The continuation appends at the *end* of the prompt, so `LATENCY.md`'s prefix
rule protects the KV cache. Measured on a 1821-token prompt: 7358 ms cold, **47 ms** on an
identical repeat, **160 ms** with the partial answer appended.

**Lesson.** A convenience wrapper that flattens a rich object to a primitive throws away
the failure signal along with the metadata. And when a cap exists, something must report
whether it was hit — `telemetry.log_query` now records `truncated`/`continuations`, because
"raise it if answers are visibly truncated" requires someone to be able to see it.

---

## 2026-08-19 · An ingest that destroyed every equation reported `done` and a chunk count

**Symptom.** Uploading a PDF returned `"Indexed. 412 total chunks for alice."` The index
contained no LaTeX at all and retrieved math badly.

**Cause.** `extract()` returned a bare `str` path on *both* the Marker path and the
`pymupdf4llm` fallback. The fallback only `print`ed a WARNING to the server's stdout and
fell through, so no exception was raised, `_run_ingest` reached its success branch, and
`app.py` echoed `job.detail` verbatim. A return type that cannot express degradation
guarantees the caller will report success.

**Also found here.** `_ingest_pool.submit(...)`'s `Future` was discarded, so anything
escaping `_run_ingest` itself — a `KeyError` on `_jobs[job_id]`, an `ImportError` from the
function-local imports, an OOM-killed thread — left the job `queued` forever with a client
polling a status that would never change. And the single broad `except` recorded a bare
`str(e)` with no stage and no traceback.

**Fix.** `extract_detailed()` returns an `ExtractResult(path, extractor, marker_error)`
with a `.degraded` property (mirroring `retrieve`/`retrieve_detailed` in `retrieval.py`);
`extract()` keeps its `-> str` signature for the CLI. `Job` gained `extractor`, `degraded`
and `stage`; the success message now leads with the bad news. `add_done_callback` marks
crashed jobs failed, and the `except` logs a traceback with the stage.

**Why `degraded` is a field and not a fifth `JobStatus`.** `api/schemas.py` is what the
TypeScript client is generated from. Adding an enum member breaks an exhaustive switch;
adding an optional field is ignored by clients that predate it. So a half-good ingest is
`status="done"` **plus** `degraded=true`, never a new status value.

**Follow-up, same day.** The first fix over-corrected: it put the raw Marker exception —
`docker binary not found. Install Docker (https://docs.docker.com/get-docker/)…` — straight
into the status box that tells a family member their upload worked. Correct information,
wrong audience. `Job` now carries **two** strings: `detail`, a plain sentence for whoever
uploaded the file, and `diagnostic`, the exception text for the log and for operators
(`KBM_SHOW_DIAGNOSTICS=1` surfaces it in the UI). The client-side messages were leaking too
— a raw FastAPI error body from `Upload rejected: {r.text}`, a bare `httpx` exception, and
an internal `GET /jobs/{id}` instruction.

**Lesson.** The same one the marker-pdf entry below reaches from the other direction: a
fallback that "works" is more dangerous than a crash. Make the degradation part of the
return value, or every layer above will faithfully report success. And the corollary the
follow-up taught: "surface the error" and "show the user the exception" are not the same
instruction. One string cannot serve both a family member and an operator — an install URL
in a success message is noise to everyone who cannot act on it.

---

## 2026-08-19 · Uploaded PDFs accumulated on the volume, and ingest took two clicks

**Symptom.** Not a crash — a policy problem. Every upload kept the source PDF (~50–100 MB
for a textbook) and its `.mmd` under `$DATA_DIR/docs/{raw,extracted}/<user>/`, with no
quota in front of them, and ingestion required selecting a file *and* clicking a button.

**Change.** User documents are no longer retained: extraction goes to the request's temp
dir and dies with it in the existing `finally`. The Gradio upload now fires on file
selection. This deliberately reverses the earlier decision recorded in the "Earlier" table
below — see the comment at the top of `_run_ingest` for the consequences accepted with it
(the indexes become the only copy; re-chunking needs a re-upload; re-*embedding* still
works, because `build_bm25` pickles the chunk text).

**The invariant that had to be checked first.** Chunk ids are `<basename>::<n>`
(`chunking.assign_chunk_ids`), and `merge_chunks` dedupes on them — that is what makes
re-uploading a document *replace* its chunks instead of duplicating them, a bug this
project already hit once (66 → 136 → 210 chunks). Moving extraction into a random temp
directory is safe **only** because `assign_chunk_ids` takes `os.path.basename`, and
`mkdtemp()` randomises the *directory* while the *file* keeps the uploaded name. Verified:
identical ids across two temp dirs, and `merge_chunks(a, b)` of 5 + 5 → 5.

> Never switch to `NamedTemporaryFile`, a uuid'd filename, or `<uuid>_<name>.pdf`. The
> randomness must live in the directory, never the file name, or every re-upload silently
> becomes a duplicate.

**Also fixed.** `metadata["source"]` was being persisted as a dead temp path; it is now
normalised to the basename (id-neutral — basename of a basename). And auto-firing on
selection needed an idempotence guard plus a `username_box.submit` trigger, or a user who
picked the file before typing their name hit a dead end with nothing to re-trigger, while
every Enter press re-ran Marker.

**Lesson.** Before moving where a file lives, find what derives identity from its path.

---

## 2026-08-18 · `Marker failed (docker binary not found)` on the first upload

**Symptom.** First PDF upload on the pod logged
`WARNING: Marker failed (docker binary not found. Install Docker ... and ensure the
daemon is running.)` and fell back to `pymupdf4llm` — so the document was ingested with
**no LaTeX at all**, which is the one outcome this project cannot tolerate.

**Cause.** `marker-pdf` **2.0.0** (released 2026-07-20) stopped running Surya in-process.
It now spawns an inference server on first use: **vLLM inside Docker** on NVIDIA GPUs,
llama.cpp elsewhere. A RunPod pod is itself a container with no Docker daemon, so the
spawn can never succeed there.

**Why the version changed underneath us.** `requirements.txt` listed a bare `marker-pdf`.
The laptop venv was resolved months ago and holds 1.10.2; a fresh `pip install` on the pod
resolved to 2.0.0. Same file, two different majors — the environments drifted silently.

**Why nothing caught it earlier.** `prefetch_models.py` calls `create_model_dict()`, which
still succeeds on 2.x — the Docker spawn happens at *conversion* time, not at model load.
So stage 4 of `startup.sh` reported success and the failure waited for a real upload,
exactly the stall `prefetch_models.py` exists to prevent, arriving by a different route.

**Fix.** Pin below the rewrite and reinstall on the pod:
```bash
pip install -r requirements.txt          # now marker-pdf<2.0
python -c "import marker; print(marker.__version__)"   # expect 1.10.x
```
Then **re-upload any PDF ingested during the fallback** — a pymupdf-extracted document is
already in the index with its equations flattened to Unicode, and re-uploading replaces
its chunks in place (`DEPLOYMENT.md §7`).

**Lesson.** An unpinned dependency is a promise that upstream will not change its
architecture. Two of the three worst bugs in this file (`torchvision`, this one) are the
laptop and the pod resolving the same requirements file differently. Also: a fallback that
"works" is more dangerous than a crash — this one produced a searchable index that was
quietly worthless for math.

---

## 2026-08-18 · Gradio UI crashed on startup; `/healthz` returned 401

Two unrelated bugs surfaced by the first full `startup.sh` run on the pod.

### `Chatbot.__init__() got an unexpected keyword argument 'type'`

**Symptom.** Stages 1–5 passed, the API came up, then stage 6 died. The UI never
started; the API was left running until the trap killed it.

**Cause.** `requirements.txt` pins `gradio==6.18.0`, but `app.py` was written against
Gradio 5.x. Gradio 6 moved `theme` from `Blocks` to `launch()` and removed `Chatbot`'s
`type` argument. The Mac had 5.x installed, so the file worked there and could not work
on the pod — **the same code could only run on one of the two machines.**

**What made it confusing.** `SETUP.md §7` claimed the pin moved gradio *to 5.50*, and §8
said this exact error meant "environment has 6.x, code targets 5.50" — the precise
inverse of the truth. Following the docs would have led to downgrading a correct
environment.

**Fix.** `app.py` now targets Gradio 6 (theme on `launch()`, no `type=` on `Chatbot`),
matching the pin. `SETUP.md §7`/`§8` corrected. Local venvs on 5.x need
`pip install -r requirements.txt`.

**Lesson.** A version pin is a contract with the code, and a doc that describes a
different pin than the file contains is worse than no doc — it argues against the fix.

### `GET /healthz 401 Unauthorized`

**Symptom.** `startup.sh` logged a 401 for its own readiness probe and then printed
`✓ Ready.` anyway.

**Cause — not the one it looks like.** The 401 was correct: `/healthz` is behind
`KBM_API_TOKEN` like every other endpoint, and the probe was not sending it. The *bug* was
that the probe tested `curl`'s exit status rather than the HTTP status code, so any
response at all counted as ready — a 401, and equally a 500. The check would have passed
against an API that answered nothing but errors.

**The wrong fix, briefly applied.** `/healthz` was first moved to an unauthenticated
router on the reasoning that liveness probes precede credentials. That argument does not
hold here: every caller today (`startup.sh`, the Gradio client, SSH) already has the
token, port 8000 is not publicly exposed, and the only caller that would need an
unauthenticated probe — a family-facing wake page — does not exist and is listed as a
known gap. Reverted.

**Fix.** Probe sends the token and switches on the status code: 200 ready, 401 a distinct
and fatal "token mismatch" message, 000 keep waiting, anything else logged and retried.

**Lesson.** A readiness check that ignores the status code is not a readiness check. And
when a fix requires relaxing a security boundary, check whether the caller that needs it
actually exists before relaxing anything.

---

## 2026-08-17 · `ollama: command not found` on a fresh pod

**Symptom.** `startup.sh` stage 3, or `ollama pull` in setup, failed — Ollama is not in
the RunPod PyTorch image, and `SETUP.md` assumed it was already there.

**The trap, not the error.** The obvious fix is Ollama's own installer:
```bash
curl -fsSL https://ollama.com/install.sh | sh      # ← don't
```
It writes to `/usr/local/bin`, which is **container filesystem** — wiped on every pod
stop. It works, then silently stops existing on the next wake, and `startup.sh` fails
again with no indication that anything was ever installed.

**Fix.** Install to the volume, preserving the tarball's `bin/` + `lib/` layout (the
binary locates its runners relative to itself):
```bash
mkdir -p /workspace/ollama
curl -fL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst \
  | tar --zstd -x -C /workspace/ollama
export PATH=/workspace/ollama/bin:$PATH
```
`startup.sh` now adds `$WORKSPACE/ollama/bin` to `PATH` when it exists, and fails with
these instructions when it doesn't — so no future session needs the export.

**Lesson.** The same rule that governs `DATA_DIR`, `HF_HOME` and `OLLAMA_MODELS` governs
*binaries*: on a pod, anything installed outside `/workspace` is temporary. A vendor
install script that assumes a normal machine will put it in the wrong place.

---

## 2026-08-17 · `torch.cuda.is_available()` is False on a working A5000

**Symptom.** `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`
printed `False 13.0`, while `nvidia-smi` showed a healthy idle RTX A5000 with 24 GB free.

**Cause.** The pod's driver was `550.127.05`, which caps at **CUDA 12.4**. Installed torch
was a **cu130** build. CUDA guarantees compatibility within a major version (any 12.x build
runs on a driver supporting 12.0+), but not across one — a 13.x build on a 12.x driver
finds no usable device. torch does not raise; `is_available()` just returns False.

**What made it slow to find.** RunPod's pod listing advertised "CUDA 14.2", and `SETUP.md`
had been written to say a 14.x pod runs cu130 fine. Both were wrong: that number is not the
driver's CUDA version. Also, `torch.version.cuda` printing `13.0` rather than `None` means
the wheel is *not* CPU-only, so the documented "reinstall the CUDA wheel" fix was a no-op.

**Fix.**
```bash
nvidia-smi | head -3        # read the driver's real CUDA version
pip install --force-reinstall torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

**Lesson.** `nvidia-smi` on the running pod is the only authority on the driver's CUDA
version. Pick the highest published index sharing that **major** version. And distinguish
`False None` (CPU wheel) from `False <version>` (CUDA build too new) — they have opposite
fixes. See `SETUP.md §1` and `§8`.

---

## 2026-08-17 · `operator torchvision::nms does not exist`

**Symptom.** Any `import torchvision` died in `torchvision/_meta_registrations.py` at
`@torch.library.register_fake("torchvision::nms")`. Would have surfaced to a user as
`Ingestion failed: ...` on the first PDF upload, since Marker imports torchvision.

**Cause.** `pip list` showed `torch 2.11.0+cu128` next to `torchvision 0.28.0` — note the
missing `+cu128`. torchvision ships compiled C++ ops linked against one exact torch build;
mismatched, the ops never register.

**Why it kept coming back.** `requirements.txt` listed a bare `torch` and no torchvision,
but `marker-pdf` depends on torchvision — so `pip install -r requirements.txt` pulled it
from **PyPI**, on top of the CUDA torch installed moments earlier. `SETUP.md`'s documented
order (torch first, then requirements) therefore *caused* the breakage every single time.

**Fix.** Uninstall both, then install as a pinned pair from one index —
`--force-reinstall` alone left stale compiled objects behind:
```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
pip check
```
`requirements.txt` now lists `torchvision` explicitly with a warning comment, and
`SETUP.md §1` installs requirements **first** so the CUDA pair is the last write.

**Lesson.** When a package's compiled extensions link against another package, they must
come from the same index, and the CUDA install has to be the last thing that runs.

---

## Earlier · fixed, documented elsewhere

Kept short — each links to where the reasoning lives.

| What broke | Cause | Where it's written up |
|---|---|---|
| Chroma grew 66 → 136 → 210 on uploads of 66/4/4 chunks | `build_chroma` re-added the user's whole accumulated corpus; no explicit ids | `DEPLOYMENT.md §7` |
| Re-uploading a PDF raised `DuplicateIDError` | surfaced by the ids fix above; previously silent duplication | `ingest.merge_chunks` |
| Uploaded PDFs and `.mmd` vanished after ingest | `shutil.rmtree(tmp_dir)` deleted the only copy | fixed by persisting them, then **deliberately reverted** on 2026-08-19 — see the entry above |
| Every entry point raised `NameError` after a merge | `resolve_device()` called but never imported — a textually clean, semantically broken merge | found by importing, not reading |
| `startup.sh` could not run at all | `ALLOW_CPU`/`DO_PULL`/`DO_PREFETCH` read but never assigned; fatal under `set -u`. `bash -n` passed | recovered from commit `eb9044c` |
| Indexes written where retrieval never read | `CHROMA_DIR`/`BM25_DIR` declared in both `retrieval.py` and `ingest.py` | `CLAUDE.md` — define a path once, import it |
| Abstention was structurally impossible | `KBM_RELEVANCE_FLOOR` documented as a raw logit, default `0.0`; the reranker applies a Sigmoid, so every score passed | `api/settings.py`, `CLAUDE.md` |
| Answers claimed document grounding they didn't have | deepseek-math ignores the "say this isn't from your documents" instruction — it is a solver, not an instruction-follower | server prepends the marker; `api/chat.py` |
| Gradio UI crashed on startup | `theme=` passed to `launch()` instead of `Blocks()` on gradio 6.18 | `app.py` |
| Prompt re-prefilled on every turn (~6 s/turn by turn 5) | sliding history window shifted the *start* of the prompt, which a KV prefix cache cannot survive | `LATENCY.md` |
| Every question paid a 4.4 s cold model load | `keep_alive` unset, so Ollama unloaded after 5 min idle | `LATENCY.md`, `DEPLOYMENT.md §5` |
| Math PDFs routed to the wrong extractor | math detected by grepping LaTeX tokens in `pymupdf4llm` output, which contains **no LaTeX at all** | `CLAUDE.md` — the repeat offender |
| Gradio died with `Cannot find empty port in range: 7860-7860` | `APP_HOST` was `0.0.0.0.` — a trailing dot, so `getaddrinfo` failed (Errno -2); gradio's port loop swallows the bind error and blames the port | `DEPLOYMENT.md §4`, `SETUP.md` troubleshooting |
