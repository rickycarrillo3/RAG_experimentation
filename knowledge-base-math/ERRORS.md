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

## 2026-08-19 · `prefetch_models.py` failed on a second pod: hf_transfer, then a "missing config.json"

**Symptom.** Stage 4 of `startup.sh` on a freshly built pod:

```
↓ embedder BAAI/bge-small-en-v1.5
  FAILED: ValueError: Fast download using 'hf_transfer' is enabled
  (HF_HUB_ENABLE_HF_TRANSFER=1) but 'hf_transfer' package is not available
↓ reranker BAAI/bge-reranker-v2-m3
  FAILED: OSError: Can't load the configuration of 'BAAI/bge-reranker-v2-m3' ...
  make sure ... is the correct path to a directory containing a config.json file
```

**Cause.** One cause, two faces. RunPod's images export `HF_HUB_ENABLE_HF_TRANSFER=1` to
speed up model downloads; `huggingface_hub` honours it by raising on *every* download
when the `hf_transfer` package is absent. It was absent because nothing in
`requirements.txt` asked for it — our code never imports it.

**What made it confusing.** The second error names the wrong problem. Nothing is wrong
with the model id or the cache: the download never ran, so there is no `config.json` on
disk, and `transformers` reports the empty cache as if the repo were bad. Chasing the
reranker message leads to checking model names and clearing caches, none of which is the
issue. **When several models fail in sequence, fix the first failure and re-run before
reading the rest** — later entries in a prefetch list are usually echoes of the first.

**Fix.** `hf_transfer` added to `requirements.txt`, with a comment saying why a package
we never import is there. On a pod that already exists, `pip install hf_transfer` inside
the venv. Unsetting `HF_HUB_ENABLE_HF_TRANSFER` also works but only for that shell — the
variable comes from the image, so the failure returns next session.

**General lesson.** The pod image's environment is part of the dependency set. A variable
someone else exported can make a correct `requirements.txt` incomplete, and the resulting
error names our config rather than theirs.

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
| Uploaded PDFs and `.mmd` vanished after ingest | `shutil.rmtree(tmp_dir)` deleted the only copy | now persisted under `$DATA_DIR/docs/{raw,extracted}/<user>/` |
| Every entry point raised `NameError` after a merge | `resolve_device()` called but never imported — a textually clean, semantically broken merge | found by importing, not reading |
| `startup.sh` could not run at all | `ALLOW_CPU`/`DO_PULL`/`DO_PREFETCH` read but never assigned; fatal under `set -u`. `bash -n` passed | recovered from commit `eb9044c` |
| Indexes written where retrieval never read | `CHROMA_DIR`/`BM25_DIR` declared in both `retrieval.py` and `ingest.py` | `CLAUDE.md` — define a path once, import it |
| Abstention was structurally impossible | `KBM_RELEVANCE_FLOOR` documented as a raw logit, default `0.0`; the reranker applies a Sigmoid, so every score passed | `api/settings.py`, `CLAUDE.md` |
| Answers claimed document grounding they didn't have | deepseek-math ignores the "say this isn't from your documents" instruction — it is a solver, not an instruction-follower | server prepends the marker; `api/chat.py` |
| Gradio UI crashed on startup | `theme=` passed to `launch()` instead of `Blocks()` on gradio 6.18 | `app.py` |
| Prompt re-prefilled on every turn (~6 s/turn by turn 5) | sliding history window shifted the *start* of the prompt, which a KV prefix cache cannot survive | `LATENCY.md` |
| Every question paid a 4.4 s cold model load | `keep_alive` unset, so Ollama unloaded after 5 min idle | `LATENCY.md`, `DEPLOYMENT.md §5` |
| Math PDFs routed to the wrong extractor | math detected by grepping LaTeX tokens in `pymupdf4llm` output, which contains **no LaTeX at all** | `CLAUDE.md` — the repeat offender |
