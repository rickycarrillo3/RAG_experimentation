# Running on a Remote GPU Pod (RunPod)

The app is machine-independent; what changes between your Mac and a rented GPU box is
**where things are stored**, **where Ollama is**, and **who can connect**. All three are
environment variables (see `config.py`) — there is no code to edit when you move.

The single rule that governs everything below: **a pod's container filesystem is wiped on
restart, only `/workspace` survives.** Anything expensive to rebuild (model caches) or
impossible to rebuild (the family's uploaded documents) must live on `/workspace`.

---

## 1. One-time pod setup

```bash
# ── Caches and data on the persistent volume ──────────────────────────────────
export WORKSPACE=/workspace
export OLLAMA_MODELS=$WORKSPACE/ollama-models          # Ollama models (~5-10GB)
export HF_HOME=$WORKSPACE/.cache/huggingface           # Marker ~3-4GB + reranker 2.2GB + embedder
export DATA_DIR=$WORKSPACE/kb-data                     # chroma_db/ + bm25_indexes/ (user documents)

# ── Code ──────────────────────────────────────────────────────────────────────
git clone https://github.com/rickycarrillo3/RAG_experimentation.git $WORKSPACE/RAG_experimentation
cd $WORKSPACE/RAG_experimentation/knowledge-base-math
# main is current — no branch checkout needed.

# ── Python deps ───────────────────────────────────────────────────────────────
python -m venv venv          # optional; skip to use the pod's system python
source venv/bin/activate
pip install -r requirements.txt   # fully pinned; see the header of that file

# ── Confirm torch still sees the GPU AFTER installing ─────────────────────────
# requirements.txt lists a bare `torch`. On most RunPod images pip keeps the CUDA
# build, but an image with a pinned torch can end up downgraded by this install.
# This is the one check worth doing by hand; `startup.sh` also enforces it.
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
#   → True 12.x       good
#   → False None      CPU-only wheel: reinstall from the CUDA index, e.g.
#                     pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121

# ── Models ────────────────────────────────────────────────────────────────────
ollama serve &
sleep 3
ollama pull t1c/deepseek-math-7b-rl:Q4   # generator (~4.5GB)
ollama pull qwen2:7b                     # eval only: gold-set author + answer judge

# ── HuggingFace models (~6GB cold) ────────────────────────────────────────────
# Worth doing explicitly here rather than letting it happen mid-use. The embedder
# and reranker would download at app startup anyway; Marker's models would NOT —
# they wait until the first PDF upload, where a multi-GB download surfaces to
# whoever uploaded it as "Ingestion failed". startup.sh runs this too, so this is
# really just to pay the cost now, on a volume you're already watching.
python prefetch_models.py
```

### Set these in the RunPod dashboard → your pod → **Environment Variables**

So every future session inherits them without re-exporting:

| Variable | Value | Why |
|---|---|---|
| `OLLAMA_MODELS` | `/workspace/ollama-models` | else a ~5GB re-pull every restart |
| `HF_HOME` | `/workspace/.cache/huggingface` | else a ~6GB re-download every restart |
| `DATA_DIR` | `/workspace/kb-data` | **else every uploaded document is lost on restart** |
| `APP_AUTH` | `alice:somepassword,bob:another` | else the pod URL is open to anyone who has it |
| `REQUIRE_GPU` | `1` | turns a silent CPU fallback into a startup error |

---

## 2. Every session

```bash
cd /workspace/RAG_experimentation/knowledge-base-math
bash startup.sh
```

`startup.sh` starts Ollama (if not already up), then the **API** on port 8000 and the
Gradio client on 7860. Open the public URL in your browser:
- RunPod dashboard → your pod → **Connect** → **HTTP Service** → port `7860` (UI) or `8000` (API, with `/docs`)

> **Set `KBM_API_TOKEN` before exposing either port.** Without it the API is open and the
> per-username "isolation" is just a guessable string — anyone who finds the host reads
> every uploaded document. See `DEPLOYMENT.md §4`.

For hosting shape, cost, idle-stop and the full environment-variable table, see
**`DEPLOYMENT.md`**.

---

## 3. What the environment variables do

All defined in `config.py`; every default reproduces the original local-Mac behaviour, so
nothing changes until you set one.

| Variable | Default | Effect |
|---|---|---|
| `DATA_DIR` | `.` | base dir for `chroma_db/` and `bm25_indexes/` |
| `CHROMA_DIR` / `BM25_DIR` | `$DATA_DIR/...` | override either index path outright |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | point the app at an Ollama on another host |
| `APP_HOST` / `APP_PORT` | `0.0.0.0` / `7860` | bind address and port |
| `APP_AUTH` | unset (no login) | `user:pass` pairs, comma-separated |
| `REQUIRE_GPU` | unset | `1` = refuse to start without CUDA |

**On `APP_AUTH`:** it is a front door lock, not real authentication. It controls *access to
the app*. Behind it, users are still just lowercased strings typed into a textbox — anyone
who logs in can read any user's documents by typing their name. Fine for family; not fine
for a link shared more widely.

---

## 4. Moving your existing documents to the pod

`chroma_db/` and `bm25_indexes/` are gitignored (private document text), so they do **not**
arrive with `git pull`. Two options:

**Re-ingest on the pod** (simplest, and faster on a GPU) — upload the PDFs or `.mmd` files,
then:
```bash
python ingest.py --user alice docs/extracted/textbook.mmd
```

**Or copy the indexes up** and skip re-ingesting:
```bash
# from your Mac, into whatever DATA_DIR points at
scp -r knowledge-base-math/chroma_db      pod:/workspace/kb-data/
scp -r knowledge-base-math/bm25_indexes   pod:/workspace/kb-data/
```
The indexes are portable as long as the embedding model matches — an index built with
`bge-small` must be queried with `bge-small`, or the vectors are meaningless.

---

## 5. GPU eval run

A **parity check** against the Mac baseline in `evaluation/EVALUATION.md §7` (does the pod
reproduce the numbers, and how fast), plus `--answers` (LLM-judged end-to-end answers — too
slow on CPU, affordable on GPU).

**Get the corpus onto the pod out-of-band.** The curated gold set
(`evaluation/goldset.jsonl`) is **tracked** and arrives with `git pull`. The extracted
`.mmd` under `docs/extracted/` is **gitignored** and does not — upload it with the RunPod
file uploader or `scp` to the matching path:
- `knowledge-base-math/docs/extracted/calculus_chainrule.mmd`

> ⚠️ Do **not** re-extract the PDF or regenerate the gold set on the pod. A fresh extraction
> shifts chunk boundaries, the `<source>::<n>` chunk_ids change, and every gold label
> silently breaks. Uploading the same `.mmd` is what makes it a *parity* run against an
> identical exam.

```bash
cd /workspace/RAG_experimentation/knowledge-base-math
bash evaluation/eval.sh              # GPU check → data check → prefetch → 9-combo sweep
bash evaluation/eval.sh --answers    # also ingest + eval.py --all --answers (Ollama-judged)
```

Results land in `evaluation/results/`. Compare recall@1/@5/@pool, MRR, nDCG against the §7
Mac table — they should match within noise; a real divergence points at an env or
model-version difference, not the pipeline. `answer_score_1to5` (judged by `qwen2:7b`) is
end-to-end answer quality per config.

---

## 6. Updating the app

```bash
cd /workspace/RAG_experimentation && git pull
```
Then restart with `startup.sh`. Your indexes live in `DATA_DIR`, outside the repo, so a
pull never touches them.

---

## 7. Syncing an existing local venv

`requirements.txt` is now pinned, and the pin moves **gradio from 6.18 to 5.50** (the
reason is in that file's header — Gradio 6 and `marker-pdf` cannot coexist). A venv
created before this change still has gradio 6.18, where `app.py` now fails with
`Chatbot.__init__() got an unexpected keyword argument 'type'`. Sync it:

```bash
cd knowledge-base-math
source venv/bin/activate
pip install -r requirements.txt
pip check          # should print "No broken requirements found."
```

`pip check` is worth running once: the pre-pin environment reported
`gradio 6.18.0 has requirement huggingface-hub<2.0,>=1.2.0, but you have 0.36.2` — an
inconsistent install that pip's resolver would never produce from scratch, which is
exactly what pinning is meant to prevent.

---

## 8. Troubleshooting

**"REQUIRE_GPU=1 but torch.cuda.is_available() is False"** — the pod has no GPU attached, or
pip installed a CPU-only torch. See the reinstall command in §1.

**The app starts but answers are very slow** — check the startup log for
`[config] CUDA available — pinning models to GPU: ...`. If it says "No CUDA device"
instead, everything is running on CPU.

**Uploaded documents disappeared after a restart** — `DATA_DIR` was not set, so the indexes
were written to the container filesystem instead of `/workspace`. Set it in the dashboard
env vars (§1) and re-ingest.

**Nothing appears in the log file** — `startup.sh` exports `PYTHONUNBUFFERED=1` for exactly
this reason; if you launch `python app.py` directly and redirect output, add it yourself or
Python buffers the diagnostics until the process exits.

**Can't reach the app** — `APP_HOST` must stay `0.0.0.0` for RunPod's HTTP proxy to reach
it; `127.0.0.1` will bind successfully and be unreachable from outside.

**`Chatbot.__init__() got an unexpected keyword argument 'type'`** — the environment has
gradio 6.x but the code targets the pinned 5.50. See §7.

**`ResolutionImpossible` when installing** — something has been added or bumped that
re-opens the gradio/`marker-pdf` conflict. Check the constraint chain in the
`requirements.txt` header before loosening a pin to make the error go away.
