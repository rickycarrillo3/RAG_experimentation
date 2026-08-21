# Running on a Remote GPU Pod (RunPod)

The app is machine-independent; what changes between your Mac and a rented GPU box is
**where things are stored**, **where Ollama is**, and **who can connect**. All three are
environment variables (see `kbm/config.py`) — there is no code to edit when you move.

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

# ── Find out which CUDA the DRIVER supports. Do this before installing torch. ──
# nvidia-smi is the only authority. The CUDA number in RunPod's pod listing is not the
# driver's CUDA version and has been wrong by two major versions.
nvidia-smi | head -3
#   → "CUDA Version: 12.4"  means install a cu12x wheel, NOT cu130.

# ── Deps, THEN torch. The order is deliberate. ────────────────────────────────
# requirements.txt pulls marker-pdf, which pulls torchvision from PyPI. torchvision
# ships compiled ops linked against one exact torch build, so a PyPI torchvision on top
# of a CUDA-index torch breaks every import with "operator torchvision::nms does not
# exist". Installing torch first therefore does NOT work — requirements.txt clobbers it.
# Install requirements first and let the CUDA pair be the last write.
pip install -r requirements.txt

# Match the index to the driver from nvidia-smi above. cu128 for a 12.x driver:
# any CUDA 12.x build runs on a driver supporting 12.0+, so 12.8 is fine on a 12.4
# driver. A 13.x build is NOT — major versions are outside that guarantee.
pip install --force-reinstall \
    torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# ── Confirm torch sees the GPU AFTER installing ───────────────────────────────
# The one check worth doing by hand; startup.sh and REQUIRE_GPU also enforce it.
python -c "import torch, torchvision; print(torch.cuda.is_available(), torch.version.cuda, torchvision.__version__)"
#   → True 12.8 0.26.0+cu128      good
#   → False None                  CPU-only wheel — reinstall from the CUDA index
#   → False 13.0                  CUDA build NEWER than the driver — install a LOWER
#                                 major version (see §8)
pip check                         # must report no broken requirements

# ── Ollama ────────────────────────────────────────────────────────────────────
# NOT in the pod image. Install it ON THE VOLUME: the official install script
# (curl https://ollama.com/install.sh | sh) writes to /usr/local/bin, which is container
# filesystem and is wiped on every pod stop — you would re-install it on every wake.
# Note this is the ollama *binary*; OLLAMA_MODELS above is where the weights go.
mkdir -p $WORKSPACE/ollama
# ~1.4GB: the archive bundles the CUDA runners. zstd, not gzip — and note the URL is
# the GitHub release asset directly. ollama.com/download/...tgz 404s: that asset name
# no longer exists (renamed to .tar.zst), and the redirect chain hides it behind a
# generic 404 from a URL that looks official.
curl -fL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst \
  | tar --zstd -x -C $WORKSPACE/ollama
# If tar lacks --zstd:  apt-get install -y zstd  then  zstd -dc file | tar -x -C ...
export PATH=$WORKSPACE/ollama/bin:$PATH
ollama --version

# PERSIST IT. The line above lasts until this shell closes, and the next SSH session gets
# `ollama: command not found` all over again — the single most repeated error in this
# project. startup.sh does not need it (it adds the directory itself), but every
# interactive `ollama` command does, and that is the one nobody remembers.
grep -q '/ollama/bin' ~/.bashrc || echo "export PATH=$WORKSPACE/ollama/bin:\$PATH" >> ~/.bashrc

# ── Models ────────────────────────────────────────────────────────────────────
# Optional: `startup.sh` pulls whichever generator KBM_LLM_MODEL names, so you can skip
# straight to §2 and let it do this. Pulling here just pays the download now, on a volume
# you are already watching, rather than inside someone's first question.
ollama serve &
sleep 3
ollama pull t1c/deepseek-math-7b-rl:Q4   # generator, the default (~4.2GB)
ollama pull qwen2:7b                     # eval only: gold-set author + answer judge
# Optional, and the only way to run AGENT MODE — the arm where the model searches the
# family's documents and runs Python itself. deepseek-math cannot: `ollama show` reports
# it as completion-only, with no tools template. See AGENT.md.
ollama pull qwen3:8b                     # tool-capable generator (~5.2GB)

# ── HuggingFace models (~6GB cold) ────────────────────────────────────────────
# Worth doing explicitly here rather than letting it happen mid-use. The embedder
# and reranker would download at app startup anyway; Marker's models would NOT —
# they wait until the first PDF upload, where a multi-GB download surfaces to
# whoever uploaded it as "Ingestion failed". startup.sh runs this too, so this is
# really just to pay the cost now, on a volume you're already watching.
python prefetch_models.py
```

### A note on the pod's CUDA version

**Trust `nvidia-smi` on the pod. Do not trust the CUDA number in RunPod's listing.**

This was learned the expensive way: a pod listed as CUDA 14.2 reported
`Driver Version: 550.127.05 / CUDA Version: 12.4` once running — two major versions
lower. cu130 wheels installed cleanly, imported cleanly, and reported
`torch.cuda.is_available() == False` with an idle A5000 sitting right there.

The rule that actually governs it:

- **Within a CUDA major version, compatibility is guaranteed.** Any 12.x build runs on a
  driver supporting 12.0 or later (driver ≥ 525). So cu128 on a 12.4 driver is fine, and
  you do not need to match minor versions.
- **Across a major version, it is not.** A 13.x build on a 12.x driver fails — the driver
  is older than the toolkit, which is the direction that does not work.

So: read the driver's CUDA version off `nvidia-smi`, then pick the **highest published
index sharing that major version** (12.4 driver → `cu128`). Judge the result by
`torch.cuda.is_available()`, never by matching version numbers to a dashboard label.

For the record, the cu14 question is moot: `download.pytorch.org/whl/cu140/` returns 403
and the newest published index is cu130.

### Set these in the RunPod dashboard → your pod → **Environment Variables**

So every future session inherits them without re-exporting:

| Variable | Value | Why |
|---|---|---|
| `OLLAMA_MODELS` | `/workspace/ollama-models` | else a ~5GB re-pull every restart |
| `HF_HOME` | `/workspace/.cache/huggingface` | else a ~6GB re-download every restart |
| `DATA_DIR` | `/workspace/kb-data` | **else every uploaded document is lost on restart** |
| `APP_AUTH` | `alice:somepassword,bob:another` | else the pod URL is open to anyone who has it |
| `REQUIRE_GPU` | `1` | turns a silent CPU fallback into a startup error |
| `KBM_LLM_MODEL` | *(optional)* `qwen3:8b` | the generator to pull **and** serve; leave unset for deepseek-math. This is the only channel `startup.sh` has for model selection |
| `KBM_TOOLS` | *(optional)* `1` / `0` | force native tool calling on or off, overriding the model's profile. Takes precedence over `KBM_TIR` |

---

## 2. Every session

```bash
cd /workspace/RAG_experimentation/knowledge-base-math
bash startup.sh                             # default generator (deepseek-math)
KBM_LLM_MODEL=qwen3:8b bash startup.sh      # agent mode: tools + document search
```

You do **not** need to `ollama pull` first — stage 3 pulls whatever `KBM_LLM_MODEL` names.
And if a bare `ollama …` gives you `command not found`, the `~/.bashrc` line from §1 is
missing on this pod; `startup.sh` itself is unaffected either way.

`startup.sh` starts Ollama (if not already up), then the **API** on port 8000 and the
Gradio client on 7860. Open the public URL in your browser:
- RunPod dashboard → your pod → **Connect** → **HTTP Service** → port `7860`

**Expose HTTP port `7860` only.** The Gradio client reaches the API over loopback inside
the container, so 8000 never needs to be public — and 11434 (Ollama, unauthenticated)
must never be. To poke the API, SSH in and use `localhost:8000`. `/docs` is *off*
whenever `KBM_API_TOKEN` is set — those routes sit outside the token — so add
`KBM_ENABLE_DOCS=1` if you need them, and only over the SSH tunnel. Full reasoning in
`DEPLOYMENT.md §4`.

`startup.sh` **installs nothing** — it starts things. Creating the venv and the two `pip
install` steps in §1 are one-time; this is what runs every session. Note that a missing
torch and a missing GPU both surface as stage 1's "No CUDA device visible to torch", so
if you see that on a fresh clone, check the install before you blame the pod.

To run it automatically on every pod start, set RunPod's **Container Start Command** —
see `DEPLOYMENT.md §5`.

> **Set `KBM_API_TOKEN` before exposing either port.** Without it the API is open and the
> per-username "isolation" is just a guessable string — anyone who finds the host reads
> every uploaded document. See `DEPLOYMENT.md §4`.

For hosting shape, cost, idle-stop and the full environment-variable table, see
**`DEPLOYMENT.md`**.

---

## 3. What the environment variables do

All defined in `kbm/config.py`; every default reproduces the original local-Mac behaviour, so
nothing changes until you set one.

| Variable | Default | Effect |
|---|---|---|
| `DATA_DIR` | `.` | base dir for `chroma_db/` and `bm25_indexes/` |
| `CHROMA_DIR` / `BM25_DIR` | `$DATA_DIR/...` | override either index path outright |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | point the app at an Ollama on another host |
| `APP_HOST` / `APP_PORT` | `0.0.0.0` / `7860` | bind address and port |
| `APP_AUTH` | unset (no login) | `user:pass` pairs, comma-separated |
| `REQUIRE_GPU` | unset | `1` = refuse to start without CUDA |
| `KBM_LLM_MODEL` | `t1c/deepseek-math-7b-rl:Q4` | which generator to pull and serve. Naming a model also configures it — window, decode budget and tool capability come from `kbm/llm_profiles.py` |
| `KBM_TOOLS` | from the model's profile | `1`/`0` forces native tool calling (`search_documents`, `run_python`, `list_documents`). Wins over `KBM_TIR` |
| `KBM_TIR` | from the model's profile | `1`/`0` forces the ` ```python ` text protocol. Forced off whenever tools are on |
| `KBM_NUM_CTX` / `KBM_NUM_PREDICT` | from the model's profile | window and decode cap; override only to test |
| `KBM_API_TOKEN` | unset (**API is open**) | bearer token for `api/`. Different lock from `APP_AUTH` — see `DEPLOYMENT.md §4` |

**Agent mode in one line.** The default generator has no tools template, so:

```bash
KBM_LLM_MODEL=qwen3:8b bash startup.sh
```

Stage 5 prints which protocol actually came up (`native tool calling`, `tool-integrated
reasoning`, or `no tool protocol`) — if you asked for one and see "no tool protocol", the
model was not pulled or has no tools template, and the log above says which.
`DEPLOYMENT.md §3` is the full variable list.

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

**Generator work** — which model, and whether a tool protocol earns its keep. Every arm
goes through the same script on purpose: it is what supplies the CUDA check, the data
check, the prefetch, `ollama pull`, and the sympy verification of the answer key. An arm
run as a bare `python` skips all of that and is not comparable with one that does not.

```bash
# Which model? Paired McNemar over the same 30 questions (EVALUATION.md §12).
python evaluation/model_bakeoff.py --check     # what is pulled, and the published numbers
python evaluation/model_bakeoff.py             # ~15 min on a pod

# Which tool protocol? Same model, one variable (EVALUATION.md §13).
GEN_MODEL=qwen3:8b bash evaluation/eval.sh --skip-sweep --self-consistency          # CoT
GEN_MODEL=qwen3:8b bash evaluation/eval.sh --skip-sweep --self-consistency --tir    # arm D
GEN_MODEL=qwen3:8b bash evaluation/eval.sh --skip-sweep --self-consistency --tools  # arm E
```

Read the **TOOL USE** block before the accuracy table: `greedy answers that ran a program:
0/30` means the arm measured plain chain-of-thought whatever the accuracy column says.

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

**`requirements.txt` pins `gradio==6.17.3`, and `app.py` targets Gradio 6.** That pin is
the contract: a venv on 5.x will fail on Gradio-6 API changes, and vice versa. The two
that bite are `theme` (moved from `Blocks` to `launch()` in 6.0) and `Chatbot`'s `type`
argument (removed — messages is now the only format).

This mattered because the Mac and the pod drifted apart: the Mac had 5.x while the pod
installed the pin, so `app.py` could only work on one of them at a time. If you
hit a `TypeError` on a `Chatbot` or `Blocks` argument, sync the environment rather than
editing the code:

```bash
cd knowledge-base-math
source venv/bin/activate
pip install -r requirements.txt
pip check          # should print "No broken requirements found."
```

**Why 6.17.3 and not something newer.** Gradio raised its `huggingface-hub` floor to
`>=1.2.0` in **6.18.0**, and `marker-pdf<2.0` caps `transformers<5.0.0`, which in turn caps
`huggingface-hub<1.0`. Those cannot both hold, so 6.18.0 makes the install unresolvable —
see the comment block in `requirements.txt` and the ERRORS.md entry. 6.17.3 is the newest
release still accepting `hub>=0.33.5`, and it is still Gradio 6, so the API changes above
are unaffected.

`pip check` is worth running once after installing; it should print "No broken
requirements found." An install that reports a `huggingface-hub` version outside a
package's declared range is inconsistent in a way pip's resolver would never produce from
scratch, which is exactly what pinning is meant to prevent.

---

## 8. Troubleshooting

**"REQUIRE_GPU=1 but torch.cuda.is_available() is False"** — three different causes, and
`torch.version.cuda` tells them apart. Check it before reinstalling anything:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
nvidia-smi | head -3
```

| Printed | Meaning | Fix |
|---|---|---|
| `False None` | CPU-only wheel | reinstall from the CUDA index (§1) |
| `False 13.0` | CUDA build **newer** than the driver | install a **lower** major version — see §1 |
| `False 12.8` + `nvidia-smi` fails | no GPU in the container | wrong pod type, or GPU not attached |

The middle row is the one that wastes time: the wheel is correct, `--force-reinstall`
changes nothing, and `is_available()` returns False silently rather than raising.
`python -c "import torch; torch.cuda.init()"` surfaces the real error.

**`RuntimeError: operator torchvision::nms does not exist`** (or any traceback from
`torchvision/_meta_registrations.py`) — torch and torchvision are from different builds.
`pip list | grep -i torch` will show one with a `+cuXXX` suffix and one without; the bare
one came from PyPI via `marker-pdf`. Uninstall both and reinstall as a pinned pair from
one index (§1). `--force-reinstall` alone is not enough — uninstall first, or stale
compiled objects survive.

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

**`Chatbot.__init__() got an unexpected keyword argument 'type'`** — the environment is on
gradio **5.x** while the code targets the pinned 6.17.3. `pip install -r requirements.txt`.
A `UserWarning` about `theme` moving to `launch()` just above it is the same mismatch. See §7.

**`ValueError: ... 'hf_transfer' package is not available`** during `prefetch_models.py`
(usually followed by the reranker failing with "Can't load the configuration of
`BAAI/bge-reranker-v2-m3`") — the pod image exports `HF_HUB_ENABLE_HF_TRANSFER=1` and the
package is missing, so every HuggingFace download raises. The reranker's message is a
red herring: the download never ran, so there is no `config.json` to load. `pip install
hf_transfer` in the venv (it is in `requirements.txt`, so `pip install -r
requirements.txt` also fixes it) and re-run. See `ERRORS.md 2026-08-19`.

**`ResolutionImpossible` when installing** — something has been added or bumped that
re-opens the gradio/`marker-pdf` conflict. Check the constraint chain in the
`requirements.txt` header before loosening a pin to make the error go away.
