# RunPod Setup Guide

Run these commands **once** after creating your pod. Everything goes to `/workspace` so it survives restarts.

---

## One-Time Workspace Setup

```bash
# 1. Pin BOTH model caches to the persistent volume (so they survive pod restarts)
export OLLAMA_MODELS=/workspace/ollama-models
export HF_HOME=/workspace/.cache/huggingface   # Surya (Marker) ~3-4GB + reranker 2.2GB + embeddings

# 2. Clone the repo
git clone https://github.com/rickycarrillo3/RAG_experimentation.git /workspace/RAG_experimentation
# Until the eval-harness PR is merged to main, check out its branch:
cd /workspace/RAG_experimentation && git checkout worktree-rag-eval-harness

# 3. Install Python dependencies
pip install -r /workspace/RAG_experimentation/knowledge-base-math/requirements.txt

# 4. Start Ollama and pull BOTH models
ollama serve &
sleep 3
ollama pull t1c/deepseek-math-7b-rl:Q4   # generator (~4.5GB)
ollama pull qwen2:7b                       # gold-set author + answer judge (make_evalset.py / eval --answers)
```

Add BOTH to your pod's **Environment Variables** in the RunPod dashboard so every session finds the caches:
```
OLLAMA_MODELS=/workspace/ollama-models
HF_HOME=/workspace/.cache/huggingface
```

---

## Every Session

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

## GPU eval run

This reruns the eval harness on the GPU as a **parity check** against the Mac baseline in
`evaluation/EVALUATION.md §7` (does the pod reproduce the numbers, and how fast), plus `--answers`
(LLM-judged end-to-end answers — too slow on CPU, affordable on GPU).

```bash
cd /workspace/RAG_experimentation/knowledge-base-math
source venv/bin/activate 2>/dev/null   # or use the pod's system python
```

**1. Confirm the GPU is actually visible** before anything expensive:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
The reranker (`CrossEncoder`), the HF embeddings, and Marker all auto-detect CUDA — no flags
needed. If this prints `False`, stop and fix the pod, or every stage silently runs on CPU.

**2. Get the corpus onto the pod out-of-band.** The curated gold set
(`knowledge-base-math/evaluation/goldset.jsonl`) is **tracked**, so it arrives with `git pull`.
The extracted `.mmd` under `docs/extracted/` is **gitignored** (it embeds private document text),
so it does **not** — upload it with the RunPod file uploader (or `scp`) into the matching path:
- `knowledge-base-math/docs/extracted/calculus_chainrule.mmd`

> ⚠️ This is what makes it a *parity* run against the identical exam. Do **not** re-extract the PDF
> or regenerate the gold set on the pod — a fresh extraction shifts chunk boundaries, the
> `<source>::<n>` chunk_ids change, and every gold label silently breaks.
> (Optional side experiment: upload `docs/raw/calculus_chainrule.pdf` too and run
> `python extract.py docs/raw/calculus_chainrule.pdf` just to *time* GPU Marker — but still eval
> against the uploaded `.mmd`, not the pod's re-extraction.)

**3. Ingest and run the full sweep with answer judging:**
```bash
python ingest.py --user calctest docs/extracted/calculus_chainrule.mmd
python evaluation/eval.py --user calctest --all --answers
```

**Or just run `evaluation/eval.sh`**, which wraps steps 1–3 (GPU check → data check → prefetch →
the embedding × chunking sweep, with an optional answer-judged run):
```bash
bash evaluation/eval.sh              # GPU checks + the 9-combo latency/quality sweep
bash evaluation/eval.sh --answers    # also ingest + eval.py --all --answers (Ollama-judged)
```

**4. Read the results.** Per-config `results_*.json` + `failures_*.json` land in `evaluation/results/`. Compare
recall@1/@5/@pool, MRR, nDCG against the §7 Mac table — they should match within noise; a real
divergence points at an env or model-version difference, not the pipeline. The new
`answer_score_1to5` field (1–5, judged by `qwen2:7b`) is the end-to-end answer quality per config.

> The gold set is auto-filtered but not yet hand-cleaned (mean `leak_score` 0.15). The run is
> valid as a parity check; `eval.py`'s health header prints the leak stats so the answer numbers
> are read with that caveat.

---

## Updating the App

When you push new code to GitHub, pull it on the pod:

```bash
cd /workspace/RAG_experimentation && git pull
```

Then restart the app with `startup.sh` as normal.
