# RAG experimentation

A math-focused **RAG question-answering system for family use** — upload a textbook,
ask questions, get worked explanations grounded in the actual book.

The goal is quality comparable to Claude or ChatGPT while running entirely on
**free and open-source models and infrastructure**. No paid APIs anywhere in the
pipeline.

## Where things are

| Path | What it is |
|---|---|
| **[`knowledge-base-math/`](knowledge-base-math/)** | **The project.** The only active code in this repo. |
| [`Base RAG explained/`](Base%20RAG%20explained/) | Background reading: how a RAG pipeline works, start to finish |
| [`info_files/`](info_files/) | Longer reference notes — embeddings, fine-tuning, sparse vs dense retrieval |
| [`CLAUDE.md`](CLAUDE.md) | The working spec: architecture, commands, conventions |
| [`plan.md`](plan.md) | ⚠️ Superseded design doc, kept for its reasoning only |

Everything else — earlier experiments in hybrid retrieval, vision, and
speech-to-text — moved to the `archive/experiments` branch once the math system
outgrew being a side project.

## The pipeline

```
PDF → extract.py → .mmd → ingest.py → BM25 + Chroma (per user)
                                            ↓
                   api/  ←  hybrid retrieval → RRF → cross-encoder rerank → LLM
                    ↑                                        ↑
                 app.py                          (agent mode: the model can
              (HTTP client)                       search again, and run Python)
```

- **Extraction** — Marker (LaTeX-faithful), falling back to pymupdf4llm
- **Retrieval** — BM25 + `bge-small` dense, fused with Reciprocal Rank Fusion
- **Reranking** — `bge-reranker-v2-m3` cross-encoder over the top-20 candidates
- **Generation** — `qwen3:8b` via Ollama by default; the generator is a
  deployment knob (`KBM_LLM_MODEL`), and naming a model also configures it
- **Tools** — the default generator is tool-capable, so out of the box the model runs
  Python in a sandbox and decides when to search the documents again
  ([`AGENT.md`](knowledge-base-math/AGENT.md))
- **Serving** — `api/` is the deployable unit; `app.py` is an HTTP client of it
- **Isolation** — per-username indexes (a naming convention, **not real auth**), which is
  load-bearing once the model can search: see [`DEPLOYMENT.md §8`](knowledge-base-math/DEPLOYMENT.md)

## Getting started

```bash
cd knowledge-base-math
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # pinned; read its header before changing
ollama pull qwen3:8b                    # the generator (tool-capable: agent mode is on)
ollama pull qwen2:7b                    # eval only (gold-set author + judge)
ollama pull t1c/deepseek-math-7b-rl:Q4  # optional: the previous default, no tool protocol

python prefetch_models.py              # warm the HF cache (~6GB, one time)

# app.py is a CLIENT of the API, not a standalone app — start the API first,
# in its own shell, or the UI comes up and cannot answer anything.
uvicorn api.main:app --port 8000        # shell 1
python app.py                           # shell 2 → http://localhost:7860
```

Agent mode is one variable: `KBM_LLM_MODEL=qwen3:8b`, on both shells.

Running on a remote GPU box: see **[`knowledge-base-math/SETUP.md`](knowledge-base-math/SETUP.md)**.

## Docs worth knowing about

- [`SETUP.md`](knowledge-base-math/SETUP.md) — remote GPU pod setup, env vars, troubleshooting
- [`LATENCY.md`](knowledge-base-math/LATENCY.md) — where query time actually goes, and the fixes applied
- [`ARCHITECTURE.md`](knowledge-base-math/ARCHITECTURE.md) — what runs on the GPU, what on the CPU, and the VRAM budget
- [`AGENT.md`](knowledge-base-math/AGENT.md) — the two tool protocols, why a model gets exactly one, and why the loop is hand-rolled
- [`DEPLOYMENT.md`](knowledge-base-math/DEPLOYMENT.md) — hosting, cost, env vars, and what the sandbox does and does not protect
- [`ERRORS.md`](knowledge-base-math/ERRORS.md) — failures actually hit and what really caused them; **check before debugging an environment problem**
- [`evaluation/EVALUATION.md`](knowledge-base-math/evaluation/EVALUATION.md) — the eval protocol; **read before changing retrieval**
