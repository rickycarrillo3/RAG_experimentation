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
                            hybrid retrieval → RRF → cross-encoder rerank → LLM
```

- **Extraction** — Marker (LaTeX-faithful), falling back to pymupdf4llm
- **Retrieval** — BM25 + `bge-small` dense, fused with Reciprocal Rank Fusion
- **Reranking** — `bge-reranker-v2-m3` cross-encoder over the top-20 candidates
- **Generation** — `deepseek-math-7b-rl` via Ollama
- **Isolation** — per-username indexes (a naming convention, not real auth)

## Getting started

```bash
cd knowledge-base-math
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # pinned; read its header before changing
ollama pull t1c/deepseek-math-7b-rl:Q4

python prefetch_models.py              # warm the HF cache (~6GB, one time)
python app.py                          # http://localhost:7860
```

Running on a remote GPU box: see **[`knowledge-base-math/SETUP.md`](knowledge-base-math/SETUP.md)**.

## Docs worth knowing about

- [`SETUP.md`](knowledge-base-math/SETUP.md) — remote GPU pod setup, env vars, troubleshooting
- [`LATENCY.md`](knowledge-base-math/LATENCY.md) — where query time actually goes, and the fixes applied
- [`evaluation/EVALUATION.md`](knowledge-base-math/evaluation/EVALUATION.md) — the eval protocol; **read before changing retrieval**
