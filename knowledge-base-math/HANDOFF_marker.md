# Handoff: switch PDF extraction from Nougat → Marker

**Status:** code changes made & committed on branch `worktree-marker-extraction` (git worktree
at `.claude/worktrees/marker-extraction/`). NOT yet pushed / no PR. Full RAG pipeline already
proven end-to-end with Marker output. One task left: smoke-test the *edited* `extract.py`, then
push + open draft PR.

## Original ask
"Try the RAG pipeline with a sample PDF." Correction from user mid-way: **do not use the
pymupdf fallback — use a math-faithful extractor** that preserves equations + context.

## Decision
Nougat is dead (unmaintained since 2023, needs ancient `transformers` that breaks
`sentence-transformers`; won't import on this env — `ImportError: cannot import name
'PretrainedConfig'`). User picked **Marker** (`marker-pdf`, uses Surya models) over MinerU /
Docling / VLM. Marker output is dramatically better than pymupdf:
- pymupdf: `_Qπ_ ( _s, a_ ) _≡_ E [ _R_ 1 + _γR_ 2` (garbage)
- Marker: `$$Q_{\pi}(s,a) \equiv \mathbb{E}[R_1 + \gamma R_2 + \dots \mid S_0 = s, ...],$$`

## Environment changes already applied (to the MAIN checkout venv, gitignored)
`/Users/ricardocarrillo/Desktop/RAG_experimentation/knowledge-base-math/venv`
- The venv `python` symlink still works, but `pip`'s shebang is stale (old pre-flatten path).
  **Use `./venv/bin/python -m pip ...`, never `./venv/bin/pip`.**
- Uninstalled `nougat-ocr` and `timm 0.5.4`.
- Installed `marker-pdf 1.10.2` + `surya-ocr 0.17.1`. This downgraded `transformers`
  5.12.0→4.57.6 and `huggingface-hub` 1.19.0→0.36.2.
- **Soft conflict (not yet resolved):** `gradio 6.18.0` wants `huggingface-hub>=1.2.0` but now
  has 0.36.2. gradio *still imports fine at 6.18.0*, but `app.py` (web UI) is UNTESTED under
  this pin — verify before trusting the web UI.
- Marker's Surya models are now cached in the HF cache (~3–4 GB), so re-runs skip the download.
- All core imports verified OK: marker, surya, sentence_transformers, chromadb, langchain_*,
  rank_bm25, ollama, gradio.

## What is PROVEN working (done in main checkout, all data dirs gitignored)
1. Sample PDF: `docs/raw/sample.pdf` (arXiv 1509.06461, Double DQN, 6 pages, has equations).
2. Marker extraction → `docs/extracted/sample.mmd` (copied from a manual `marker_single` run).
   548 lines, faithful LaTeX + clean markdown tables. Took ~11 min on this Mac's CPU
   (mostly inference; GPU would be far faster).
3. `python ingest.py --user sampletest docs/extracted/sample.mmd` → 219 chunks, BM25 pickle +
   Chroma collection `user_sampletest` built.
4. Retrieval-only query works, hybrid BM25+dense RRF, LaTeX preserved in chunks.
5. Full query through `t1c/deepseek-math-7b-rl:Q4` (Ollama, already running) → coherent, correct
   answer on "Double DQN target vs DQN target" with sources. **Pipeline works end to end.**

Non-interactive query pattern (query.py is an input() loop):
```bash
printf 'YOUR QUESTION\nquit\n' | ./venv/bin/python query.py --user sampletest
```

## Code changes on this branch (committed)
- `knowledge-base-math/extract.py` — replaced `extract_nougat()` (subprocess) with
  `extract_marker()` using the Marker Python API (lazy imports so `--force-pymupdf` stays
  light); `extract()` now tries Marker on math PDFs and falls back to pymupdf on ANY exception.
- `knowledge-base-math/requirements.txt` — `nougat-ocr` → `marker-pdf`.
- `CLAUDE.md` — extractor description, `--force-pymupdf` comment, gitignore note.
- `info_files/system_deep_dive.md` — Nougat → Marker references.

Marker API used (verified for marker-pdf 1.10.2):
```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
converter = PdfConverter(artifact_dict=create_model_dict())
rendered = converter(pdf_path)
text, _, _ = text_from_rendered(rendered)   # text = markdown+LaTeX
```

## LEFT TO DO (resume here)
1. **Smoke-test the edited `extract.py`** end-to-end (the manual run used `marker_single` CLI;
   the code path via `extract_marker()` is verified only by import + py_compile, not a full run).
   The worktree has no venv — run with the main venv's python from the worktree dir:
   ```bash
   cd .claude/worktrees/marker-extraction/knowledge-base-math
   MAINVENV=/Users/ricardocarrillo/Desktop/RAG_experimentation/knowledge-base-math/venv/bin/python
   $MAINVENV extract.py /Users/ricardocarrillo/Desktop/RAG_experimentation/knowledge-base-math/docs/raw/sample.pdf
   # ~several min on CPU; expect docs/extracted/sample.mmd with $$...$$ LaTeX
   ```
   (An unfinished background run of exactly this was killed when we paused.)
2. Optionally verify `app.py` still works under the huggingface-hub 0.36.2 downgrade, or bump
   the gradio/hf-hub pins to resolve the soft conflict.
3. `git push -u origin worktree-marker-extraction` and `gh pr create --draft`.
4. Consider whether to always use Marker vs keep the math-autodetect switch (currently kept:
   math→Marker, non-math→pymupdf).
