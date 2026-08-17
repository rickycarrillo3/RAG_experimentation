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
