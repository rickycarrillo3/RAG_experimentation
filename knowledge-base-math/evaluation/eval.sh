#!/usr/bin/env bash
#
# eval.sh — run the retrieval evaluation on a RunPod GPU pod.
#
# Centrepiece is the embedding × chunking sweep (evaluation/embed_chunk_sweep.py):
# it rebuilds all 9 indexes and prints per-stage GPU latency next to recall/MRR — the
# GPU counterpart to the Mac-CPU table in EVALUATION.md §7, and the number to check
# before deciding bge-m3's query cost is acceptable on the pod.
#
# Usage (from knowledge-base-math/, on the pod):
#     bash evaluation/eval.sh                 # GPU checks + the 9-combo sweep
#     bash evaluation/eval.sh --answers       # also run eval.py --all --answers (LLM-judged, needs Ollama)
#     bash evaluation/eval.sh --skip-sweep --answers   # answers only
#     bash evaluation/eval.sh --skip-sweep --self-consistency           # college tier (the deciding one)
#     bash evaluation/eval.sh --skip-sweep --self-consistency --sc-easy  # grade-school regression tier
#     bash evaluation/eval.sh --allow-cpu     # run even if CUDA is absent (for a local dry-run)
#
# Overridable via env:
#     EMBED_MODEL   embedder for the --answers parity run   (default: BAAI/bge-m3)
#     EVAL_USER     index/user name for the --answers run   (default: calctest)
#     EVAL_DOC      source .mmd for the --answers ingest     (default: docs/extracted/calculus_chainrule.mmd)
#     NORMALIZE=1   LaTeX-normalize embeddings in the --answers run (matches ingest+query)
#     SC_SAMPLES    samples per question for --self-consistency   (default: 10)
#
set -euo pipefail

# ── Options ────────────────────────────────────────────────────────────────────
RUN_SWEEP=1
RUN_ANSWERS=0
RUN_SC=0
SC_TIER="--baseline"
ALLOW_CPU=0
for arg in "$@"; do
  case "$arg" in
    --answers)    RUN_ANSWERS=1 ;;
    --skip-sweep) RUN_SWEEP=0 ;;
    --self-consistency) RUN_SC=1 ;;
    --sc-easy)    SC_TIER="--easy" ;;
    --allow-cpu)  ALLOW_CPU=1 ;;
    -h|--help)    sed -n '2,23p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)"; exit 2 ;;
  esac
done

EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-m3}"
EVAL_USER="${EVAL_USER:-calctest}"
EVAL_DOC="${EVAL_DOC:-docs/extracted/calculus_chainrule.mmd}"
GOLDSET="evaluation/goldset.jsonl"
NORMALIZE="${NORMALIZE:-0}"
SC_SAMPLES="${SC_SAMPLES:-10}"

# ── Model caches on the persistent volume (survive pod restarts) ────────────────
export OLLAMA_MODELS="${OLLAMA_MODELS:-/workspace/ollama-models}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

# This script lives in evaluation/; run from its parent (knowledge-base-math/) so the
# indexes (chroma_db/, bm25_indexes/) and relative paths resolve, regardless of caller cwd.
cd "$(dirname "$0")/.."

# Prefer the project venv if present; otherwise the pod's system python (per SETUP.md).
if [ -f venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi
PY="${PYTHON:-python}"

hr() { printf '─%.0s' $(seq 1 72); echo; }

# ── 1. GPU visible? (the whole point — CPU silently "works" but is 10-40x slower) ─
hr; echo "1. GPU check"
GPU_OK=$($PY - <<'PY'
import torch
ok = torch.cuda.is_available()
print("YES " + torch.cuda.get_device_name(0) if ok else "NO")
PY
)
echo "   CUDA: $GPU_OK"
if [[ "$GPU_OK" == NO* && "$ALLOW_CPU" -ne 1 ]]; then
  echo "   ✗ No CUDA device. The reranker, embeddings, and Marker would all run on CPU."
  echo "     Fix the pod, or pass --allow-cpu to run anyway (slow)."
  exit 1
fi

# ── 2. Required private data present (gitignored — must be uploaded to the pod) ──
hr; echo "2. Data check"
missing=0
for f in "$EVAL_DOC" "$GOLDSET"; do
  if [ -f "$f" ]; then echo "   ✓ $f"; else echo "   ✗ MISSING: $f"; missing=1; fi
done
if [ "$missing" -eq 1 ]; then
  echo
  echo "   The gold set (evaluation/goldset.jsonl) is tracked and arrives via 'git pull'."
  echo "   The extracted .mmd under docs/extracted/ is gitignored (private document text) and"
  echo "   does NOT — upload it with the RunPod file uploader or scp. See SETUP.md § GPU eval run."
  echo "   Do NOT re-extract the PDF on the pod — it shifts chunk_ids and breaks every gold label."
  exit 1
fi

# ── 3. Prefetch ALL models up front, so nothing downloads mid-test ──────────────
# Latency numbers are only meaningful once every model is warm in the cache; a
# first-touch download mid-sweep would both stall and pollute the timings.
hr; echo "3. Prefetch models (before any testing)"

# Embedding models the selected stages will load (the sweep always uses both; the
# reranker is loaded unconditionally below). Prefetch via the pipeline's own loaders,
# not snapshot_download — the loaders fetch only the files they use (not the repo's
# unused ONNX/OpenVINO variants) and double as a "does it load?" check.
EMBED_MODELS=()
if [ "$RUN_SWEEP" -eq 1 ]; then
  EMBED_MODELS+=("BAAI/bge-small-en-v1.5" "BAAI/bge-m3")
fi
if [ "$RUN_ANSWERS" -eq 1 ]; then
  EMBED_MODELS+=("$EMBED_MODEL")
fi

if [ "${#EMBED_MODELS[@]}" -gt 0 ]; then
  echo "   HuggingFace (into HF_HOME=$HF_HOME):"
  # Delegated to prefetch_models.py rather than an inline heredoc, so the app's
  # warm-up and the eval's warm-up cannot drift into fetching different things.
  # --skip-marker: the eval runs against an already-extracted .mmd and must never
  # re-extract (that shifts chunk_ids and breaks every gold label), so the ~3-4GB
  # surya set is dead weight here.
  PREFETCH_ARGS=()
  for m in "${EMBED_MODELS[@]}"; do PREFETCH_ARGS+=(--embed-model "$m"); done
  $PY prefetch_models.py --skip-marker "${PREFETCH_ARGS[@]}"
fi

# Ollama models — only the --answers run needs them; pull before testing, not during.
if [ "$RUN_ANSWERS" -eq 1 ] || [ "$RUN_SC" -eq 1 ]; then
  echo "   Ollama (into OLLAMA_MODELS=$OLLAMA_MODELS):"
  if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "     starting ollama serve..."
    ollama serve > /dev/null 2>&1 &
    until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do sleep 1; done
  fi
  ollama pull t1c/deepseek-math-7b-rl:Q4   # generator (~4.5GB)
  # The judge is only needed for --answers; self-consistency grades against known answers.
  if [ "$RUN_ANSWERS" -eq 1 ]; then
    ollama pull qwen2:7b                     # LLM judge (~4.4GB)
  fi
fi
echo "   All required models present."

# ── 4. The sweep: 9 indexes × {dense, hybrid+rerank}, with GPU latency ──────────
if [ "$RUN_SWEEP" -eq 1 ]; then
  hr; echo "4. Embedding × chunking sweep (GPU)"
  $PY evaluation/embed_chunk_sweep.py --doc "$EVAL_DOC" --goldset "$GOLDSET"
  echo "   → evaluation/results/sweep_results.json  (compare 'dense'/'retr' ms against the Mac numbers)"
fi

# ── 5. Answer-judged parity run (optional; models already pulled in stage 3) ────
if [ "$RUN_ANSWERS" -eq 1 ]; then
  hr; echo "5. Answer-judged eval  (embed: $EMBED_MODEL, normalize_latex=$NORMALIZE)"

  # Ollama was started and models pulled in stage 3; just confirm it's still up.
  if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    ollama serve > /dev/null 2>&1 &
    until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do sleep 1; done
  fi

  NORM_FLAG=""; [ "$NORMALIZE" -eq 1 ] && NORM_FLAG="--normalize-latex"

  echo "   Ingesting '$EVAL_USER' (embed model must match the eval below)..."
  $PY ingest.py --user "$EVAL_USER" "$EVAL_DOC" --embed-model "$EMBED_MODEL" $NORM_FLAG

  echo "   Running eval.py --all --answers..."
  $PY evaluation/eval.py --user "$EVAL_USER" --all --answers --embed-model "$EMBED_MODEL" $NORM_FLAG
  echo "   → evaluation/results/results_*.json + failures_*.json  (answer_score_1to5 = qwen2-judged quality)"
fi

# ── 6. Self-consistency: does majority voting beat greedy? (generator only) ─────
if [ "$RUN_SC" -eq 1 ]; then
  hr; echo "6. Self-consistency ($SC_TIER, $SC_SAMPLES samples/question, closed-book — no retrieval)"

  # A wrong answer key marks a right model wrong. Verify before spending GPU hours on it.
  $PY evaluation/verify_reasoning_set.py

  if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    ollama serve > /dev/null 2>&1 &
    until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do sleep 1; done
  fi

  $PY evaluation/self_consistency.py $SC_TIER --samples "$SC_SAMPLES"
  echo "   → evaluation/results/self_consistency_*.json  (read the VERDICT block, not just the table)"
fi

hr
echo "Done. Results in evaluation/results/. Compare GPU latency against evaluation/EVALUATION.md §7 (Mac CPU)."
