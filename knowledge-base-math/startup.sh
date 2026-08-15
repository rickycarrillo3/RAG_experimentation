#!/usr/bin/env bash
#
# startup.sh — start the math RAG web app on a remote GPU box (per session).
#
# Usage (from knowledge-base-math/, on the pod):
#     bash startup.sh                # GPU-required (refuses to start on CPU)
#     bash startup.sh --allow-cpu    # start anyway without CUDA (local dry-run)
#     bash startup.sh --no-pull      # skip the ollama model check (faster restarts)
#
# Overridable via env (defaults suit a RunPod pod with a /workspace volume):
#     WORKSPACE       persistent volume root          (default: /workspace)
#     OLLAMA_MODELS   Ollama model cache              (default: $WORKSPACE/ollama-models)
#     HF_HOME         HuggingFace cache               (default: $WORKSPACE/.cache/huggingface)
#     DATA_DIR        where chroma_db/ + bm25_indexes/ live (default: $WORKSPACE/kb-data)
#     APP_AUTH        "user:pass" login pairs, comma-separated
#     APP_PORT        port to serve on                (default: 7860)
#
set -euo pipefail

ALLOW_CPU=0
DO_PULL=1
for arg in "$@"; do
  case "$arg" in
    --allow-cpu) ALLOW_CPU=1 ;;
    --no-pull)   DO_PULL=0 ;;
    -h|--help)   sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)"; exit 2 ;;
  esac
done

# Run from this script's directory, so the relative paths the Python modules use
# (docs/, evaluation/) resolve regardless of where the caller invoked it from.
cd "$(dirname "$0")"

# ── Persistence ────────────────────────────────────────────────────────────────
# Everything that is expensive to rebuild goes on the persistent volume. A pod's
# container filesystem is wiped on restart: caches left there mean re-downloading
# ~10GB of models every session, and indexes left there mean the family's uploaded
# documents silently vanish. DATA_DIR is why chroma_db/bm25_indexes are configurable.
WORKSPACE="${WORKSPACE:-/workspace}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$WORKSPACE/ollama-models}"
export HF_HOME="${HF_HOME:-$WORKSPACE/.cache/huggingface}"
export DATA_DIR="${DATA_DIR:-$WORKSPACE/kb-data}"
export APP_PORT="${APP_PORT:-7860}"

# Unbuffered, or Python block-buffers stdout the moment this script's output is
# redirected to a log file — which is exactly how it runs as a background service.
# The device line and the auth banner are the two things you most need to see, and
# they would otherwise sit in a buffer until the process exits.
export PYTHONUNBUFFERED=1

mkdir -p "$DATA_DIR"

# Prefer the project venv if present; otherwise the pod's system python.
if [ -f venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi
PY="${PYTHON:-python}"

hr() { printf '─%.0s' $(seq 1 72); echo; }

# ── 1. GPU check ───────────────────────────────────────────────────────────────
# This is the whole reason for renting the box. A CPU fallback is not an error —
# the app starts and answers questions, just 10-40x slower — so without an explicit
# gate you can pay for a GPU for days without ever touching it. REQUIRE_GPU makes
# the Python side refuse too, in case the app is started some other way.
hr; echo "1. GPU check"
if [ "$ALLOW_CPU" -eq 1 ]; then
  echo "   --allow-cpu: skipping the CUDA requirement (expect slow answers)."
  unset REQUIRE_GPU || true
else
  export REQUIRE_GPU=1
  if ! $PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "   ✗ No CUDA device visible to torch."
    echo "     The embedder, reranker and Marker would all run on CPU."
    echo "     Check the pod has a GPU attached and torch has CUDA support:"
    echo "       $PY -c 'import torch; print(torch.__version__, torch.version.cuda)'"
    echo "     Or run 'bash startup.sh --allow-cpu' to start anyway (slow)."
    exit 1
  fi
  echo "   ✓ CUDA: $($PY -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

# ── 2. Access control ──────────────────────────────────────────────────────────
# The app binds 0.0.0.0 so RunPod's HTTP proxy can reach it, which means the pod URL
# is open to anyone who has it. Warn loudly rather than silently serving the family's
# documents to the internet.
hr; echo "2. Access"
if [ -n "${APP_AUTH:-}" ]; then
  echo "   ✓ APP_AUTH set — the app will require a login."
else
  echo "   ⚠ APP_AUTH is not set: anyone with the pod URL can use the app and read"
  echo "     any user's documents. Set it in the RunPod dashboard's Environment"
  echo "     Variables, or inline:  APP_AUTH='name:password' bash startup.sh"
fi

# ── 3. Ollama ──────────────────────────────────────────────────────────────────
hr; echo "3. Ollama"
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "   Already running."
else
    echo "   Starting ollama serve (models in $OLLAMA_MODELS)..."
    ollama serve > /dev/null 2>&1 &
    until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do sleep 1; done
    echo "   Ready."
fi

# Pull on a cold volume rather than on the family's first question — a missing model
# otherwise surfaces as a ~5GB stall inside the first chat request.
if [ "$DO_PULL" -eq 1 ]; then
  GEN_MODEL="t1c/deepseek-math-7b-rl:Q4"
  if ollama list 2>/dev/null | grep -q "${GEN_MODEL%%:*}"; then
    echo "   ✓ $GEN_MODEL present."
  else
    echo "   Pulling $GEN_MODEL (~4.5GB, one time on a fresh volume)..."
    ollama pull "$GEN_MODEL"
  fi
fi

# ── 4. App ─────────────────────────────────────────────────────────────────────
hr; echo "4. Starting app on port $APP_PORT  (indexes in $DATA_DIR)"
echo "   RunPod dashboard → your pod → Connect → HTTP Service → port $APP_PORT"
hr
exec $PY app.py
