#!/usr/bin/env bash
#
# startup.sh — start the math RAG web app on a remote GPU box (per session).
#
# Usage (from knowledge-base-math/, on the pod):
#     bash startup.sh                 # GPU-required (refuses to start on CPU)
#     bash startup.sh --allow-cpu     # start anyway without CUDA (local dry-run)
#     bash startup.sh --no-pull       # skip the ollama model check (faster restarts)
#     bash startup.sh --no-prefetch   # skip the HuggingFace model warm-up
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
DO_PREFETCH=1
for arg in "$@"; do
  case "$arg" in
    --allow-cpu)   ALLOW_CPU=1 ;;
    --no-pull)     DO_PULL=0 ;;
    --no-prefetch) DO_PREFETCH=0 ;;
    -h|--help)     sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)"; exit 2 ;;
  esac
done

# The Gradio client needs the same token as the API it calls. This is a *different*
# lock from APP_AUTH: APP_AUTH gates the Gradio login page, KBM_API_TOKEN gates the API
# behind it. A login page in front of an open API only protects the page.
export KBM_API_TOKEN=${KBM_API_TOKEN:-}
if [ -z "$KBM_API_TOKEN" ]; then
    echo "WARNING: KBM_API_TOKEN is unset — the API will be OPEN. See DEPLOYMENT.md §4." >&2
fi

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
export API_PORT="${API_PORT:-8000}"
# app.py talks to the API over loopback; only APP_PORT needs exposing publicly.
export KBM_API_URL="${KBM_API_URL:-http://127.0.0.1:$API_PORT}"

# Unbuffered, or Python block-buffers stdout the moment this script's output is
# redirected to a log file — which is exactly how it runs as a background service.
# The device line and the auth banner are the two things you most need to see, and
# they would otherwise sit in a buffer until the process exits.
export PYTHONUNBUFFERED=1

mkdir -p "$DATA_DIR"

# Ollama is not in the pod image, and the official install script puts it in
# /usr/local/bin — container filesystem, wiped on every restart. It is installed to the
# volume instead (SETUP.md §1), so put that on PATH here rather than relying on every
# future shell to export it.
if [ -x "$WORKSPACE/ollama/bin/ollama" ]; then
  export PATH="$WORKSPACE/ollama/bin:$PATH"
fi

if ! command -v ollama > /dev/null 2>&1; then
  echo "✗ ollama not found on PATH, and not at $WORKSPACE/ollama/bin/ollama." >&2
  echo "  The pod image does not ship it. Install it ON THE VOLUME — an install to" >&2
  echo "  /usr/local/bin does not survive a pod restart:" >&2
  echo "    mkdir -p $WORKSPACE/ollama" >&2
  echo "    curl -fsSL https://ollama.com/download/ollama-linux-amd64.tgz | tar -xz -C $WORKSPACE/ollama" >&2
  exit 1
fi

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

# ── 4. HuggingFace models ──────────────────────────────────────────────────────
# Warm the HF cache before serving. The embedder and reranker would download at
# app.py import anyway (making startup look hung); Marker would not download until
# the first PDF upload, surfacing a multi-GB fetch as an "Ingestion failed" error
# in front of whoever uploaded it. Cheap no-op once the volume is warm.
hr; echo "4. HuggingFace models (cache: $HF_HOME)"
if [ "$DO_PREFETCH" -eq 1 ]; then
  $PY prefetch_models.py || {
    echo "   ✗ Prefetch reported failures — see above. Not starting the app."
    echo "     Re-run with --no-prefetch to start anyway."
    exit 1
  }
else
  echo "   --no-prefetch: skipped."
fi

# ── 5. API ─────────────────────────────────────────────────────────────────────
# The deployable unit. app.py is a client of this and cannot start without it.
hr; echo "5. API (:$API_PORT)"

if [ -n "${KBM_IDLE_STOP_MINUTES:-}" ] && [ "${KBM_IDLE_STOP_MINUTES:-0}" -gt 0 ]; then
  if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "   Idle-stop armed: pod stops after ${KBM_IDLE_STOP_MINUTES} min without a query."
  else
    echo "   ⚠ KBM_IDLE_STOP_MINUTES is set but RUNPOD_API_KEY/RUNPOD_POD_ID are not."
    echo "     The pod will NOT stop itself and will bill continuously (~\$115/mo)."
  fi
else
  echo "   ⚠ Idle-stop disabled: this pod bills until you stop it by hand."
fi

$PY -m uvicorn api.main:app --host 0.0.0.0 --port "$API_PORT" &
API_PID=$!

# Trap so Ctrl-C or a failure below does not leave a headless uvicorn holding the GPU.
trap 'kill "$API_PID" 2>/dev/null || true' EXIT

echo "   Loading embedder + 2.2GB reranker..."
until curl -s -o /dev/null "http://localhost:$API_PORT/healthz"; do
    # If uvicorn died (bad env, missing model, port taken), stop waiting forever.
    kill -0 "$API_PID" 2>/dev/null || { echo "   ✗ API failed to start (see above)." >&2; exit 1; }
    sleep 2
done
echo "   ✓ Ready."

# ── 6. UI ──────────────────────────────────────────────────────────────────────
hr; echo "6. Web UI (:$APP_PORT)"
echo "   RunPod dashboard → Connect → HTTP Service → port $APP_PORT"
$PY app.py
