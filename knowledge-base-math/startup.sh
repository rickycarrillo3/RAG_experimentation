#!/bin/bash
set -e

# ── Model caches (persistent volume, survive pod restarts) ─────────────────────
export OLLAMA_MODELS=${OLLAMA_MODELS:-/workspace/ollama-models}
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}   # Surya/Marker, reranker, embeddings

# ── Data (indexes, telemetry) on the persistent volume, not the container disk ──
export KBM_DATA_DIR=${KBM_DATA_DIR:-/workspace/kbm-data}
mkdir -p "$KBM_DATA_DIR"

# The Gradio client needs the same token as the API it calls.
export KBM_API_TOKEN=${KBM_API_TOKEN:-}
if [ -z "$KBM_API_TOKEN" ]; then
    echo "WARNING: KBM_API_TOKEN is unset — the API will be OPEN. See DEPLOYMENT.md §4." >&2
fi

# ── Ollama ────────────────────────────────────────────────────────────────────

echo "Starting Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama already running."
else
    ollama serve &
    echo "Waiting for Ollama to be ready..."
    until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
        sleep 1
    done
    echo "Ollama ready."
fi

# ── API ───────────────────────────────────────────────────────────────────────
# The deployable unit. app.py is a client of this and cannot start without it.

echo "Starting API on :8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

echo "Waiting for API (loads embeddings + 2.2GB reranker)..."
until curl -s -o /dev/null http://localhost:8000/healthz; do
    # If uvicorn died (bad env, missing model, port taken), stop waiting forever.
    kill -0 "$API_PID" 2>/dev/null || { echo "API failed to start." >&2; exit 1; }
    sleep 2
done
echo "API ready."

# ── UI ────────────────────────────────────────────────────────────────────────
echo "Starting Gradio client on :7860..."
python app.py
