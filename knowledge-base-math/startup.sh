#!/bin/bash
set -e

# ── Model caches (persistent volume, survive pod restarts) ─────────────────────
export OLLAMA_MODELS=/workspace/ollama-models
export HF_HOME=/workspace/.cache/huggingface   # Surya/Marker, reranker, embeddings

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

# ── App ───────────────────────────────────────────────────────────────────────
echo "Starting app..."
python app.py
