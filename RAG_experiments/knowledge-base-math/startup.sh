#!/bin/bash
set -e

# ── Ollama ────────────────────────────────────────────────────────────────────
export OLLAMA_MODELS=/workspace/ollama-models

echo "Starting Ollama..."
ollama serve &

echo "Waiting for Ollama to be ready..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 1
done
echo "Ollama ready."

# ── App ───────────────────────────────────────────────────────────────────────
cd /workspace/RAG_experimentation/RAG_experiments/knowledge-base-math

echo "Starting app..."
python app.py
