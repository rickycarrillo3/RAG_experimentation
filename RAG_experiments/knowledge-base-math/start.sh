#!/bin/bash
set -e

# Start Ollama in the background
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 1
done
echo "Ollama ready."

# Pull the model if not already present (no-op if cached on persistent volume)
ollama pull deepseek-math:7b-instruct

# Launch the Gradio app
python app.py

# Clean up Ollama on exit
wait $OLLAMA_PID
