# RunPod Setup Guide

Run these commands **once** after creating your pod. Everything goes to `/workspace` so it survives restarts.

---

## One-Time Workspace Setup

```bash
# 1. Tell Ollama to store models on the persistent volume
export OLLAMA_MODELS=/workspace/ollama-models

# 2. Clone the repo
git clone https://github.com/YOUR_USERNAME/RAG_experimentation.git /workspace/RAG_experimentation

# 3. Install Python dependencies
pip install -r /workspace/RAG_experimentation/RAG_experiments/knowledge-base-math/requirements.txt

# 4. Start Ollama and pull the model (~4.5GB, takes a few minutes)
ollama serve &
sleep 3
ollama pull deepseek-math:7b
```

Add this to your pod's **Environment Variables** in the RunPod dashboard so Ollama always finds the models:
```
OLLAMA_MODELS=/workspace/ollama-models
```

---

## Every Session

```bash
bash /workspace/RAG_experimentation/RAG_experiments/knowledge-base-math/startup.sh
```

Then open the public URL in your browser:
- RunPod dashboard → your pod → **Connect** → **HTTP Service** → port `7860`

---

## Updating the App

When you push new code to GitHub, pull it on the pod:

```bash
cd /workspace/RAG_experimentation && git pull
```

Then restart the app with `startup.sh` as normal.
