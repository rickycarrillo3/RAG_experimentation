"""
api/main.py - FastAPI application.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

from knowledge-base-math/, with the venv active and Ollama running.
"""

import contextlib
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .deps import models
from .routes import public_router, router
from .settings import API_TOKEN, DATA_DIR, IDLE_STOP_MINUTES


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_TOKEN:
        print(
            "[api] WARNING: KBM_API_TOKEN is unset — this API is OPEN. Fine on localhost; "
            "never deploy a pod like this. Anyone who finds the host can read every "
            "uploaded document.",
            file=sys.stderr,
        )
    print(f"[api] data dir: {DATA_DIR}")
    print("[api] loading embeddings, reranker, LLM client...")
    models.load()
    print("[api] ready.")

    if IDLE_STOP_MINUTES > 0:
        from ops.idle_stop import start_watchdog

        start_watchdog()

    yield


app = FastAPI(
    title="knowledge-base-math API",
    description="Math RAG QA: hybrid retrieval + cross-encoder rerank + local LLM.",
    version="0.1.0",
    lifespan=lifespan,
)

# The TypeScript frontend will be served from a different origin during development.
# Tighten `allow_origins` to the real frontend host before this is publicly reachable —
# a wildcard plus a bearer token means any page the family visits can spend their token.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:7860"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(public_router)   # /healthz — unauthenticated, see routes.py
app.include_router(router)
