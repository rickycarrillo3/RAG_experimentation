"""
api/main.py - FastAPI application.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

from knowledge-base-math/, with the venv active and Ollama running.
"""

import contextlib
import logging
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .deps import models
from .routes import router
from .settings import API_TOKEN, DATA_DIR, ENABLE_DOCS, IDLE_STOP_MINUTES

# Give the application's loggers a handler and a level. Without this the root logger has
# none, so Python falls back to its handler-of-last-resort — which emits WARNING and above
# and drops INFO entirely. Every `log.info` in api/routes.py was therefore invisible,
# including the sandbox-failure line whose own comment says it is logged, and every tool
# call. Errors still appeared, which is exactly why nobody noticed the rest was missing.
#
# basicConfig and not dictConfig: uvicorn owns its own loggers and this must not fight
# them. It only installs a handler on the ROOT logger, which is what `logging.getLogger(
# __name__)` in this package resolves to. KBM_LOG_LEVEL=WARNING restores the old quiet.
logging.basicConfig(
    level=os.environ.get("KBM_LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s:     %(name)s - %(message)s",
)


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
    print(f"[api] OpenAPI docs: {'/docs (open)' if ENABLE_DOCS else 'disabled'}")
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
    # None removes the route entirely rather than hiding a link to it. See
    # settings.ENABLE_DOCS: these are the one part of the surface the bearer token does
    # not cover, because they belong to the app and the token guards the router.
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
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

app.include_router(router)
