"""
api/main.py - FastAPI application.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

from knowledge-base-math/, with the venv active and Ollama running.
"""

import contextlib
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from kbm.logsetup import configure_logging

from .deps import models
from .routes import router
from .settings import API_TOKEN, CORS_ORIGINS, DATA_DIR, ENABLE_DOCS, IDLE_STOP_MINUTES

# Give the application's loggers a handler and a level, before anything else runs. The
# rationale — and the bug that made it necessary — now lives with the implementation in
# kbm/logsetup.py, because app.py needs the same configuration and had drifted into its
# own second copy of it.
configure_logging()

log = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_TOKEN:
        log.warning(
            "KBM_API_TOKEN is unset — this API is OPEN. Fine on localhost; "
            "never deploy a pod like this. Anyone who finds the host can read every "
            "uploaded document."
        )
    log.info("data dir: %s", DATA_DIR)
    log.info("OpenAPI docs: %s", "/docs (open)" if ENABLE_DOCS else "disabled")
    log.info("loading embeddings, reranker, LLM client...")
    models.load()
    log.info("ready.")

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

# The TypeScript frontend will be served from a different origin during development,
# and since the split the Gradio UI is on its own host too. Set KBM_CORS_ORIGINS to the
# real frontend origins before this is publicly reachable — a wildcard plus a bearer
# token means any page the family visits can spend their token, which is also why
# allow_credentials=True below makes "*" invalid rather than merely unwise.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
