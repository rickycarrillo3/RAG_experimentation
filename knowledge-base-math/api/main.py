"""
api/main.py - FastAPI application.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

from knowledge-base-math/, with the venv active and Ollama running.
"""

import contextlib
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from kbm.logsetup import configure_logging

from .deps import models
from .routes import router
from .settings import API_TOKEN, CORS_ORIGINS, DATA_DIR, ENABLE_DOCS, IDLE_STOP_MINUTES

# Give the application's loggers a handler and a level, before anything else runs. The
# rationale — and the bug that made it necessary — now lives with the implementation in
# kbm/logsetup.py.
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

# Only matters for a frontend on ANOTHER origin (the Vite dev server); the built one
# below is same-origin. A wildcard plus a bearer token would mean any page the family
# visits can spend their token, which is also why allow_credentials=True below makes
# "*" invalid rather than merely unwise.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Twice, on purpose. At the root for the CLI, startup.sh's /healthz gate and the
# generated schema; under /api for the frontend, which calls `/api/...` both here and
# through the Vite dev proxy (which strips the prefix). Hidden from the schema so the
# TS types are generated once, not twice.
app.include_router(router)
app.include_router(router, prefix="/api", include_in_schema=False)

# The React build (web/dist, from `npm run build` or docker/Dockerfile.gpu), served by
# this process so one pod and one port is the whole deployment. Mounted LAST: a mount at
# "/" matches everything, so it must only see paths no route claimed. The static files
# are not behind the bearer token — they are the public JS bundle; every request that
# reads a document goes through the router above, which is.
_WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
if _WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIST, html=True), name="web")
else:
    log.info("no web/dist — serving the API only (run `npm run build` in web/)")
