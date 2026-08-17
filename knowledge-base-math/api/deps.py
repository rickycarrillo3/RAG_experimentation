"""
api/deps.py - Shared model handles and the auth dependency.

The embedder, reranker and LLM client are loaded once at process start and shared
across requests, exactly as app.py did at module scope. Loading the 2.2GB reranker
per request would dominate every other cost in the system.
"""

import os
import pickle
import threading

from fastapi import Header, HTTPException, status
from langchain_ollama import ChatOllama

import retrieval
from config import OLLAMA_BASE_URL
from retrieval import BM25_DIR, OLLAMA_MODEL, load_embeddings, load_reranker

from .settings import API_TOKEN, KEEP_ALIVE, NUM_PREDICT


class Models:
    """Lazily-built singletons, so importing this module stays cheap (tests, CLI tools)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.embeddings = None
        self.reranker = None
        self.llm = None

    def load(self) -> None:
        with self._lock:
            if self.embeddings is None:
                self.embeddings = load_embeddings()
            if self.reranker is None:
                self.reranker = load_reranker()
            if self.llm is None:
                self.llm = ChatOllama(
                    model=OLLAMA_MODEL,
                    # Same knob query.py and test_chat.py use, so the API can point at
                    # an Ollama on another host instead of assuming it shares the box.
                    base_url=OLLAMA_BASE_URL,
                    temperature=0,
                    num_predict=NUM_PREDICT,
                    # Without this Ollama unloads after 5 min idle and the next question
                    # pays a cold model load. Note it interacts with the idle-stop
                    # watchdog: keep_alive should be shorter than the idle-stop window,
                    # or the pod stops while still holding the model resident for nothing.
                    keep_alive=KEEP_ALIVE,
                )

    @property
    def ready(self) -> bool:
        return all(x is not None for x in (self.embeddings, self.reranker, self.llm))


models = Models()


# ── Per-user index helpers ────────────────────────────────────────────────────

def bm25_path(user: str) -> str:
    return os.path.join(BM25_DIR, f"user_{user}.pkl")


def has_index(user: str) -> bool:
    return os.path.exists(bm25_path(user))


def index_summary(user: str) -> tuple[int, list[str]]:
    """(chunk count, distinct source basenames) from the BM25 pickle.

    Read from BM25 rather than Chroma because the pickle already holds the Document
    objects; asking Chroma would mean a collection scan for the same answer.
    """
    if not has_index(user):
        return 0, []
    with open(bm25_path(user), "rb") as f:
        data = pickle.load(f)
    chunks = data["chunks"]
    sources = sorted({os.path.basename(c.metadata.get("source", "unknown")) for c in chunks})
    return len(chunks), sources


def normalize_user(user: str) -> str:
    """Usernames are lowercased and stripped — the same rule app.py applied.

    This is the *entire* isolation mechanism (CLAUDE.md: "per-username directory naming,
    not real auth"), so it must be applied in exactly one place. Path separators are
    rejected because the username lands in a filename.
    """
    u = user.strip().lower()
    if not u:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Username is required.")
    if any(c in u for c in "/\\.\0") or u in (".", ".."):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Username may not contain path characters.")
    return u


# ── Auth ──────────────────────────────────────────────────────────────────────

async def require_token(authorization: str = Header(default="")) -> None:
    """Shared-secret gate. See settings.API_TOKEN for what this does and does not buy.

    An unset token leaves the API open. That is intended for localhost development and
    is loudly warned about at startup — it must never be the state of a deployed pod.
    """
    if not API_TOKEN:
        return
    expected = f"Bearer {API_TOKEN}"
    if authorization != expected:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid or missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
