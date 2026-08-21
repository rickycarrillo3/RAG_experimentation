"""
api/deps.py - Shared model handles and the auth dependency.

The embedder, reranker and LLM client are loaded once at process start and shared
across requests, exactly as app.py did at module scope. Loading the 2.2GB reranker
per request would dominate every other cost in the system.
"""

import logging
import os
import pickle
import threading

import httpx
from fastapi import Header, HTTPException, status
from langchain_ollama import ChatOllama

import agent
import retrieval
from config import OLLAMA_BASE_URL
from retrieval import BM25_DIR, OLLAMA_MODEL, load_embeddings, load_reranker

from .settings import (
    API_TOKEN,
    KEEP_ALIVE,
    NUM_CTX,
    NUM_PREDICT,
    THINK,
    TIR_ENABLED,
    TOOLS_ENABLED,
)

log = logging.getLogger(__name__)


def _model_supports_tools(model: str) -> bool | None:
    """Does Ollama report a tools template for this model? None if it could not be asked.

    llm_profiles.py exists because a model's capabilities are facts about the weights,
    and KBM_TOOLS is an environment variable — so the two can disagree, and only one of
    them is ever right. Forced onto a completion-only model, `tools` makes Ollama reject
    every /api/chat with a 400 that surfaces as an SSE error frame on every answer, with
    the real cause named only inside an exception string.

    So ask the runtime rather than trusting the table, which is the same lesson
    ERRORS.md 2026-08-20 records for `think=`: assert on what actually goes out, not on
    what the config says. Verified locally —

        t1c/deepseek-math-7b-rl:Q4  ->  ['completion']
        qwen2:7b                    ->  ['completion', 'tools']

    httpx rather than the `ollama` package because httpx is a named dependency and
    `ollama` only arrives transitively via langchain-ollama.
    """
    try:
        r = httpx.post(f"{OLLAMA_BASE_URL}/api/show", json={"model": model}, timeout=10.0)
        r.raise_for_status()
        return "tools" in (r.json().get("capabilities") or [])
    except Exception as e:
        # Deliberately not fatal. startup.sh already waits on Ollama, and turning this
        # probe into a second liveness gate would mean a slow Ollama start looks like a
        # broken model load.
        log.warning("could not ask Ollama whether %s supports tools (%s) — "
                    "trusting the profile", model, e)
        return None


class Models:
    """Lazily-built singletons, so importing this module stays cheap (tests, CLI tools)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.embeddings = None
        self.reranker = None
        self.llm = None
        # The same client with agent.TOOL_SCHEMAS bound, or None when this model does not
        # get the native protocol. Two handles rather than one because binding is not
        # free of consequence: a `tools` array sent to a completion-only model is a 400
        # from Ollama on every single request.
        self.llm_tools = None
        # The EFFECTIVE answer, after the probe below — which is not the same thing as
        # settings.TOOLS_ENABLED. routes.py must read this one.
        self.tools_enabled = False

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
                    # Ollama does not take the window from the model file — it applies
                    # its own default, and shifts the context from the LEFT when a prompt
                    # overflows. The left end is the system prompt, so an over-long turn
                    # (a TIR trace especially, since every round re-sends the whole
                    # transcript) does not error: the tutor persona and the grounding
                    # rules just quietly stop being in the prompt. Stating the model's
                    # real window is what makes that bounded. See config.NUM_CTX.
                    num_ctx=NUM_CTX,
                    # How long Ollama keeps the weights in VRAM after the last request.
                    # Without it, Ollama unloads after 5 min and the second question of
                    # an evening pays a ~10-20s reload.
                    #
                    # This costs nothing either way — the pod bills whether the model is
                    # resident or not. Only KBM_IDLE_STOP_MINUTES affects the bill, and it
                    # is the outer bound: the model cannot outlive the pod, so a keep_alive
                    # longer than the idle window is simply never reached. Setting it
                    # *shorter* is the actual mistake — it unloads while the pod is still
                    # alive and billing, so a question in that gap pays a reload for
                    # nothing. Keep it >= the idle window.
                    #
                    # The exception is VRAM pressure: upload peaks at ~12-13GB with the
                    # generator resident (EVALUATION.md §6), so drop this toward 0 during
                    # ingest if it ever OOMs.
                    keep_alive=KEEP_ALIVE,
                    # Qwen3-style thinking mode. None for every model that has none, and
                    # passing None leaves the request unchanged; False turns it off for
                    # models that default it on, whose <think> block would otherwise be
                    # streamed into the student's answer bubble.
                    #
                    # The kwarg is `reasoning`, NOT `think`. langchain-ollama renames it on
                    # the way to Ollama's `think` field (chat_models._chat_params), and
                    # ChatOllama accepts unknown kwargs without complaining — so `think=`
                    # here constructs cleanly, is never sent, and leaves thinking mode on.
                    reasoning=THINK,
                )
            if TOOLS_ENABLED and self.llm_tools is None:
                supported = _model_supports_tools(OLLAMA_MODEL)
                if supported is False:
                    # Loud, and then carry on serving. An unusable /chat is a worse
                    # outcome than a downgraded one, and the operator who set KBM_TOOLS
                    # needs to be told which of the two facts won.
                    log.error(
                        "KBM_TOOLS is on but Ollama reports no tools template for %s — "
                        "serving with tool calling DISABLED. Pick a model that supports "
                        "tools (qwen3:8b) or unset KBM_TOOLS.", OLLAMA_MODEL,
                    )
                else:
                    self.tools_enabled = True
                    # Bound once, at load, for the same reason the client itself is:
                    # bind_tools() is a pure Runnable wrapper, but building it per
                    # request would re-run schema conversion on every answer.
                    self.llm_tools = self.llm.bind_tools(agent.TOOL_SCHEMAS)
                    log.info("native tool calling enabled for %s (%s)", OLLAMA_MODEL,
                             ", ".join(sorted(agent.TOOL_NAMES)))

    @property
    def ready(self) -> bool:
        # llm_tools is deliberately NOT in here: it is derived, and a model that simply
        # does not support tools would otherwise look like a failed load.
        return all(x is not None for x in (self.embeddings, self.reranker, self.llm))

    @property
    def protocol(self) -> str:
        """Which tool arm is actually live — for /healthz, and for telemetry.

        Reported rather than inferred because it is the product of a profile, two
        environment variables and a runtime probe, and 'which arm ran?' is the first
        question to ask of any answer during the bake-off.
        """
        if self.tools_enabled:
            return "tools"
        return "tir" if TIR_ENABLED else "none"


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
