"""
api/schemas.py - Request/response models.

This is the contract the TypeScript frontend will be generated from, so field names
are chosen deliberately and changing one is a breaking change, not a rename.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Mode(str, Enum):
    """Where an answer's content came from.

    The distinction is the whole point of the provenance work: without it a grounded
    answer and a confident hallucination are indistinguishable to the user, to the
    frontend, and to the faithfulness eval. `grounded` answers are the only ones the
    faithfulness metric is defined over.
    """

    GROUNDED = "grounded"  # answered from retrieved document chunks, cited
    GENERAL = "general"    # answered from the model's own knowledge, labelled as such


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    """One turn.

    `history` is client-held. Gradio kept it in a `gr.State`; an HTTP API cannot, so
    the client sends the conversation back each turn and the server trims it (see
    api/chat.py:history_window — the trimming shape is a latency fix, not a style
    choice). History must contain only prior turns, not the current question.
    """

    user: str = Field(..., description="Username; selects the isolated index. Lowercased server-side.")
    message: str = Field(..., min_length=1)
    history: list[Message] = Field(default_factory=list)


class Source(BaseModel):
    source: str = Field(..., description="Basename of the document the chunk came from")
    chunk_id: str
    score: float = Field(..., description="Cross-encoder relevance score (raw logit; higher is better)")
    preview: str = Field(..., description="First ~200 chars of the chunk, for UI display")


class Timings(BaseModel):
    """Per-stage wall-clock in milliseconds. Mirrors RetrievalResult.timings plus generation."""

    bm25_ms: float | None = None
    dense_ms: float | None = None
    rrf_ms: float | None = None
    rerank_ms: float | None = None
    ttft_ms: float | None = Field(None, description="Time to first generated token")
    generate_ms: float | None = None


# ── SSE event payloads ────────────────────────────────────────────────────────
# Emitted in order: `sources` once, `token` many, `done` once. `error` may replace
# any of them. Each is sent as an SSE frame with a matching `event:` name.

class SourcesEvent(BaseModel):
    type: Literal["sources"] = "sources"
    mode: Mode
    sources: list[Source]


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    text: str


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    mode: Mode
    answer: str
    sources: list[Source]
    timings: Timings
    event_id: str = Field(..., description="Telemetry id; pass to POST /feedback to rate this answer")
    truncated: bool = Field(
        False,
        description="Generation hit the token limit and could not be finished within "
                    "KBM_MAX_CONTINUATIONS. `answer` already carries a visible marker, so "
                    "a client that ignores this field still shows an honest answer.",
    )
    continuations: int = Field(
        0, description="Continuation passes used to finish the answer (0 = finished first try)"
    )


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str


# ── Upload / jobs ─────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Job(BaseModel):
    """Ingest is minutes-long (Marker OCR), so upload returns a handle, not a result."""

    job_id: str
    status: JobStatus
    filename: str
    user: str
    detail: str = ""
    n_chunks: int | None = None
    # Added rather than encoded as a new JobStatus member on purpose: adding an enum
    # variant breaks an exhaustive switch in the generated TS client, while an optional
    # field is ignored by clients that predate it. A half-good ingest is `status=done`
    # plus `degraded=true`, never a fifth status.
    extractor: str | None = Field(
        None,
        description="Which extractor produced the text: 'marker' (LaTeX-faithful) or "
                    "'pymupdf4llm' (equations flattened to Unicode)",
    )
    degraded: bool = Field(
        False,
        description="Indexed, but at lower fidelity than intended — currently means Marker "
                    "failed and pymupdf4llm was used, so the chunks contain no LaTeX",
    )
    stage: str | None = Field(
        None, description="Pipeline stage in progress, or the stage that failed: extract | chunk | index"
    )


class UserStatus(BaseModel):
    user: str
    has_index: bool
    n_chunks: int = 0
    sources: list[str] = Field(default_factory=list)


class Feedback(BaseModel):
    event_id: str
    rating: Literal["up", "down"]
    note: str = ""


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    model: str
    model_loaded: bool = Field(..., description="Whether Ollama currently holds the generator resident")
    retrieval_ready: bool
