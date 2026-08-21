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
    tool_ms: float | None = Field(
        None,
        description="Total wall-clock spent in the Python sandbox across all tool rounds. "
                    "The sandbox runs between generation passes, so this is a slice OF "
                    "generate_ms rather than a sibling of it — broken out so a slow answer "
                    "can be attributed to decoding rather than to computing. Sandbox only, "
                    "both tool protocols; mid-answer retrieval is search_ms.",
    )
    search_ms: float | None = Field(
        None,
        description="Wall-clock spent on searches the MODEL asked for mid-answer (agent "
                    "mode). A sibling of tool_ms rather than part of it: folding retrieval "
                    "into a field documented as sandbox time would make every previously "
                    "logged tool_ms incomparable. The upfront retrieval is still reported "
                    "as bm25_ms/dense_ms/rrf_ms/rerank_ms, which now accumulate across "
                    "every search in the request rather than describing a single one.",
    )


# ── SSE event payloads ────────────────────────────────────────────────────────
# Emitted in order: `sources` once, `token` many, `done` once. `error` may replace
# any of them. Each is sent as an SSE frame with a matching `event:` name.
#
# `sources` stays exactly-once even in agent mode, where the model can retrieve again
# part-way through the answer. The frame's job is to fill the dead time BEFORE the first
# token, and by the time a mid-answer search runs that job is done; a second frame would
# double-render in any client written to this contract. The complete list — upfront plus
# anything found later — is on `done`, and `done.late_sources` says how much of it
# arrived after generation started.

class SourcesEvent(BaseModel):
    type: Literal["sources"] = "sources"
    mode: Mode
    sources: list[Source]


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    text: str
    server_marker: bool = Field(
        False,
        description="True for text the server wrote rather than the model — the sources "
                    "footer and the truncation notice. Render it like any other token; the "
                    "flag exists so a client can leave it out of the history it sends back, "
                    "since the model did not write it and should not be shown it as its own "
                    "words. A client that ignores the field still displays a correct answer.",
    )


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
    tool_calls: int = Field(
        0,
        description="Python programs the model ran while answering, under either tool "
                    "protocol (TIR's text blocks or agent mode's native calls). 0 for a "
                    "model with neither, and 0 for a model that did not need to compute "
                    "anything. Non-zero means part of this answer was calculated rather "
                    "than recalled. Searches are counted separately, in `searches`.",
    )
    searches: int = Field(
        0,
        description="Searches the model itself asked for while answering (agent mode). "
                    "The unconditional retrieval every request begins with is NOT counted "
                    "here — this is only the model deciding it needed to look again.",
    )
    late_sources: int = Field(
        0,
        description="How many entries of `sources` were found by a mid-answer search "
                    "rather than by the upfront retrieval. `mode` is deliberately not "
                    "revised when this is non-zero: it records the provenance decision "
                    "the calibrated relevance floor made up front, on the question as "
                    "asked. This field is how a client sees that the answer became better "
                    "grounded than that decision, without the label quietly changing "
                    "meaning between one answer and the next.",
    )
    tool_errors: int = Field(
        0,
        description="Of those programs, how many failed — refused by the sandbox policy, "
                    "raised, or timed out. The model sees the error text and usually "
                    "recovers, so this is a quality signal rather than a request failure.",
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
    diagnostic: str | None = Field(
        None,
        description="Technical cause — exception text, tool errors, install URLs. For "
                    "operators and logs. NOT for display: `detail` is the user-facing "
                    "sentence, and clients should render that instead.",
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
    protocol: Literal["tools", "tir", "none"] = Field(
        "none",
        description="Which tool arm is actually live: native tool calling, TIR's text "
                    "protocol, or neither. The EFFECTIVE value, after the startup probe — "
                    "KBM_TOOLS can ask for a protocol the model has no template for, and "
                    "the server downgrades rather than failing every answer, so this is "
                    "the only place that fact is visible.",
    )
    model_loaded: bool = Field(..., description="Whether Ollama currently holds the generator resident")
    retrieval_ready: bool
