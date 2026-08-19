"""
api/chat.py - Prompt construction, history trimming, and answer-provenance mode.

Ported from app.py. The prompt ordering and the history-trimming shape are latency
fixes with measurements behind them (LATENCY.md) — read the comments before changing
either. Everything here is pure: no I/O, no model handles, so it is testable and the
eval can build exactly the prompt that ships.
"""

import os

from langchain_core.documents import Document

from .schemas import Message, Mode, Source
from .settings import RELEVANCE_FLOOR

# Prompt order is load-bearing for latency: static text → history → context → question.
# Ollama caches the KV of the longest common prompt *prefix* between consecutive requests.
# Everything up to the first token that changes is free; everything after it is re-prefilled.
# The static block never changes and `history` only ever grows by appending, so both stay
# cached across turns. `context` is fresh every query, which is exactly why it must come
# last — it lives in the human message, after the system message. Putting it before the
# history (as app.py once did) invalidated the cache at token ~150 and re-prefilled the
# whole ~3.4k-token prompt every single turn. See LATENCY.md.
_TEACHING_STYLE = """You are an expert mathematician and dedicated teacher. Your deep love for mathematics drives you to help students not just find answers, but truly understand the underlying concepts and develop their own mathematical thinking.

When answering:
- Don't just solve the problem — explain the reasoning behind each step so the student understands why, not just how.
- If a student makes a conceptual error, gently point it out and guide them toward the correct understanding.
- Encourage curiosity: point out interesting patterns, connections to other concepts, or follow-up questions worth thinking about.
- Show your working step by step, but match the length of the answer to the question: a conceptual "why" question wants a short, clear explanation, not a full derivation. Stop once the student has what they asked for.
- Use LaTeX for all equations (e.g. $x^2$, \\frac{{a}}{{b}}).
"""

# The two modes differ only in this trailing block, and it is deliberately the *last*
# part of the static prefix: two prompts that share a prefix also share the KV cache
# for that prefix, so alternating modes mid-conversation costs less than a full reload.
_GROUNDED_RULES = """- Context from the student's uploaded documents is provided below. Answer from it and cite the source you used.
- If the context does not cover part of the question, say so explicitly rather than filling the gap silently.

Conversation so far:
{history}"""

_GENERAL_RULES = """- No relevant material was found in the student's uploaded documents, so answer from your own expertise.
- Do not claim or imply that any uploaded document supports what you say, and do not cite sources.

Conversation so far:
{history}"""

# The provenance marker is prepended by the server, NOT requested from the model.
# Measured: asked to open with this line verbatim, deepseek-math-7b-rl ignored the
# instruction and answered directly ("The Battle of Hastings took place in 1066 AD.").
# That is consistent with what EVALUATION.md already says about the model — it is a
# solver, not an instruction-follower. Provenance is a fact the server knows from
# `decide_mode`, so making it depend on the generator's compliance would be both
# unreliable and untestable. Emitting it deterministically also means the honesty of
# the label is guaranteed rather than merely measured.
GENERAL_MODE_MARKER = "_Not from your uploaded documents — answering from general knowledge._\n\n"

# Same reasoning as GENERAL_MODE_MARKER, for the same reason: the server knows the
# answer was cut off (Ollama says so, via done_reason == "length"), and the model cannot
# be relied on to say it. Appended — not prepended — because it is a fact about the end
# of the answer, and because prepending it would change the prompt prefix that the next
# turn's KV cache depends on.
TRUNCATION_MARKER = (
    "\n\n_This answer was cut off at the length limit. "
    'Ask "please continue" for the rest._'
)


class PrefillEcho:
    """Strips the assistant-prefill that Ollama echoes back when resuming a generation.

    To continue a truncated answer we send the partial text back as an assistant
    message and let the model keep writing. Ollama treats a trailing assistant message
    as a *prefill* — which is what makes seamless continuation possible at all — but it
    replays the whole prefill at the head of the stream before emitting anything new.
    Forwarded unfiltered, the student sees the first half of the answer twice.

    Verified on ollama 0.32.6 with deepseek-math-7b-rl: the echo is byte-identical to
    what was sent, including a leading space, and arrives in the first one or two
    chunks. That is an observation, not a guarantee — hence `mismatch`.

    On mismatch (a different Ollama build that opens a fresh turn instead of prefilling,
    or retokenisation at the boundary) this emits NOTHING and the caller abandons the
    continuation, keeping the first-pass answer and labelling it truncated. Degrading to
    a shorter honest answer is always correct; emitting duplicated text never is.
    """

    def __init__(self, prefill: str) -> None:
        self._prefill = prefill
        self._buf = ""
        self._done = not prefill      # nothing to strip on the first pass
        self.mismatch = False

    def feed(self, text: str) -> str:
        """Return the portion of `text` that is genuinely new."""
        if self.mismatch:
            return ""
        if self._done:
            return text

        self._buf += text
        # Still a prefix of what we sent: keep swallowing, emit nothing yet.
        if len(self._buf) < len(self._prefill):
            if not self._prefill.startswith(self._buf):
                self.mismatch = True
            return ""

        if not self._buf.startswith(self._prefill):
            self.mismatch = True
            return ""

        self._done = True
        return self._buf[len(self._prefill):]


SYSTEM_PROMPTS = {
    Mode.GROUNDED: _TEACHING_STYLE + _GROUNDED_RULES,
    Mode.GENERAL: _TEACHING_STYLE + _GENERAL_RULES,
}

HUMAN_PROMPT = """{context}

Question: {input}"""

# History is trimmed in blocks, not one message at a time — see history_window().
# These count *messages* (a turn is two: student + tutor).
HISTORY_KEEP = 8    # smallest the window is ever trimmed back to
HISTORY_BLOCK = 8   # the window start only ever moves in steps of this

PREVIEW_CHARS = 200


def history_window(history: list[Message]) -> list[Message]:
    """The slice of history sent to the model, trimmed in blocks rather than per turn.

    A plain `history[-N:]` sliding window drops the oldest message on *every* turn once
    it is full. That shifts the start of the history block, which is the one thing the
    prompt KV cache cannot survive (the cache only reuses a common *prefix*), so every
    turn would pay a full re-prefill — measured at 13.4s by turn 5.

    Trimming to a block boundary instead means the window start only moves once every
    HISTORY_BLOCK messages. In between, history grows purely by appending and stays
    cached. The cost is that we sometimes carry a few more messages than the minimum,
    which is cheap: those tokens are cached, the alternative is recomputing all of them.

    There is deliberately no separate "max length" constant. Until history reaches
    HISTORY_KEEP + HISTORY_BLOCK messages the division below is 0, so the window starts
    at 0 and keeps everything — the "grow freely at first" behaviour falls out of the
    arithmetic. Adding a max that disagreed with this grid only made the first trim
    lurch twice in consecutive turns.

    max(0, ...) guards Python's floor division on negatives: (2 - 8) // 8 == -1, and a
    negative start would silently index from the end and restore per-turn sliding.
    """
    start = max(0, ((len(history) - HISTORY_KEEP) // HISTORY_BLOCK) * HISTORY_BLOCK)
    return history[start:]


def format_history(history: list[Message]) -> str:
    lines = []
    for m in history_window(history):
        role = "Student" if m.role.value == "user" else "Tutor"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines) if lines else "None yet."


def to_sources(results: list[tuple[Document, float]]) -> list[Source]:
    from retrieval import chunk_id

    return [
        Source(
            source=os.path.basename(doc.metadata.get("source", "unknown")),
            chunk_id=chunk_id(doc),
            score=float(score),
            preview=doc.page_content[:PREVIEW_CHARS],
        )
        for doc, score in results
    ]


def decide_mode(results: list[tuple[Document, float]], has_index: bool) -> Mode:
    """Grounded only if retrieval actually produced something plausibly relevant.

    Three ways to land in `general` mode, and they are genuinely different situations
    that the user experiences identically: no index at all, an index that returned
    nothing, and an index whose best hit is below the relevance floor. The last is the
    one that matters — it is the case where the old code would hand the model five
    irrelevant chunks and let it answer as though they were sources.

    RELEVANCE_FLOOR is an unmeasured default; see settings.py.
    """
    if not has_index or not results:
        return Mode.GENERAL
    best = max(score for _, score in results)
    return Mode.GROUNDED if best >= RELEVANCE_FLOOR else Mode.GENERAL


def build_context(results: list[tuple[Document, float]], mode: Mode) -> str:
    """The context block for the human message.

    In `general` mode this is empty rather than a sentence telling the model there are
    no documents. The instruction to answer from general knowledge already lives in the
    system prompt, where it is part of the cacheable prefix; repeating it here would put
    per-query-varying text in front of the question for no benefit.
    """
    if mode is Mode.GENERAL:
        return ""
    parts = ["Context from your documents:", ""]
    for doc, _ in results:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)
