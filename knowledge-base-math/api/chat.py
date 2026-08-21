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
_GROUNDED_RULES = """- Context from the student's uploaded documents is provided below. Answer from it.
- If the context does not cover part of the question, say so explicitly rather than filling the gap silently.
- Do not write a source list or citation of your own: the server appends the exact one below your answer.

Conversation so far:
{history}"""

_GENERAL_RULES = """- No relevant material was found in the student's uploaded documents, so answer from your own expertise.
- Do not claim or imply that any uploaded document supports what you say, and do not cite sources.

Conversation so far:
{history}"""

# Provenance is written by the server, NOT requested from the model.
# Measured: asked to state its provenance verbatim, deepseek-math-7b-rl ignored the
# instruction and answered directly ("The Battle of Hastings took place in 1066 AD.").
# That is consistent with what EVALUATION.md already says about the model — it is a
# solver, not an instruction-follower. Provenance is a fact the server knows from
# `decide_mode`, so making it depend on the generator's compliance would be both
# unreliable and untestable. Emitting it deterministically also means the honesty of
# the label is guaranteed rather than merely measured.
#
# It is one `Sources:` line rather than a sentence of explanation because the student
# is already looking at the answer that came out of their own documents — the line
# only has to name which document.
#
# In `general` mode there is NO footer at all. A line reading "Sources: general
# knowledge" is not provenance, it is the absence of provenance spelled out, and
# printing it under every unanswered-from-documents reply trained the eye to skip the
# footer entirely — which is the one thing it must not do when it *does* name a file.
# Provenance stays a server fact regardless: `mode` is on the `sources` and `done` SSE
# frames, so a client that wants to say something about general-knowledge answers has
# what it needs without the server writing a line into the answer text.

# Same reasoning as the sources footer, for the same reason: the server knows the
# answer was cut off (Ollama says so, via done_reason == "length"), and the model cannot
# be relied on to say it. Appended — not prepended — because it is a fact about the end
# of the answer, and because prepending it would change the prompt prefix that the next
# turn's KV cache depends on.
TRUNCATION_MARKER = (
    "\n\n_This answer was cut off at the length limit. "
    'Ask "please continue" for the rest._'
)


# How far into the stream a would-be echo may diverge before we stop believing it was
# ever an echo. Divergence at character 0-23 means the model simply continued (see
# PrefillEcho); divergence at character 200 of a 350-character prefill means something
# stranger, and that is the case worth abandoning.
#
# 24 characters is short enough that a genuine continuation almost never collides with
# it by chance, and long enough not to be fooled by a shared opening word.
ECHO_PROBE_CHARS = 24


class PrefillEcho:
    """Reconciles the two ways Ollama can resume a generation, without duplicating text.

    To continue a truncated answer — or to hand back a tool result mid-answer, which is
    the same mechanism (kbm/tools/tir.py) — we send the partial text back as a trailing assistant
    message. Ollama treats that as a *prefill* and the model writes on from it. What
    Ollama does with the prefill ITSELF turns out to depend on the model's chat template,
    and both behaviours are live in this repo:

        deepseek-math-7b-rl  replays the prefill byte-for-byte at the head of the
                             stream, then continues. Forwarded unfiltered, the student
                             sees the first half of the answer twice.
        qwen2 / ChatML       emits nothing but the continuation. Prefill "…are 2, 3"
                             came back as ", and 5, followed by 7 and 11." — diverging
                             at index 0, because there was never an echo to match.

    Measured locally on ollama with both models, 2026-08-20. This class used to assume
    the first behaviour and treat its absence as a fault: anything that was not a
    byte-identical echo set `mismatch` and emitted NOTHING for the rest of the pass. On
    a Qwen model that silently discards every continuation and every tool round — the
    answer just stops, with no error anywhere. So the three cases are now distinguished:

        full echo               strip it, emit what follows          (deepseek)
        diverges immediately    there was no echo; emit everything   (qwen / ChatML)
        diverges deep in        genuinely unexpected; abandon        (mismatch)

    The third case is the original conservative path, kept for what it was always for: a
    build that restates the answer in different words, or retokenisation at the boundary.
    It emits nothing and the caller abandons the continuation, keeping the first-pass
    answer and labelling it truncated. Degrading to a shorter honest answer is always
    correct; emitting duplicated text never is.

    Note that a *restatement* cannot reach the second case. Text that restates the answer
    from the beginning IS a prefix of the prefill, so it matches for far longer than
    ECHO_PROBE_CHARS and lands in the third. Divergence at character 0 is positive
    evidence that no echo was attempted, not merely the absence of one.
    """

    def __init__(self, prefill: str) -> None:
        self._prefill = prefill
        self._buf = ""
        self._done = not prefill      # nothing to strip on the first pass
        self.mismatch = False
        self.echoed: bool | None = None   # which behaviour this pass saw; None = no prefill

    def feed(self, text: str) -> str:
        """Return the portion of `text` that is genuinely new."""
        if self.mismatch:
            return ""
        if self._done:
            return text

        self._buf += text

        # The whole prefill came back: strip it, pass on whatever trails it.
        if self._buf.startswith(self._prefill):
            self._done = True
            self.echoed = True
            buf, self._buf = self._buf, ""
            return buf[len(self._prefill):]

        # Still consistent with an echo in progress: keep swallowing, decide later.
        if self._prefill.startswith(self._buf):
            return ""

        # Diverged. Where it diverged is what tells us which behaviour this is.
        shared = _common_prefix_len(self._buf, self._prefill)
        if shared < ECHO_PROBE_CHARS:
            self._done = True
            self.echoed = False
            buf, self._buf = self._buf, ""
            return buf

        self.mismatch = True
        return ""


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# Below this length a question is not distinctive enough to recognise as an echo:
# "hi" is a prefix of half the greetings a model might open with, and swallowing it
# would damage a perfectly good answer to save nothing.
ECHO_MIN_CHARS = 20

# What the student gets instead of an empty bubble. Reached when the model produced
# nothing, or produced only a restatement of the question, which comes to the same
# thing from the student's side.
NO_ANSWER_TEXT = (
    "_No answer came back this time — try asking the question a different way._"
)


def _norm(text: str) -> str:
    """Whitespace- and case-insensitive form, for comparing text to the question."""
    return " ".join(text.split()).casefold()


class QuestionEcho:
    """Drops a verbatim restatement of the question from the head of an answer.

    Reported from the running app: with no documents ingested, an answer came back that
    was the student's own question and nothing else. It is not reproducible here — a
    dozen prompts through the shipped general-mode path all answered normally — so this
    is a guard on the symptom, not a fix for a diagnosed cause. It costs a good answer
    nothing: the only text it can ever remove is an exact copy of the question.

    Related to PrefillEcho but the opposite contract, and they must not be merged.
    PrefillEcho *expects* the prefix and treats its absence as a reason to abandon the
    continuation. This one expects nothing: it holds text back only while that text is
    still a prefix of the question, and flushes everything the moment it diverges — which
    for a real answer is within a character or two.

    A question restated *inside* a longer answer is left alone. Opening with "What is a
    derivative? A derivative is..." is how a teacher talks, and only the leading copy is
    dropped; text after it streams through untouched.
    """

    def __init__(self, question: str) -> None:
        self._question = question
        self._buf = ""
        # Short questions are not distinctive enough to match on — pass everything.
        self._done = len(question.strip()) < ECHO_MIN_CHARS
        self.fired = False

    def feed(self, text: str) -> str:
        """Return the part of `text` that is safe to show the student."""
        if self._done:
            return text

        self._buf += text
        nq = _norm(self._question)
        nb = _norm(self._buf)

        # Still on track to be the question: hold it back and wait for more.
        if nq.startswith(nb):
            return ""

        self._done = True
        cut = self._echo_cut()
        if cut is None:
            # Diverged — never was an echo. Release everything held back.
            return self._flush_buf()
        self.fired = True
        # The echo, plus whatever punctuation and whitespace trailed it, is dropped;
        # a real answer that followed it streams on from here.
        return self._buf[cut:].lstrip(" .:\n")

    def flush(self) -> str:
        """Whatever is still held back when the stream ends.

        The model can stop part-way through a copy of the question, which is a prefix
        and therefore still buffered. Dropping it silently would turn a bad answer into
        no answer at all, so it is released — the empty-answer path is what handles the
        case where there was never anything else.
        """
        if self._done:
            return ""
        self._done = True
        if _norm(self._buf) == _norm(self._question):
            self.fired = True
            return ""
        return self._flush_buf()

    def _flush_buf(self) -> str:
        buf, self._buf = self._buf, ""
        return buf

    def _echo_cut(self) -> int | None:
        """Index just past a verbatim copy of the question at the head of the buffer.

        Scans rather than slicing at `len(question)` because normalisation moves the
        boundary: the model's copy can differ from the original in whitespace and case
        and still be the same sentence.
        """
        nq = _norm(self._question)
        for i in range(1, len(self._buf) + 1):
            if _norm(self._buf[:i]) == nq:
                return i
        return None


_FENCE = "```"
# Longest a language tag can be before we stop waiting for the newline that ends it.
# "python" is six; the slack covers "python3" and stray whitespace without stalling the
# stream on a line of prose that merely happens to start with three backticks.
_TAG_LOOKAHEAD = 16


def _partial_fence_tail(text: str) -> int:
    """Length of the trailing run of backticks that might be the start of a fence.

    The stream arrives in tokens, and "``" is a perfectly ordinary way for one to end.
    Emitting it and then discovering the next token was "`python" would put two stray
    backticks in the student's answer, so a possible fence start is always held back.
    """
    for n in range(len(_FENCE) - 1, 0, -1):
        if text.endswith(_FENCE[:n]):
            return n
    return 0


class CodeFenceFilter:
    """Keeps ```python blocks out of what the student sees, while the model still writes them.

    Third in the family, and the same contract as PrefillEcho and QuestionEcho: it
    filters what is SHOWN and never touches `generated`. The transcript the model is
    handed back has to contain its own code verbatim — that is the TIR protocol (kbm/tools/tir.py)
    and it is also what PrefillEcho matches against on the next round.

    Why hide it at all: the reader is a family member asking about a derivative. They
    want the reasoning and the number. Eight lines of sympy in the middle of the
    explanation is something they have to learn to skip, and the thing they must NOT
    learn to skip is the sources footer at the bottom. The computed *result* is still
    shown (see tool_output_marker) — hiding the arithmetic entirely would make a wrong
    sandbox answer indistinguishable from a hallucination, which is the failure this
    whole feature exists to remove. KBM_SHOW_TOOL_CODE=1 turns the filter off.

    Only a `python`-tagged fence is hidden. A bare ``` fence passes through untouched:
    the model may legitimately quote a matrix or a table, and a filter that swallowed
    any fenced block would eat those too.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._hiding = False
        # A hidden block ends with "```\n", and that newline closes the block rather
        # than opening the prose after it — left in, every computation leaves a blank
        # line behind. It cannot be dropped at the moment the fence closes, because with
        # small chunks the newline has not arrived yet: whether the output had a stray
        # blank line then depended on how Ollama happened to split its tokens. So the
        # intent is recorded and applied whenever the newline does turn up.
        self._eat_newline = False

    def feed(self, text: str) -> str:
        self._buf += text
        out = []

        while True:
            if self._eat_newline and self._buf:
                if self._buf.startswith("\n"):
                    self._buf = self._buf[1:]
                self._eat_newline = False

            if self._hiding:
                close = self._buf.find(_FENCE)
                if close < 0:
                    # Hold back only what could be the start of the closing fence.
                    keep = _partial_fence_tail(self._buf)
                    self._buf = self._buf[len(self._buf) - keep:] if keep else ""
                    break
                self._buf = self._buf[close + len(_FENCE):]
                self._hiding = False
                self._eat_newline = True
                continue

            open_at = self._buf.find(_FENCE)
            if open_at < 0:
                keep = _partial_fence_tail(self._buf)
                cut = len(self._buf) - keep
                out.append(self._buf[:cut])
                self._buf = self._buf[cut:]
                break

            head, rest = self._buf[:open_at], self._buf[open_at:]
            newline = rest.find("\n")
            if newline < 0:
                if len(rest) < len(_FENCE) + _TAG_LOOKAHEAD:
                    # The tag is still arriving; emit what precedes it and wait.
                    out.append(head)
                    self._buf = rest
                    break
                newline = len(rest)

            tag = rest[len(_FENCE):newline].strip().lower()
            out.append(head)
            self._buf = rest[newline + 1:]
            if tag in ("python", "python3", "py"):
                self._hiding = True
            else:
                # Not ours — put the fence line back on the wire exactly as it came.
                out.append(rest[:newline + 1])
            continue

        return "".join(out)

    def flush(self) -> str:
        """Whatever is still held when the stream ends.

        An unterminated ```python block is dropped: the model was cut off mid-program,
        and half a program shown to a student is worse than nothing. The truncation
        marker is what tells them the answer stopped early.
        """
        if self._hiding:
            self._buf = ""
            return ""
        buf, self._buf = self._buf, ""
        return buf


def tool_output_marker(result, ok: bool, after: str = "") -> str:
    """What the student is shown in place of a raw ```output block.

    Two strings for one event, exactly as Job.detail and Job.diagnostic are (CLAUDE.md):
    this is the sentence for the family member, while the sandbox's real text — a
    traceback line, a refused import, a timeout — goes into the model's transcript, the
    telemetry record and the log. "Surface the error" has never meant "show the student
    the traceback", and it does not start meaning it here.

    On failure the student gets a neutral note rather than silence, because the answer
    above and below it will read as though a number arrived; saying that it did not is
    what keeps the two halves honest.

    `after` is the answer text emitted so far. The marker needs a blank line above it to
    be its own paragraph, and the model has usually already written one or two newlines
    before its code block — so the separator is computed from what is actually there
    rather than assumed, which is the difference between one paragraph break and a gap.
    """
    if not ok:
        body = "_The calculation didn't run — working it through by hand instead._"
    else:
        text = str(result).strip()
        if not text:
            return ""
        body = (f"_Computed:_\n\n{_FENCE}\n{text}\n{_FENCE}"
                if "\n" in text else f"_Computed:_ `{text}`")

    lead = "\n\n"[: max(0, 2 - (len(after) - len(after.rstrip("\n"))))]
    return f"{lead}{body}\n\n"


def search_marker(query: str, n_found: int, after: str = "") -> str:
    """What the student is shown when the model searches their documents mid-answer.

    Native tool calls are invisible in a way TIR never was: with TIR the model's
    ```python block was at least IN the token stream and merely filtered out, whereas a
    tool call lives in a structured field the stream never carries. Without this, the
    student watches a still bubble for several seconds while retrieval and a re-prefill
    happen, and then the answer changes direction for no visible reason.

    The query is shown, not just the fact of a search. It is the model's words, not the
    student's, and it is how someone notices that an answer came from the wrong chapter
    because the search asked for the wrong thing — the same argument that keeps the
    computed result visible in tool_output_marker rather than hidden as noise.

    `after` computes the separator from the text already emitted, exactly as
    tool_output_marker does; see its docstring.
    """
    shown = query.strip()
    if len(shown) > 60:
        shown = shown[:60].rsplit(" ", 1)[0] + "…"
    if n_found:
        body = (f'_Searched your documents for "{shown}" — '
                f"found {n_found} passage{'s' if n_found != 1 else ''}._")
    else:
        body = f'_Searched your documents for "{shown}" — nothing relevant._'
    lead = "\n\n"[: max(0, 2 - (len(after) - len(after.rstrip("\n"))))]
    return f"{lead}{body}\n\n"


def tool_code_marker(code: str, after: str = "") -> str:
    """The operator escape hatch (KBM_SHOW_TOOL_CODE) for native tool calls.

    CodeFenceFilter cannot serve here and inverting it would not help: in TIR the code
    is in the token stream and the filter's job is to REMOVE it, while in agent mode the
    code was never in the stream at all — it arrives in tool_calls[i]["args"]. So the
    same environment variable needs its own renderer, or it silently stops working for
    the newer arm, which is the worst outcome for a debugging switch.
    """
    body = code.strip()
    if not body:
        return ""
    lead = "\n\n"[: max(0, 2 - (len(after) - len(after.rstrip("\n"))))]
    return f"{lead}{_FENCE}python\n{body}\n{_FENCE}\n\n"


_MODE_RULES = {
    Mode.GROUNDED: _GROUNDED_RULES,
    Mode.GENERAL: _GENERAL_RULES,
}


def system_prompt(mode: Mode, tir: bool = False, tools: bool = False) -> str:
    """The system message, with the tool rules spliced in when the model can use them.

    The tool block goes AFTER the teaching style and BEFORE the mode block, which keeps
    two properties the rest of the file depends on. `history` stays the last thing in
    the message, so it can grow by appending without moving anything in front of it; and
    the two modes still share every token up to the mode block, so alternating between
    grounded and general mid-conversation reuses the cached prefix instead of
    re-prefilling (LATENCY.md).

    Four literal prompts would have been the same four strings with four places to forget
    to change one — which is why SYSTEM_PROMPTS below is derived from this function rather
    than written out beside it.

    `tir` and `tools` are the two tool protocols and are mutually exclusive — kbm/config.py
    enforces that in one line, upstream of every caller, so this function does not
    re-decide it. Both default False, so every existing call returns a byte-identical
    string and the deepseek path is untouched.

    A note on `tools` and `mode`, because they interact in a way TIR never did.
    agent.search_documents can retrieve DURING the answer, which means the general-mode
    rules ("No relevant material was found in the student's uploaded documents") can be
    falsified after this prompt is frozen. The mode block still goes in — it is the right
    description of what the model was handed up front, and swapping it for a
    mode-independent block would throw away the grounding instruction in the case that
    matters most. agent.AGENT_RULES carries the override instead, in its last line.
    """
    from kbm.tools.tir import TIR_RULES

    head = _TEACHING_STYLE
    if tir:
        head += TIR_RULES
    if tools:
        from kbm.tools.agent import AGENT_RULES

        head += AGENT_RULES
    return head + _MODE_RULES[mode]


# The no-tool prompts, by mode — what the service sent before TIR existed, and what the
# eval reads when it wants "the prompt the model is given" without a generator in hand.
# DERIVED, not a second copy: an independent literal here would drift from system_prompt()
# silently, and the drift would only show up as a quietly different prompt in the eval than
# in production. Same rule as kbm/retrieval.py.
SYSTEM_PROMPTS = {mode: system_prompt(mode) for mode in _MODE_RULES}

# `context` carries its own trailing blank line (see build_context) rather than the
# template hard-coding one. In `general` mode the context is empty, and a template with
# the blank line baked in handed the model a human turn that opened with two blank lines
# before "Question:" — a continuation prompt with nothing above it to continue.
HUMAN_PROMPT = "{context}Question: {input}"

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
    from kbm.retrieval import chunk_id

    return [
        Source(
            source=os.path.basename(doc.metadata.get("source", "unknown")),
            chunk_id=chunk_id(doc),
            score=float(score),
            preview=doc.page_content[:PREVIEW_CHARS],
        )
        for doc, score in results
    ]


def merge_sources(existing: list[Source], new: list[Source]) -> list[Source]:
    """Fold a mid-answer search's sources into the ones already announced.

    Only agent mode produces this: retrieval used to run exactly once per request, so
    there was never a second list to merge. Deduped on `chunk_id` and NOT re-sorted by
    score — the order is "when the answer leaned on it", which is what the footer reads
    top-down, and re-ranking a merged list by score would put a late chunk above the
    context the answer actually opened from.

    Lives here rather than in kbm/tools/agent.py because this file already owns `Source` and is
    already pure. kbm/tools/agent.py importing api.schemas would invert the dependency that keeps
    kbm/tools/tir.py and kbm/retrieval.py importable from the eval harness.
    """
    seen = {s.chunk_id for s in existing}
    return existing + [s for s in new if s.chunk_id not in seen]


def select_context(results: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
    """The retrieved chunks that actually earn a place in the prompt, best first.

    Retrieval always returns TOP_N chunks — that is what a top-N cut means, not a claim
    that all N are relevant. Measured on the shipping pipeline, a top-5 routinely looks
    like `0.9956  0.5881  0.0520  0.0089  0.0068`: one right answer, one near-miss, and
    three chunks the reranker is actively telling us are unrelated. Every one of them
    used to be pasted into the prompt under a "[Source: …]" heading.

    So the floor is applied per chunk, not just to the best one. The same number does
    both jobs, and that is deliberate rather than convenient: "is this chunk about what
    was asked" is one question, and the corpus-level answer is just whether *any* chunk
    passes it — which is why decide_mode below is now a non-empty check.

    Deliberately NOT a softmax/top-p (nucleus) selection over the scores. Nucleus is
    relative, so a chunk is dropped for having a better sibling rather than for being
    bad: measured on the same queries, `p=0.9` discards chunks scoring 0.79 and 0.77
    because one chunk scored 0.99, and it never keeps five chunks even when all five are
    good. Worse, it cannot abstain at all — five garbage chunks softmax to exactly the
    same distribution as five excellent ones, because normalising throws away the
    absolute scale that tells them apart. See evaluation/calibrate_floor.py.
    """
    return [(doc, score) for doc, score in results if score >= RELEVANCE_FLOOR]


def decide_mode(selected: list[tuple[Document, float]], has_index: bool) -> Mode:
    """Grounded only if at least one chunk cleared the floor.

    Three ways to land in `general` mode, and they are genuinely different situations
    that the user experiences identically: no index at all, an index that returned
    nothing, and an index whose every hit fell below the relevance floor. The last is
    the one that matters — it is the case where the old code would hand the model five
    irrelevant chunks and let it answer as though they were sources.

    Takes the ALREADY-SELECTED list, so the mode and the context can never disagree
    about which chunks counted. RELEVANCE_FLOOR is calibrated; see settings.py.
    """
    return Mode.GROUNDED if has_index and selected else Mode.GENERAL


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
    # Trailing blank line, so HUMAN_PROMPT does not have to supply one that `general`
    # mode would then emit with nothing in front of it.
    return "\n\n".join(parts) + "\n\n"


def sources_footer(sources: list[Source], mode: Mode) -> str:
    """The provenance line appended to a grounded answer — and nothing otherwise.

    Deduped because the top-N chunks usually come from the same document, and a footer
    that named `calculus.pdf` five times would say nothing five times. Ordered by
    retrieval rank rather than sorted: the first name is the document the answer leans
    on hardest. Scores and chunk ids stay off it — they are in the `sources` SSE frame
    for any client that wants to show them, and they mean nothing to a student.

    Returns "" in `general` mode: the footer exists to name documents, and there are
    none to name. No sources means nothing grounded the answer whatever `mode` claims,
    so that is the empty case too. Callers must not emit an empty token frame.
    """
    if mode is Mode.GENERAL or not sources:
        return ""
    names = list(dict.fromkeys(s.source for s in sources))
    return "\n\n_Sources: " + ", ".join(names) + "_"
