"""
api/routes.py - HTTP surface.

The retrieval pipeline is *imported* from kbm/retrieval.py and never reimplemented here.
CLAUDE.md is explicit about why: a second copy drifts, and then the CLI, the web UI
and the eval quietly stop describing the same system.
"""

import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor

import anyio
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from kbm import retrieval, telemetry
from kbm.retrieval import OLLAMA_MODEL
from kbm.tools import agent, sandbox, tir

from . import chat as chatmod
from .deps import has_index, index_summary, models, normalize_user, require_token
from .schemas import (
    ChatRequest,
    DoneEvent,
    ErrorEvent,
    Feedback,
    Health,
    Job,
    JobStatus,
    Mode,
    SourcesEvent,
    TokenEvent,
    UserStatus,
)
from .settings import (
    MAX_CONTINUATIONS,
    SANDBOX_TIMEOUT_S,
    SHOW_TOOL_CODE,
    TIR_ENABLED,
)

log = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_token)])

# Ingest is GPU-heavy (Marker/Surya) and must not run concurrently with itself, or two
# uploads will fight over VRAM that the generator and reranker are already holding —
# EVALUATION.md §6 measures upload as the tightest moment in the memory budget.
_ingest_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
_jobs: dict[str, Job] = {}

# Pipeline stages in words a family member can act on. The stage name itself ("chunk")
# is ours, not theirs.
_STAGE_BLAME = {
    "extract": "the text could not be read out of the PDF",
    "chunk": "the text could not be split into sections",
    "index": "the search index could not be built",
}

# Set by the idle-stop watchdog; every /chat touches it. See ops/idle_stop.py.
last_chat_at = time.monotonic()


def _add_retrieval_timings(timings: dict, stage_seconds: dict) -> None:
    """Fold one retrieval's per-stage seconds into the request's millisecond totals.

    ADDING, not assigning. Retrieval used to run exactly once per request, so `update`
    was correct; in agent mode the model can search again mid-answer, and an assignment
    would report the last search's cost as though it were the whole request's — an answer
    that searched three times would look as cheap as one that searched once.
    """
    for stage, seconds in stage_seconds.items():
        key = f"{stage}_ms"
        timings[key] = round(timings.get(key, 0.0) + seconds * 1000, 2)


def _sse(event: str, payload) -> str:
    return f"event: {event}\ndata: {payload.model_dump_json()}\n\n"


# ── Chat ──────────────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(req: ChatRequest):
    """Stream one answer as Server-Sent Events: `sources`, then `token`*, then `done`.

    Sources are emitted *before* generation starts, not bundled into the final payload,
    so the UI can show what it found during the seconds the model spends decoding —
    generation is ~95% of query time (LATENCY.md), and that window is otherwise dead.
    """
    user = normalize_user(req.user)
    return StreamingResponse(
        _chat_stream(user, req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _chat_stream(user: str, req: ChatRequest):
    global last_chat_at
    last_chat_at = time.monotonic()

    event_id = telemetry.new_event_id()
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    sources = []
    mode = Mode.GENERAL
    answer = ""

    try:
        user_has_index = has_index(user)
        results = []
        if user_has_index:
            # retrieve_detailed rather than retrieve: it returns the same `final` list
            # plus per-stage timings, which telemetry needs and which cost nothing extra.
            # Retrieval is synchronous and CPU/GPU-bound; running it inline would block
            # the event loop and stall every other client's token stream.
            detail = await anyio.to_thread.run_sync(
                lambda: retrieval.retrieve_detailed(
                    req.message, user, models.embeddings, reranker=models.reranker
                )
            )
            results = detail.final
            _add_retrieval_timings(timings, detail.timings)

        # Filter once, here, and let mode / sources / context / footer all derive from
        # the same list — otherwise the answer can cite a chunk the model never saw.
        selected = chatmod.select_context(results)
        mode = chatmod.decide_mode(selected, user_has_index)
        sources = chatmod.to_sources(selected) if mode is Mode.GROUNDED else []
        yield _sse("sources", SourcesEvent(mode=mode, sources=sources))

        prompt = ChatPromptTemplate.from_messages([
            ("system", chatmod.system_prompt(mode, tir=TIR_ENABLED,
                                             tools=models.tools_enabled)),
            ("human", chatmod.HUMAN_PROMPT),
        ])
        # Messages, not `prompt | llm | StrOutputParser()`. The parser maps each
        # AIMessageChunk to its .content and discards response_metadata — which is
        # exactly where Ollama reports done_reason. With it in the chain a generation
        # cut off at NUM_PREDICT is indistinguishable from one that finished, which is
        # why truncated answers shipped silently for as long as they did.
        #
        # Building the messages here does NOT change the prompt's shape: order stays
        # static text -> history -> context -> question, and continuation only ever
        # APPENDS after it. LATENCY.md's prefix rule depends on that and is load-bearing.
        base_messages = prompt.format_messages(
            context=chatmod.build_context(selected, mode),
            history=chatmod.format_history(req.history),
            input=req.message,
        )

        t_gen = time.perf_counter()
        first_token_at = None

        # Two accumulators, deliberately. `answer` is what the student saw, markers and
        # all; `generated` is model output only. Only `generated` may go back as the
        # prefill — feeding the markers back would have the model continue from text it
        # never wrote.
        generated = ""
        truncated = False
        continuations = 0
        # Closed assistant turns and their tool results, in agent mode only — always
        # empty on the TIR and no-tool paths, which is what keeps those byte-identical.
        #
        # It exists because the two tool protocols end a turn differently. TIR pauses
        # INSIDE the assistant turn: the ```output block is spliced into `generated` and
        # the same turn resumes, which is why one string was enough. A native tool call
        # ENDS the turn — the transcript needs a real AIMessage carrying the structured
        # call, then a ToolMessage per result, then a fresh assistant turn.
        #
        # Append-only, and always after the human message, so LATENCY.md's prefix rule
        # holds exactly as it does for TIR: `base_messages` is never mutated and
        # everything before the append point is unchanged, so only new tokens prefill.
        # The one visible difference is that the chat template renders a genuine turn
        # boundary here instead of more text inside one turn; both are pure appends at
        # the end of the prompt.
        turns: list = []

        # Guards the head of the answer against coming back as a copy of the question —
        # see chat.QuestionEcho. It filters what is *shown*, never `generated`: the
        # continuation prefill has to be the model's own text, byte for byte, or
        # PrefillEcho will not recognise it.
        qecho = chatmod.QuestionEcho(req.message)
        # Hides ```python blocks from the student while leaving them in `generated`,
        # which is where the protocol and the next prefill both need them. Disabled
        # outright by KBM_SHOW_TOOL_CODE, and irrelevant for a model without TIR.
        fences = chatmod.CodeFenceFilter() if (TIR_ENABLED and not SHOW_TOOL_CODE) else None
        shown = ""

        # Tool rounds and continuations are two reasons to re-enter the SAME loop, and
        # they are budgeted separately because they mean different things: a continuation
        # is "the answer did not fit", a tool round is "the model asked a question of the
        # interpreter". Sharing one budget would let a talkative answer starve the
        # arithmetic, or three computations eat the room the answer needed to finish.
        tool_rounds = 0
        tool_errors = 0
        tool_seconds = 0.0
        # Agent mode only. Search and Python are budgeted apart for the same reason tool
        # rounds and continuations are: three sympy calls must not eat the search the
        # answer actually needed. `notice_sent` is the one-shot budget notice, per tool.
        search_rounds = 0
        python_rounds = 0
        search_seconds = 0.0
        tool_counts: dict[str, int] = {}
        search_queries: list[str] = []
        late_results: list = []
        late_source_count = 0
        notice_sent: set[str] = set()
        # Set when a tool is asked for AGAIN after its budget notice already went out.
        # It cannot be a `break`: we are inside the per-call loop, and every call still
        # owes exactly one ToolMessage or the chat template breaks. So the loop finishes
        # answering, and the arm honours this immediately afterwards.
        stop_after_tools = False
        # Loaded on the first mid-answer search and reused by the second, so a re-search
        # does not re-open the BM25 pickle and the Chroma collection each time.
        bm25_index = None
        chroma_store = None
        budget_notice_sent = False
        # Backstop. Every arm below either breaks or consumes a budget, so this should be
        # unreachable — which is exactly why it is here. The loop is `while True` around a
        # network call driven by model output, and the cost of being wrong about "should
        # be unreachable" is a request that never returns and a pod that bills for it.
        passes = 0
        max_passes = MAX_CONTINUATIONS + (
            agent.MAX_PASSES if models.tools_enabled else tir.MAX_TOOL_ROUNDS
        ) + 2
        # Stop the moment the model opens an output block, or it invents the interpreter's
        # answer and reasons on from a number nobody computed — strictly worse than having
        # no tool at all. Ollama strips the stop text and reports done_reason == "stop",
        # the same value it reports for EOS, so the loop below must detect a tool call
        # from the TEXT (tir.pending_program) and never from done_reason.
        stop_words = tir.STOP_WORDS if TIR_ENABLED else None

        while True:
            passes += 1
            if passes > max_passes:
                log.error(
                    "generation loop hit the pass ceiling (%d) for user=%s — "
                    "tool_rounds=%d continuations=%d", max_passes, user, tool_rounds, continuations,
                )
                break
            prefix = [*base_messages, *turns]
            messages = prefix if not generated else [*prefix, AIMessage(content=generated)]
            # In agent mode the tool arm below closes the turn and resets `generated` to
            # "", so PrefillEcho is constructed empty and is a pass-through: it carries
            # continuations, exactly as it does today, and is structurally absent from the
            # native tool path. No flag, no branch — which matters, because ERRORS.md
            # 2026-08-20 is the record of what a wrong PrefillEcho verdict costs (every
            # continuation silently emitting nothing, on one model only).
            echo = chatmod.PrefillEcho(generated)
            done_reason = None
            produced = ""
            calls: list[dict] = []
            seen_calls: set[tuple] = set()

            stream = (models.llm_tools if models.tools_enabled else models.llm)
            async for chunk in stream.astream(messages, stop=stop_words):
                done_reason = chunk.response_metadata.get("done_reason") or done_reason
                for tc in (getattr(chunk, "tool_calls", None) or []):
                    # Deduped on (name, args) and NEVER on id: langchain_ollama mints a
                    # fresh uuid4 every time it parses a response (chat_models.py:218), so
                    # the id is a runtime artefact, not the model's. If Ollama ever
                    # repeated a tool_calls block on the terminal chunk, id-keyed dedupe
                    # would run the program twice. Measured on qwen2:7b it does not — but
                    # this costs nothing and the failure it prevents is a double execution.
                    key = (tc["name"], json.dumps(tc.get("args") or {}, sort_keys=True))
                    if key not in seen_calls:
                        seen_calls.add(key)
                        calls.append(tc)
                text = echo.feed(str(chunk.text))
                if not text:
                    continue
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                    timings["ttft_ms"] = round((first_token_at - t_start) * 1000, 1)
                produced += text

                visible = qecho.feed(text)
                if fences is not None and visible:
                    visible = fences.feed(visible)
                if not visible:
                    continue
                shown += visible
                answer += visible
                yield _sse("token", TokenEvent(text=visible))

            generated += produced

            # Three independent stop conditions, all required. Without `not produced` a
            # model that immediately emits EOS-at-length would spin to the cap emitting
            # nothing; without `echo.mismatch` a build that restates instead of
            # prefilling would duplicate the answer.
            #
            # `and not calls` is load-bearing, and it is the whole feature. A native tool
            # call very often arrives with NO text beside it — measured on qwen2:7b,
            # two of three probe questions produced zero content characters alongside
            # the call. Without this clause the loop breaks here, before the tool arm is
            # ever reached, and the student gets NO_ANSWER_TEXT instead of an answer. The
            # symptom would be silent, model-dependent, and would look like the model
            # failing rather than the server discarding its request — the same shape as
            # ERRORS.md 2026-08-20.
            if echo.mismatch or (not produced and not calls):
                truncated = done_reason == "length"
                break

            # ── Native tool arm (agent mode) ──────────────────────────────────
            # The third arm, and the sibling of the TIR arm below rather than a rival to
            # it: both mean "the model asked something of the world and is waiting", and
            # kbm/config.py guarantees only one of the two can ever be live. Ahead of the
            # length arm for the same reason TIR is — a pass can end with tool calls AND
            # done_reason == "length", and the tool call is the live request.
            #
            # THE INVARIANT: every tool_call in the AIMessage appended below gets exactly
            # one ToolMessage — including refusals, bad arguments, unknown names and
            # budget denials. A dangling call breaks the chat template's rendering, and a
            # model left waiting on a result it never receives stops mid-sentence. This is
            # the native form of the TIR arm splicing an output block rather than leaving
            # a ```python fence open.
            if models.tools_enabled and calls:
                tool_msgs = []
                for tc in calls[: agent.MAX_CALLS_PER_PASS]:
                    name = tc["name"]
                    call_id = tc.get("id") or uuid.uuid4().hex
                    args, err = agent.coerce_args(name, tc.get("args"))
                    if err:
                        # The model's mistake, addressed to the model. Nothing is shown to
                        # the student: a malformed call it will retry next round is noise,
                        # not an event in the answer.
                        log.info("tool call rejected for user=%s: %s", user, err)
                        tool_msgs.append(ToolMessage(content=err, tool_call_id=call_id))
                        continue

                    spent = {agent.TOOL_SEARCH: search_rounds,
                             agent.TOOL_PYTHON: python_rounds}.get(name, 0)
                    cap = {agent.TOOL_SEARCH: agent.MAX_SEARCH_ROUNDS,
                           agent.TOOL_PYTHON: agent.MAX_PYTHON_ROUNDS}.get(name)
                    if cap is not None and spent >= cap:
                        # Once per tool. This arm consumes no budget, so a model that
                        # answers the notice with another call of the same tool would spin
                        # for as long as the pod stayed up — telling it twice is a loop
                        # with no exit, exactly as in the TIR arm.
                        if name in notice_sent:
                            log.warning("model asked for %s again after its cap (%d), "
                                        "user=%s — ending the answer here", name, cap, user)
                            tool_msgs.append(ToolMessage(content=agent.budget_notice(name),
                                                         tool_call_id=call_id))
                            stop_after_tools = True
                            continue
                        notice_sent.add(name)
                        log.warning("%s round cap (%d) reached for user=%s", name, cap, user)
                        tool_msgs.append(ToolMessage(content=agent.budget_notice(name),
                                                     tool_call_id=call_id))
                        continue

                    tool_counts[name] = tool_counts.get(name, 0) + 1

                    if name == agent.TOOL_PYTHON:
                        python_rounds += 1
                        tool_rounds += 1
                        code = args["code"]
                        t_tool = time.perf_counter()
                        # Threaded for the same reason the TIR arm is: the sandbox forks a
                        # process and waits on it, and inline that stalls every other
                        # client's token stream.
                        result = await anyio.to_thread.run_sync(
                            # Loop variables bound as defaults, not captured. Correct today
                            # either way because the lambda is awaited immediately, but a
                            # closure over a loop variable is one refactor away from running
                            # the wrong program — and this one runs code.
                            lambda c=code: sandbox.run_python(c, timeout_s=SANDBOX_TIMEOUT_S)
                        )
                        tool_seconds += time.perf_counter() - t_tool
                        if not result.ok:
                            tool_errors += 1
                            log.info("sandbox %s for user=%s: %s",
                                     result.reason, user, result.output)
                        tool_msgs.append(ToolMessage(content=result.output,
                                                     tool_call_id=call_id))
                        if SHOW_TOOL_CODE:
                            # The code is never in the token stream in agent mode — it
                            # arrives in tool_calls[i]["args"] — so the operator switch
                            # needs its own renderer or it silently stops working here.
                            code_marker = chatmod.tool_code_marker(code, after=answer)
                            if code_marker:
                                answer += code_marker
                                yield _sse("token", TokenEvent(text=code_marker,
                                                               server_marker=True))
                        marker = chatmod.tool_output_marker(result.output, result.ok,
                                                            after=answer)
                        if marker:
                            answer += marker
                            yield _sse("token", TokenEvent(text=marker, server_marker=True))

                    elif name == agent.TOOL_SEARCH:
                        query = args["query"]
                        if not user_has_index:
                            # Answered before the budget is charged. There is nothing to
                            # search, so this call did no work — charging for it would let
                            # a student with no documents exhaust MAX_SEARCH_ROUNDS on two
                            # no-ops and then be told it was out of searches.
                            tool_msgs.append(ToolMessage(content=agent.format_documents([], 0),
                                                         tool_call_id=call_id))
                            continue
                        search_rounds += 1
                        search_queries.append(query)
                        t_search = time.perf_counter()
                        if bm25_index is None:
                            bm25_index = await anyio.to_thread.run_sync(
                                lambda: retrieval.load_bm25(user))
                        if chroma_store is None:
                            chroma_store = await anyio.to_thread.run_sync(
                                lambda: retrieval.load_chroma(user, models.embeddings))
                        detail2 = await anyio.to_thread.run_sync(
                            lambda q=query, b=bm25_index, st=chroma_store:
                                retrieval.retrieve_detailed(
                                    q, user, models.embeddings,
                                    reranker=models.reranker,
                                    bm25_index=b, store=st,
                                )
                        )
                        search_seconds += time.perf_counter() - t_search
                        _add_retrieval_timings(timings, detail2.timings)
                        # Unfiltered, for telemetry: the floor can be re-applied offline,
                        # a chunk that was never logged cannot.
                        late_results.extend(detail2.final)
                        # The SAME floor as the upfront retrieval, applied per chunk. A
                        # search the model asked for is not evidence that what came back
                        # is relevant, and pasting a 0.006 chunk into the transcript
                        # because the model chose to look is the failure select_context
                        # exists to prevent.
                        kept = chatmod.select_context(detail2.final)
                        new_sources = chatmod.to_sources(kept)
                        before = len(sources)
                        sources = chatmod.merge_sources(sources, new_sources)
                        late_source_count += len(sources) - before
                        tool_msgs.append(ToolMessage(
                            content=agent.format_search_result(
                                [(os.path.basename(d.metadata.get("source", "unknown")),
                                  d.page_content) for d, _ in kept],
                                len(kept),
                            ),
                            tool_call_id=call_id,
                        ))
                        marker = chatmod.search_marker(query, len(kept), after=answer)
                        if marker:
                            answer += marker
                            yield _sse("token", TokenEvent(text=marker, server_marker=True))

                    else:  # agent.TOOL_LIST
                        n_chunks, names = await anyio.to_thread.run_sync(
                            lambda: index_summary(user))
                        tool_msgs.append(ToolMessage(
                            content=agent.format_documents(names, n_chunks),
                            tool_call_id=call_id,
                        ))
                        # No student-facing marker. tool_output_marker exists so a wrong
                        # sandbox answer cannot be mistaken for a hallucination; "which
                        # files do I have" carries no such risk, and narrating it would
                        # clutter the answer. It stays visible in telemetry.

                # Anything past MAX_CALLS_PER_PASS still needs a result, per the invariant.
                for tc in calls[agent.MAX_CALLS_PER_PASS:]:
                    tool_msgs.append(ToolMessage(
                        content=agent.budget_notice(tc["name"]),
                        tool_call_id=tc.get("id") or uuid.uuid4().hex,
                    ))

                # Close the turn. `generated` becomes the assistant message carrying the
                # calls, and resets — so the next pass opens a fresh turn and
                # PrefillEcho("") is inert. This is the whole reason `turns` exists.
                turns.append(AIMessage(content=generated, tool_calls=calls))
                turns.extend(tool_msgs)
                generated = ""
                truncated = False
                if stop_after_tools:
                    # The notice has now gone out twice for some tool. Telling a model a
                    # third time is a loop with no exit — the TIR arm breaks here for the
                    # same reason. The `continue` above only advanced the per-call loop, so
                    # without this the arm re-entered generation and the pass ceiling was
                    # the only thing stopping it, while the log claimed the answer ended.
                    break
                continue

            # ── Tool arm ──────────────────────────────────────────────────────
            # Checked BEFORE the length arm. A model that opens a ```python block and is
            # cut off at the cap has not asked for anything yet (pending_program requires
            # a CLOSED block), so the two cannot both fire; checking the tool first keeps
            # that ordering explicit rather than relying on it.
            program = tir.pending_program(generated) if TIR_ENABLED else None
            if program is not None:
                if tool_rounds >= tir.MAX_TOOL_ROUNDS:
                    # Out of budget. Splice a result saying so rather than leaving the
                    # block dangling: the model is mid-sentence waiting on a number, and
                    # an empty transcript there is how you get an answer that just stops.
                    #
                    # Once only. Telling it a second time is a loop with no exit — the
                    # arm consumes no budget, so a model that answers the notice with
                    # another program would spin here for as long as the pod stayed up.
                    if budget_notice_sent:
                        log.warning(
                            "model asked for another program after the TIR cap (%d), "
                            "user=%s — ending the answer here", tir.MAX_TOOL_ROUNDS, user,
                        )
                        break
                    budget_notice_sent = True
                    log.warning("TIR round cap (%d) reached for user=%s", tir.MAX_TOOL_ROUNDS, user)
                    generated += tir.format_result(
                        "(no more calculations available — finish from what you have)"
                    )
                    continue

                tool_rounds += 1
                t_tool = time.perf_counter()
                # The sandbox forks a process and waits on it. Inline, that blocks the
                # event loop and stalls every other client's token stream — the same
                # reason retrieval runs in a thread above.
                result = await anyio.to_thread.run_sync(
                    lambda: sandbox.run_python(program, timeout_s=SANDBOX_TIMEOUT_S)
                )
                tool_seconds += time.perf_counter() - t_tool
                if not result.ok:
                    tool_errors += 1
                    log.info(
                        "sandbox %s for user=%s: %s", result.reason, user, result.output
                    )

                # The model gets the real text — a traceback line, a refused import — in
                # Qwen's own ```output shape, because that is what it can act on. The
                # student gets a sentence. Two strings for one event, as with Job.detail
                # and Job.diagnostic.
                generated += tir.format_result(result.output)
                marker = chatmod.tool_output_marker(result.output, result.ok, after=answer)
                if marker:
                    answer += marker
                    yield _sse("token", TokenEvent(text=marker, server_marker=True))
                truncated = False
                continue

            # ── Continuation arm ──────────────────────────────────────────────
            if done_reason != "length":
                truncated = False
                break
            truncated = True
            if continuations >= MAX_CONTINUATIONS:
                break
            continuations += 1

        # Anything the display filters were still holding when the stream ended.
        tail = qecho.flush()
        if fences is not None:
            tail = fences.feed(tail) + fences.flush()
        if tail:
            shown += tail
            answer += tail
            yield _sse("token", TokenEvent(text=tail))

        if qecho.fired:
            log.warning(
                "answer opened with a verbatim copy of the question (mode=%s, user=%s) — "
                "the echo was dropped; see chat.QuestionEcho", mode.value, user,
            )

        # An empty bubble followed by a sources footer is not an answer. This is the
        # honest end of the same failure QuestionEcho guards: the model returned nothing,
        # or nothing but the question.
        if not shown.strip():
            answer += chatmod.NO_ANSWER_TEXT
            yield _sse("token", TokenEvent(text=chatmod.NO_ANSWER_TEXT, server_marker=True))

        if truncated:
            answer += chatmod.TRUNCATION_MARKER
            yield _sse("token", TokenEvent(text=chatmod.TRUNCATION_MARKER, server_marker=True))

        # Server-written provenance; see chat.sources_footer for why the model is not
        # asked for it. It goes out as a normal token frame so a client needs no special
        # case, and it lands in `answer` so telemetry and the faithfulness eval see
        # exactly what the student saw. Last, because it is a statement about the answer
        # above it — and because appending cannot disturb the prompt prefix the next
        # turn's KV cache depends on (LATENCY.md).
        # Skipped when nothing was answered: there is no answer for the line to
        # attribute, and naming a document under "no answer came back" reads as though
        # the document were the reason. Also empty in `general` mode — the footer names
        # documents and there are none, so nothing is emitted rather than a line saying
        # so (see chat.sources_footer). `mode` on the sources/done frames is still the
        # authoritative provenance for any client that wants to show it.
        # `mode` is NOT revised when a mid-answer search found documents, and that is
        # deliberate: it records the provenance decision the calibrated relevance floor
        # made up front, on the question as asked, and it is what EVALUATION.md §10.8
        # depends on to make faithfulness measurable. A label that quietly means something
        # different on answers where the model happened to search again is not a label.
        #
        # But the footer's own invariant is "names documents iff there are documents to
        # name", and a `general` answer that searched and found something now has some.
        # Passing GROUNDED here satisfies the footer without rewriting the decision;
        # DoneEvent.late_sources is how a client sees the difference.
        footer = chatmod.sources_footer(
            sources, Mode.GROUNDED if sources else mode
        ) if shown.strip() else ""
        if footer:
            answer += footer
            yield _sse("token", TokenEvent(text=footer, server_marker=True))

        timings["generate_ms"] = round((time.perf_counter() - t_gen) * 1000, 1)
        if tool_rounds:
            timings["tool_ms"] = round(tool_seconds * 1000, 1)
        if search_rounds:
            timings["search_ms"] = round(search_seconds * 1000, 1)

        yield _sse("done", DoneEvent(
            mode=mode,
            answer=answer,
            sources=sources,
            timings=_timings_model(timings),
            event_id=event_id,
            truncated=truncated,
            continuations=continuations,
            tool_calls=tool_rounds,
            tool_errors=tool_errors,
            searches=search_rounds,
            late_sources=late_source_count,
        ))
        # Telemetry logs every RETRIEVED chunk, not just the ones that cleared the floor.
        # The floor can be re-applied offline, but a chunk that was dropped and never
        # logged is gone — and re-calibrating the floor against real family questions is
        # exactly what this log exists to make possible (EVALUATION.md §10.9). Each row
        # carries its score, so "what would floor=X have done" stays answerable.
        telemetry.log_query(
            event_id=event_id, user=user, question=req.message, mode=mode.value,
            sources=[s.model_dump() for s in chatmod.to_sources(results)], timings=timings,
            model=OLLAMA_MODEL, n_completion_chars=len(answer),
            truncated=truncated, continuations=continuations,
            tool_calls=tool_rounds, tool_errors=tool_errors,
            protocol=models.protocol, tool_counts=tool_counts,
            search_queries=search_queries,
            late_sources=[s.model_dump() for s in chatmod.to_sources(late_results)],
        )

    except Exception as e:  # noqa: BLE001 - surfaced to the client as an SSE error frame
        yield _sse("error", ErrorEvent(message=str(e)))
        telemetry.log_query(
            event_id=event_id, user=user, question=req.message, mode=mode.value,
            sources=[], timings=timings, model=OLLAMA_MODEL,
            n_completion_chars=len(answer), error=str(e),
        )


def _timings_model(t: dict):
    """The timings dict as the wire model.

    Built by name from Timings' own fields rather than by listing them here: the previous
    version enumerated each key, so `search_ms` was collected, logged to telemetry, and
    then silently dropped from the `done` frame — a field that exists everywhere except
    where a client can see it. Deriving the list from the schema makes the schema the one
    place a timing is declared.
    """
    from .schemas import Timings

    return Timings(**{k: t.get(k) for k in Timings.model_fields})


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=Job)
async def upload(user: str = Form(...), file: UploadFile = File(...)):
    """Accept a PDF and ingest it in the background.

    Returns a job handle rather than blocking: Marker takes ~0.3-1 s/page on GPU, so a
    300-page textbook is minutes, well past any sensible HTTP timeout.
    """
    user = normalize_user(user)
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only .pdf files are accepted.")

    # Read to a temp file that outlives this request — the worker thread needs it after
    # FastAPI has closed the UploadFile.
    tmp_dir = tempfile.mkdtemp(prefix="kbm_upload_")
    pdf_path = os.path.join(tmp_dir, os.path.basename(file.filename))
    with open(pdf_path, "wb") as f:
        while block := await file.read(1 << 20):
            f.write(block)

    job = Job(job_id=uuid.uuid4().hex, status=JobStatus.QUEUED, filename=file.filename, user=user)
    _jobs[job.job_id] = job
    # Keep the Future. Discarding it swallowed anything that escaped _run_ingest itself
    # — a KeyError on _jobs[job_id], an ImportError from the function-local imports, a
    # thread killed by the OOM reaper — leaving the job "queued" forever and the client
    # polling a status that would never change.
    fut = _ingest_pool.submit(_run_ingest, job.job_id, pdf_path, tmp_dir, user)
    fut.add_done_callback(lambda f: _mark_crashed(job.job_id, f))
    return job


def _mark_crashed(job_id: str, fut: Future) -> None:
    """Backstop for exceptions _run_ingest could not report through the job record."""
    exc = fut.exception()
    job = _jobs.get(job_id)
    if job is None:
        log.error("ingest job %s vanished from the registry", job_id)
        return
    if exc is not None:
        log.exception("ingest job %s crashed outside its own handler", job_id, exc_info=exc)
        job.status = JobStatus.FAILED
        job.detail = f"Could not add {job.filename} — the import stopped unexpectedly."
        job.diagnostic = f"{type(exc).__name__} escaped _run_ingest: {exc}"
    elif job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
        # Returned without reaching either terminal branch — should be unreachable.
        job.status = JobStatus.FAILED
        job.detail = f"Could not add {job.filename} — the import stopped unexpectedly."
        job.diagnostic = "worker returned without setting a terminal status"


def _run_ingest(job_id: str, pdf_path: str, tmp_dir: str, user: str) -> None:
    """The same sequence app.py's handle_upload ran, moved off the request thread."""
    from extract import extract_detailed
    from ingest import build_bm25, build_chroma, load_mmd_files, merge_chunks
    from kbm.chunking import split_baseline

    job = _jobs[job_id]
    job.status = JobStatus.RUNNING
    try:
        # The uploaded PDF and its .mmd are NOT kept. Both live in tmp_dir and die with
        # it in the finally below; the chunks in the indexes are the only copy the pod
        # holds. This deliberately reverses the earlier policy of persisting them under
        # $DATA_DIR/docs/{raw,extracted}/<user>/, so before "fixing" it back, here are
        # the consequences that were accepted along with it:
        #   - Recovery depends on the family still having their own PDFs. Nothing on the
        #     volume rebuilds a user's corpus, so the indexes are what to back up now.
        #   - Re-chunking a user document (baseline -> eqaware) needs a re-upload and
        #     another Marker run. Re-EMBEDDING still works offline: build_bm25 pickles
        #     the chunk text, so only the boundaries are frozen.
        #   - EVALUATION.md's "freeze the .mmd" rule is about the repo-relative eval
        #     corpus built by the extract.py CLI, which this does not touch.
        job.stage = "extract"
        result = extract_detailed(pdf_path, out_dir=tmp_dir)
        job.extractor = result.extractor
        job.degraded = result.degraded

        job.stage = "chunk"
        docs = load_mmd_files([result.path])
        # TextLoader stamps the full path as `source`, which is now a temp dir that will
        # not exist by the time anyone reads the index. Chunk ids are basename-derived
        # (chunking.assign_chunk_ids), so this is id-neutral — it just stops the stored
        # metadata naming a file that never existed on the volume.
        for d in docs:
            d.metadata["source"] = os.path.basename(result.path)
        chunks = split_baseline(docs)

        # Merge with the user's existing chunks so a second upload adds to the corpus
        # rather than replacing it — build_bm25 overwrites the pickle wholesale.
        # merge_chunks dedupes by chunk_id, so re-uploading a document replaces its
        # chunks instead of indexing them twice.
        job.stage = "index"
        if has_index(user):
            _, existing = retrieval.load_bm25(user)
            chunks = merge_chunks(existing, chunks)

        build_bm25(chunks, user)
        build_chroma(chunks, user, models.embeddings)

        job.status = JobStatus.DONE
        job.stage = None
        job.n_chunks = len(chunks)
        n = len(chunks)
        counted = f"{n} section{'' if n == 1 else 's'} indexed"
        # `detail` is read by a family member, `diagnostic` by whoever runs the pod. The
        # two used to be one string, so a Marker failure put a raw exception — install
        # URLs and all — into the box that says the upload worked. Say what it means for
        # them here; put the cause where it is useful, in the log and in `diagnostic`.
        if result.degraded:
            job.detail = (
                f"Added {job.filename}, but without proper equation formatting — "
                f"maths questions about it may not find the right sections. Worth "
                f"uploading again once that is fixed. {counted}."
            )
            job.diagnostic = (
                f"Marker unavailable, fell back to pymupdf4llm: {result.marker_error}"
                if result.marker_error
                else "pymupdf4llm was requested explicitly (--force-pymupdf)"
            )
            log.warning("ingest job %s degraded: %s", job_id, job.diagnostic)
        else:
            job.detail = f"Added {job.filename} — {counted} and ready to search."
    except Exception as e:  # noqa: BLE001 - reported through the job record
        # A bare str(e) was all this used to record: no indication of which stage raised
        # it, and no traceback anywhere. Log both.
        log.exception("ingest job %s failed during %s", job_id, job.stage)
        job.status = JobStatus.FAILED
        # Plain sentence for the person who uploaded; exception text for the log and for
        # `diagnostic`. A stack trace in the status box helps nobody who can act on it.
        job.detail = f"Could not add {job.filename} — {_STAGE_BLAME.get(job.stage, 'something went wrong')}."
        job.diagnostic = f"{type(e).__name__} during {job.stage or 'ingest'}: {e}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.get("/jobs/{job_id}", response_model=Job)
async def get_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown job id.")
    return job


# ── Users, feedback, health ───────────────────────────────────────────────────

@router.get("/users/{user}/status", response_model=UserStatus)
async def user_status(user: str):
    user = normalize_user(user)
    n_chunks, sources = index_summary(user)
    return UserStatus(user=user, has_index=has_index(user), n_chunks=n_chunks, sources=sources)


@router.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def feedback(fb: Feedback):
    telemetry.log_feedback(fb.event_id, fb.rating, fb.note)


@router.get("/healthz", response_model=Health)
async def healthz():
    """Liveness plus whether Ollama holds the generator resident.

    The wake path needs the second fact: after a cold pod start the API answers long
    before the model is loaded, and a client that pings only for HTTP 200 will fire its
    first question into a ~20s model load and look broken.
    """
    loaded = False
    try:
        import httpx

        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get("http://localhost:11434/api/ps")
            loaded = any(m.get("name") == OLLAMA_MODEL for m in r.json().get("models", []))
    except Exception:  # noqa: BLE001 - Ollama not up yet is a normal cold-start state
        loaded = False
    return Health(model=OLLAMA_MODEL, model_loaded=loaded, retrieval_ready=models.ready,
                  protocol=models.protocol)
