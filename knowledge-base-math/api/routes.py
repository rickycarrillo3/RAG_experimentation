"""
api/routes.py - HTTP surface.

The retrieval pipeline is *imported* from retrieval.py and never reimplemented here.
CLAUDE.md is explicit about why: a second copy drifts, and then the CLI, the web UI
and the eval quietly stop describing the same system.
"""

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
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

import retrieval
import telemetry
from retrieval import OLLAMA_MODEL

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
from .settings import MAX_CONTINUATIONS

log = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_token)])

# Ingest is GPU-heavy (Marker/Surya) and must not run concurrently with itself, or two
# uploads will fight over VRAM that the generator and reranker are already holding —
# EVALUATION.md §6 measures upload as the tightest moment in the memory budget.
_ingest_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
_jobs: dict[str, Job] = {}

# Set by the idle-stop watchdog; every /chat touches it. See ops/idle_stop.py.
last_chat_at = time.monotonic()


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
            # RetrievalResult.timings is in seconds; the API reports milliseconds.
            timings.update({f"{k}_ms": round(v * 1000, 2) for k, v in detail.timings.items()})

        mode = chatmod.decide_mode(results, user_has_index)
        sources = chatmod.to_sources(results) if mode is Mode.GROUNDED else []
        yield _sse("sources", SourcesEvent(mode=mode, sources=sources))

        prompt = ChatPromptTemplate.from_messages([
            ("system", chatmod.SYSTEM_PROMPTS[mode]),
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
            context=chatmod.build_context(results, mode),
            history=chatmod.format_history(req.history),
            input=req.message,
        )

        # Server-emitted provenance marker; see chat.GENERAL_MODE_MARKER for why this
        # is not left to the model. It goes out as a normal token frame so the client
        # needs no special case, and it lands in `answer` so telemetry and the
        # faithfulness eval see exactly what the student saw.
        if mode is Mode.GENERAL:
            answer += chatmod.GENERAL_MODE_MARKER
            yield _sse("token", TokenEvent(text=chatmod.GENERAL_MODE_MARKER))

        t_gen = time.perf_counter()
        first_token_at = None

        # Two accumulators, deliberately. `answer` is what the student saw, markers and
        # all; `generated` is model output only. Only `generated` may go back as the
        # prefill — feeding the markers back would have the model continue from text it
        # never wrote.
        generated = ""
        truncated = False
        continuations = 0

        for attempt in range(MAX_CONTINUATIONS + 1):
            messages = base_messages if not generated else [
                *base_messages, AIMessage(content=generated)
            ]
            echo = chatmod.PrefillEcho(generated)
            done_reason = None
            produced = ""

            async for chunk in models.llm.astream(messages):
                done_reason = chunk.response_metadata.get("done_reason") or done_reason
                text = echo.feed(str(chunk.text))
                if not text:
                    continue
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                    timings["ttft_ms"] = round((first_token_at - t_start) * 1000, 1)
                produced += text
                answer += text
                yield _sse("token", TokenEvent(text=text))

            generated += produced

            # Three independent stop conditions, all required. Without `not produced` a
            # model that immediately emits EOS-at-length would spin to the cap emitting
            # nothing; without `echo.mismatch` a build that restates instead of
            # prefilling would duplicate the answer.
            if echo.mismatch or not produced:
                truncated = done_reason == "length"
                break
            if done_reason != "length":
                truncated = False
                break
            truncated = True
            if attempt < MAX_CONTINUATIONS:
                continuations += 1

        if truncated:
            answer += chatmod.TRUNCATION_MARKER
            yield _sse("token", TokenEvent(text=chatmod.TRUNCATION_MARKER))

        timings["generate_ms"] = round((time.perf_counter() - t_gen) * 1000, 1)

        yield _sse("done", DoneEvent(
            mode=mode,
            answer=answer,
            sources=sources,
            timings=_timings_model(timings),
            event_id=event_id,
            truncated=truncated,
            continuations=continuations,
        ))
        telemetry.log_query(
            event_id=event_id, user=user, question=req.message, mode=mode.value,
            sources=[s.model_dump() for s in sources], timings=timings,
            model=OLLAMA_MODEL, n_completion_chars=len(answer),
            truncated=truncated, continuations=continuations,
        )

    except Exception as e:  # noqa: BLE001 - surfaced to the client as an SSE error frame
        yield _sse("error", ErrorEvent(message=str(e)))
        telemetry.log_query(
            event_id=event_id, user=user, question=req.message, mode=mode.value,
            sources=[], timings=timings, model=OLLAMA_MODEL,
            n_completion_chars=len(answer), error=str(e),
        )


def _timings_model(t: dict):
    from .schemas import Timings

    return Timings(
        bm25_ms=t.get("bm25_ms"), dense_ms=t.get("dense_ms"), rrf_ms=t.get("rrf_ms"),
        rerank_ms=t.get("rerank_ms"), ttft_ms=t.get("ttft_ms"), generate_ms=t.get("generate_ms"),
    )


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
        job.detail = f"Ingest crashed: {exc}"
    elif job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
        # Returned without reaching either terminal branch — should be unreachable.
        job.status = JobStatus.FAILED
        job.detail = "Ingest ended without reporting a result."


def _run_ingest(job_id: str, pdf_path: str, tmp_dir: str, user: str) -> None:
    """The same sequence app.py's handle_upload ran, moved off the request thread."""
    from chunking import split_baseline
    from extract import extract_detailed
    from ingest import build_bm25, build_chroma, load_mmd_files, merge_chunks

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
        # The message has to carry the degradation, because this is the exact path that
        # used to report "Indexed. N total chunks" for an index containing no LaTeX at
        # all: extract() returned the same string whether Marker ran or the pymupdf4llm
        # fallback did, so nothing upstream could tell them apart. Lead with the bad news.
        if result.degraded:
            why = (
                f"Marker failed ({result.marker_error})"
                if result.marker_error
                else "pymupdf4llm was requested"
            )
            job.detail = (
                f"Indexed WITHOUT LaTeX. {why}, so equations are flattened to plain "
                f"text and math retrieval will be poor. Fix Marker and re-upload to "
                f"replace these chunks. {len(chunks)} total chunks for {user}."
            )
        else:
            job.detail = f"Indexed with Marker. {len(chunks)} total chunks for {user}."
    except Exception as e:  # noqa: BLE001 - reported through the job record
        # A bare str(e) was all this used to record: no indication of which stage raised
        # it, and no traceback anywhere. Log both.
        log.exception("ingest job %s failed during %s", job_id, job.stage)
        job.status = JobStatus.FAILED
        job.detail = f"{job.stage or 'ingest'} failed: {e}"
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
    return Health(model=OLLAMA_MODEL, model_loaded=loaded, retrieval_ready=models.ready)
