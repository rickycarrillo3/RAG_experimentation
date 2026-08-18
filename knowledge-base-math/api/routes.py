"""
api/routes.py - HTTP surface.

The retrieval pipeline is *imported* from retrieval.py and never reimplemented here.
CLAUDE.md is explicit about why: a second copy drifts, and then the CLI, the web UI
and the eval quietly stop describing the same system.
"""

import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import anyio
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from langchain_core.output_parsers import StrOutputParser
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

router = APIRouter(dependencies=[Depends(require_token)])

# /healthz is deliberately OUTSIDE the authenticated router. It is a liveness probe:
# startup.sh polls it to know when the API is up, and the wake path polls `model_loaded`
# to know when the first question will not land in a 20s model load. Both of those run
# before, or without, a token — behind auth the probe gets a 401 and "is it up?" becomes
# indistinguishable from "is my token right?". It exposes only the model name and two
# booleans, no user data.
public_router = APIRouter()

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
        chain = prompt | models.llm | StrOutputParser()

        # Server-emitted provenance marker; see chat.GENERAL_MODE_MARKER for why this
        # is not left to the model. It goes out as a normal token frame so the client
        # needs no special case, and it lands in `answer` so telemetry and the
        # faithfulness eval see exactly what the student saw.
        if mode is Mode.GENERAL:
            answer += chatmod.GENERAL_MODE_MARKER
            yield _sse("token", TokenEvent(text=chatmod.GENERAL_MODE_MARKER))

        t_gen = time.perf_counter()
        first_token_at = None
        async for token in chain.astream({
            "context": chatmod.build_context(results, mode),
            "history": chatmod.format_history(req.history),
            "input": req.message,
        }):
            if first_token_at is None:
                first_token_at = time.perf_counter()
                timings["ttft_ms"] = round((first_token_at - t_start) * 1000, 1)
            answer += token
            yield _sse("token", TokenEvent(text=token))
        timings["generate_ms"] = round((time.perf_counter() - t_gen) * 1000, 1)

        yield _sse("done", DoneEvent(
            mode=mode,
            answer=answer,
            sources=sources,
            timings=_timings_model(timings),
            event_id=event_id,
        ))
        telemetry.log_query(
            event_id=event_id, user=user, question=req.message, mode=mode.value,
            sources=[s.model_dump() for s in sources], timings=timings,
            model=OLLAMA_MODEL, n_completion_chars=len(answer),
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
    _ingest_pool.submit(_run_ingest, job.job_id, pdf_path, tmp_dir, user)
    return job


def _run_ingest(job_id: str, pdf_path: str, tmp_dir: str, user: str) -> None:
    """The same sequence app.py's handle_upload ran, moved off the request thread."""
    import shutil

    from .settings import DATA_DIR

    from chunking import split_baseline
    from extract import extract
    from ingest import build_bm25, build_chroma, load_mmd_files, merge_chunks

    job = _jobs[job_id]
    job.status = JobStatus.RUNNING
    try:
        # Keep the source PDF and the extracted .mmd on the persistent volume rather
        # than only their derived vectors. Three reasons, all learned the hard way:
        #   - The indexes are rebuildable from these; these are rebuildable from
        #     nothing. Discarding them made "back up docs/raw/" impossible to follow
        #     for anything uploaded through the web UI.
        #   - Re-chunking (baseline -> eqaware) needs the .mmd. Without it, evaluating
        #     a chunking change means asking the family to re-upload their textbooks
        #     and paying for Marker again.
        #   - EVALUATION.md §10.2 requires frozen .mmd files, because re-extracting
        #     shifts chunk boundaries and silently breaks every gold label.
        raw_dir = os.path.join(DATA_DIR, "docs", "raw", user)
        mmd_dir = os.path.join(DATA_DIR, "docs", "extracted", user)
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(mmd_dir, exist_ok=True)
        shutil.copy2(pdf_path, os.path.join(raw_dir, os.path.basename(pdf_path)))

        mmd_path = extract(pdf_path, out_dir=mmd_dir)
        chunks = split_baseline(load_mmd_files([mmd_path]))

        # Merge with the user's existing chunks so a second upload adds to the corpus
        # rather than replacing it — build_bm25 overwrites the pickle wholesale.
        # merge_chunks dedupes by chunk_id, so re-uploading a document replaces its
        # chunks instead of indexing them twice.
        if has_index(user):
            _, existing = retrieval.load_bm25(user)
            chunks = merge_chunks(existing, chunks)

        build_bm25(chunks, user)
        build_chroma(chunks, user, models.embeddings)

        job.status = JobStatus.DONE
        job.n_chunks = len(chunks)
        job.detail = f"Indexed. {len(chunks)} total chunks for {user}."
    except Exception as e:  # noqa: BLE001 - reported through the job record
        job.status = JobStatus.FAILED
        job.detail = str(e)
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


@public_router.get("/healthz", response_model=Health)
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
