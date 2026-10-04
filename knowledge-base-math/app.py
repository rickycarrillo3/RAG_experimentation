"""
app.py - Gradio web UI, as a client of the FastAPI service.

Run the API first, then this:
    uvicorn api.main:app --port 8000        # terminal 1
    python app.py                           # terminal 2  → http://localhost:7860

This file used to import the pipeline directly and hold the models itself. It no
longer does: it speaks HTTP to api/, exactly as the TypeScript frontend will. Keeping
a working UI on top of the real API is what proves the API is complete — if something
cannot be done over HTTP, it shows up here before any TypeScript is written.

This is deliberately temporary. When the TS frontend lands, delete this file rather
than porting it.
"""

import json
import logging
import os
import time

import gradio as gr
import httpx

from kbm.config import APP_HOST, APP_PORT, app_auth
from kbm.logsetup import configure_logging

log = logging.getLogger(__name__)

API_URL = os.environ.get("KBM_API_URL", "http://127.0.0.1:8000")
API_TOKEN = os.environ.get("KBM_API_TOKEN", "")
# Generation of a full answer runs to tens of seconds; the default httpx timeout would
# abort mid-stream. Read timeout is the one that matters for SSE.
TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=600.0, pool=10.0)


def _headers() -> dict:
    return {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}


# ── Waking the API host ───────────────────────────────────────────────────────
#
# Since the split this UI runs on an always-on CPU pod and the API runs on a GPU pod
# that stops itself when idle (ops/idle_stop.py). Sleep was already automatic; wake was
# not, and DEPLOYMENT.md §5 called that "the largest remaining gap in the day-to-day
# experience" — a family member who opened the link on a stopped pod got a dead URL and
# no explanation, and the only fix was to text whoever holds the RunPod credentials.
#
# Wake is TWO things, and building only the first is the documented trap: the pod must
# be started, AND its processes must come back, because a start recreates the container
# from the image with nothing running. So starting is not readiness — /healthz is.
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "").strip()
RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID", "").strip()
WAKE_TIMEOUT = float(os.environ.get("KBM_WAKE_TIMEOUT_MIN", "5")) * 60
WAKE_POLL_SECONDS = 5.0


def _healthz() -> tuple[int, dict]:
    """(status code, body). Code 0 means nothing answered at all.

    Read the STATUS CODE, not merely whether the request completed. A 401 and a 500 are
    both "not ready" but they are not the same problem, and treating any response as
    success is how a broken API reads as healthy. `model_loaded` is the field that
    matters: after a cold start the API answers 200 well before the generator is
    resident, so a client that checks only for 200 fires its first question into a model
    load and looks broken.
    """
    try:
        r = httpx.get(f"{API_URL}/healthz", headers=_headers(), timeout=5.0)
    except Exception:
        return 0, {}
    if r.status_code != 200:
        return r.status_code, {}
    try:
        return 200, r.json()
    except ValueError:
        return 200, {}


def ensure_awake():
    """Yield human-readable progress while the API host comes up; return when there is
    nothing further to wait for.

    Best-effort by design. This makes the 1-2 minute wake legible instead of silent; it
    is not the error path. Whatever state we return in, the caller goes on to make its
    real request, and the existing failure handling reports what actually happened. That
    keeps one error path instead of two that can disagree.
    """
    code, health = _healthz()
    if code == 200 and health.get("model_loaded"):
        return

    if code == 401:
        # The pod is up and talking; the token is wrong. Polling for five minutes would
        # not fix a credential, and starting a pod that is already running does nothing.
        yield "The API rejected this app's token. KBM_API_TOKEN here does not match the API host's."
        return

    if code == 0:
        if not (RUNPOD_API_KEY and RUNPOD_POD_ID):
            yield (
                "_The tutor's server is not responding. It may be asleep — "
                "ask whoever runs it to start it._"
            )
            return
        yield "_Starting the tutor's server… this takes a minute or two._"
        try:
            from ops.runpod import start_pod

            start_pod(RUNPOD_POD_ID, RUNPOD_API_KEY)
        except Exception as e:  # noqa: BLE001 - a failed start is a message, not a crash
            log.warning("wake: start_pod failed: %s", e)
            yield "_Could not start the tutor's server. Trying anyway…_"

    started = time.monotonic()
    while time.monotonic() - started < WAKE_TIMEOUT:
        time.sleep(WAKE_POLL_SECONDS)
        code, health = _healthz()
        if code == 200 and health.get("model_loaded"):
            return
        waited = int(time.monotonic() - started)
        if code == 200:
            # Container is back and uvicorn is serving; Ollama is still pulling the
            # generator into VRAM. This is the second of the two waits, and naming it
            # separately is what stops "still starting" looking like a hang.
            yield f"_Server is up, loading the model… ({waited}s)_"
        else:
            yield f"_Waiting for the tutor's server… ({waited}s)_"

    yield "_The server is taking longer than usual. Trying your question anyway…_"


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


# ── Upload ────────────────────────────────────────────────────────────────────

# Marker on a large textbook is minutes; this is the point at which we stop watching and
# tell the user where to look instead. It does NOT cancel the job — the server keeps
# going, and GET /jobs/{id} still has the answer.
UPLOAD_POLL_TIMEOUT = float(os.environ.get("KBM_UPLOAD_POLL_TIMEOUT_MIN", "45")) * 60

# The family sees plain sentences; whoever runs the pod can opt into the exception text
# without reading the server log. Off by default — an install URL or a stack trace in the
# status box is noise to everyone who cannot act on it.
SHOW_DIAGNOSTICS = os.environ.get("KBM_SHOW_DIAGNOSTICS", "").strip() not in ("", "0")


def _diagnostic(job: dict) -> str:
    """The technical cause, only when this deployment asked to see it."""
    d = job.get("diagnostic")
    return f"\n\n<small>{d}</small>" if (d and SHOW_DIAGNOSTICS) else ""


def handle_upload(pdf_file, username: str, last_ingested):
    """Ingest a selected PDF, streaming status back as it goes.

    A generator rather than a plain function so a multi-minute Marker run shows progress
    instead of an empty box. Yields (status_text, last_ingested, file_update);
    `last_ingested` is what makes auto-fire safe — see the guard below.

    The third value is almost always `gr.update()` (leave the file box alone). It is
    `gr.update(value=None)` only where the upload definitively failed, because clearing
    the box is what lets the user retry by dropping the same file again — there is no
    retry button any more. Do NOT clear it on a *polling* failure: the job may still be
    running on the server.
    """
    username = (username or "").strip().lower()

    if pdf_file is None:
        # Also the "clear" path: forget what we ingested so re-selecting it works.
        yield "No file selected.", None, gr.update()
        return
    if not username:
        # Named recovery action, because this fires on file selection now: a user who
        # picks the file first would otherwise hit a dead end with nothing to re-trigger.
        yield "Enter your name above, then press Enter.", last_ingested, gr.update()
        return
    if pdf_file.name == last_ingested:
        # This function is wired to three triggers, one of which is the name box's submit
        # event, so it can fire repeatedly for one file. Without this guard every Enter
        # press would re-run Marker — minutes of GPU for a document already indexed.
        yield f"Already ingested {os.path.basename(pdf_file.name)}. Choose another file to add more.", last_ingested, gr.update()
        return

    # An upload to a sleeping pod fails the same way a question does, and Marker is the
    # more expensive thing to have to retry — so wake here too rather than only in chat.
    for note in ensure_awake():
        yield note, last_ingested, gr.update()

    job_id = None
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            yield f"Uploading {os.path.basename(pdf_file.name)}...", last_ingested, gr.update()
            with open(pdf_file.name, "rb") as fh:
                r = client.post(
                    f"{API_URL}/upload",
                    headers=_headers(),
                    data={"user": username},
                    files={"file": (os.path.basename(pdf_file.name), fh, "application/pdf")},
                )
            if r.status_code != 200:
                # r.text is a raw FastAPI error body. Show the message if there is a
                # readable one, never the JSON envelope.
                try:
                    why = r.json().get("detail") or "the server rejected it"
                except Exception:
                    why = "the server rejected it"
                yield (
                    f"Could not upload {os.path.basename(pdf_file.name)} — {why}"
                    "\n\nPlease try uploading the file again.",
                    last_ingested,
                    gr.update(value=None),
                )
                return
            job_id = r.json()["job_id"]

            # Poll rather than block the request: Marker is minutes, not seconds.
            started = time.monotonic()
            while True:
                time.sleep(3)
                elapsed = time.monotonic() - started
                job = client.get(f"{API_URL}/jobs/{job_id}", headers=_headers()).json()
                status = job["status"]

                if status == "done":
                    # `detail` is the user-facing sentence and `diagnostic` is the
                    # technical cause; only the first belongs on screen. A Marker failure
                    # still lands here — the pymupdf4llm fallback indexes fine, it just
                    # produces no LaTeX — so the warning prefix is what distinguishes it.
                    prefix = "⚠️ " if job.get("degraded") else "✅ "
                    yield prefix + job["detail"] + _diagnostic(job), pdf_file.name, gr.update()
                    return
                if status == "failed":
                    yield (
                        "❌ " + job["detail"] + _diagnostic(job)
                        + "\n\nPlease try uploading the file again.",
                        last_ingested,
                        gr.update(value=None),
                    )
                    return

                if elapsed > UPLOAD_POLL_TIMEOUT:
                    yield (
                        f"Still working after {elapsed / 60:.0f} minutes. The import has "
                        f"not been cancelled — it should finish on its own.",
                        last_ingested,
                        gr.update(),
                    )
                    return

                yield f"{status.title()}... ({elapsed / 60:.1f} min elapsed)", last_ingested, gr.update()
    except Exception as e:
        # Where this landed decides what to say. Before the server handed back a job id
        # nothing was ever accepted, so it is an ordinary upload failure and the box gets
        # cleared for a retry. After that, the job may well still be running on the
        # server — say so, and do NOT clear the file the user would then re-send.
        log.warning("upload failed (job_id=%s)", job_id, exc_info=e)
        if job_id is None:
            yield (
                "Could not reach the server. Please try uploading the file again.",
                last_ingested,
                gr.update(value=None),
            )
        else:
            yield (
                "Lost contact with the server while importing. The import may still be "
                "running — wait a moment, then re-upload the file to check.",
                last_ingested,
                gr.update(),
            )


# ── Chat ──────────────────────────────────────────────────────────────────────

def handle_chat(message: str, history: list, clean_history: list, username: str):
    """Stream one answer from the API, yielding Gradio state as tokens arrive.

    Yields (cleared input, display history, clean history, event_id). `clean_history`
    holds model text only — token frames flagged `server_marker` (the sources footer,
    the truncation notice) are displayed but kept out of it, so the model is never sent
    back a transcript in which it appears to have written the server's own words.
    """
    username = (username or "").strip().lower()
    if not username:
        yield "", history + [_msg("user", message), _msg("assistant", "Please enter your name first.")], clean_history, None
        return
    if not message.strip():
        yield "", history, clean_history, None
        return

    base = history + [_msg("user", message)]
    # Before anything else: the API host may be a stopped GPU pod. Yields nothing at all
    # in the normal case where it is already awake, so this costs one /healthz call.
    for note in ensure_awake():
        yield "", base + [_msg("assistant", note)], clean_history, None
    yield "", base + [_msg("assistant", "_Searching your documents…_")], clean_history, None

    answer = ""   # what the student sees: model text plus the server's footer
    clean = ""    # model text only; this is what goes back as conversation context
    event_id = None
    payload = {
        "user": username,
        "message": message,
        "history": [{"role": m["role"], "content": m["content"]} for m in clean_history],
    }

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            with client.stream("POST", f"{API_URL}/chat", headers=_headers(), json=payload) as resp:
                if resp.status_code != 200:
                    resp.read()
                    raise RuntimeError(f"API returned {resp.status_code}: {resp.text}")

                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = json.loads(line[5:].strip())
                    kind = data.get("type")

                    if kind == "token":
                        answer += data["text"]
                        # The server names the sources itself, in one line at the end of
                        # the answer (api/chat.py:sources_footer). Building a second list
                        # here from the `sources` frame only printed the same filename
                        # five times, once per retrieved chunk, under the one the student
                        # was actually meant to read.
                        if not data.get("server_marker"):
                            clean += data["text"]
                        yield "", base + [_msg("assistant", answer)], clean_history, event_id
                    elif kind == "done":
                        event_id = data["event_id"]
                    elif kind == "error":
                        raise RuntimeError(data["message"])
    except Exception as e:
        answer = f"Error: {e}"
        clean = ""

    # A failed turn is not conversation. Recording "Error: ..." — or an empty reply —
    # as the tutor's turn would put it in the transcript the model continues from.
    new_clean = (
        clean_history + [_msg("user", message), _msg("assistant", clean)] if clean else clean_history
    )
    yield "", base + [_msg("assistant", answer)], new_clean, event_id


def send_feedback(event_id: str | None, rating: str) -> str:
    """Thumbs feed the telemetry log, which is where gold set v4 and the embedding
    fine-tune pairs eventually come from — see kbm/telemetry.py."""
    if not event_id:
        return "Ask something first."
    try:
        httpx.post(
            f"{API_URL}/feedback",
            headers=_headers(),
            json={"event_id": event_id, "rating": rating},
            timeout=10.0,
        )
        return "Thanks — recorded."
    except Exception as e:
        return f"Could not record feedback: {e}"


# ── UI layout ──────────────────────────────────────────────────────────────────

# Gradio moved `theme` from Blocks to launch() in 6.0, and dropped Chatbot's `type`
# argument (messages is now the only format). requirements.txt pins gradio==6.17.3 — if
# this file raises TypeError on a Chatbot or Blocks argument, the environment is on 5.x
# and needs `pip install -r requirements.txt`, not a code change. See ERRORS.md.
# `analytics_enabled=False`: Gradio otherwise POSTs usage pings to its own servers on
# launch. Nothing here needs that, and a family's private pod should not be talking to a
# third party at all.
with gr.Blocks(title="Math Tutor", analytics_enabled=False) as app:
    gr.Markdown("# Math Tutor\nYour personal math knowledge base. Upload your textbooks and ask anything.")

    clean_history_state = gr.State([])  # LLM-facing history, no sources noise
    event_id_state = gr.State(None)     # last answer's telemetry id, for feedback
    last_ingested_state = gr.State(None)  # path already ingested; guards the auto-fire

    with gr.Row():
        username_box = gr.Textbox(label="Your name", placeholder="e.g. alice", scale=1)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload a document")
            upload_box = gr.File(label="PDF file", file_types=[".pdf"])
            upload_status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### Ask a question")
            chatbot = gr.Chatbot(height=500, allow_tags=False, latex_delimiters=[
                {"left": "$$", "right": "$$", "display": True},
                {"left": "$", "right": "$", "display": False},
                {"left": "\\(", "right": "\\)", "display": False},
                {"left": "\\[", "right": "\\]", "display": True},
            ])
            msg_box = gr.Textbox(label="Your question", placeholder="e.g. How do I solve a quadratic equation?")
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                up_btn = gr.Button("👍", scale=0)
                down_btn = gr.Button("👎", scale=0)
            feedback_status = gr.Markdown("")

    # `api_visibility="private"` on every binding below. Without it Gradio turns each event
    # into a named, externally callable route and lists it in the schema it serves at
    # /gradio_api/info — the whole backend surface, published to anyone who loads the page.
    # It lives inside these dicts so the multi-trigger bindings cannot drift apart.
    # ⚠️ On Gradio 5 this was `api_name=False`. Gradio 6 narrowed `api_name` to `str | None`
    # and moved visibility to `api_visibility`, but it does *not* reject the old value — it
    # accepts `api_name=False` and publishes the handler as a public endpoint literally
    # named `/False`, i.e. the exact opposite of what the argument used to mean. Silent
    # inversion, so verify with `Blocks.get_api_info()`, never by reading the argument.
    # This does not empty the page source: `window.gradio_config` still carries the
    # component tree and unnamed dependency indices, because that is how the Gradio client
    # bootstraps itself. What goes away is the documented, callable API.
    chat_io = dict(
        fn=handle_chat,
        inputs=[msg_box, chatbot, clean_history_state, username_box],
        outputs=[msg_box, chatbot, clean_history_state, event_id_state],
        api_visibility="private",
    )

    # Selecting a file *is* the request to ingest it — there is no separate button, and a
    # failure is answered by clearing the box and asking for another upload. Submitting the
    # name box is the one other trigger: it rescues the user who picked a file before
    # typing their name.
    #
    # `upload` (not `change`) is load-bearing. handle_upload clears the file box on
    # failure, and a programmatic value change fires `change` — which would re-enter this
    # handler with pdf_file=None and instantly overwrite the error with "No file selected."
    # `upload` fires only on a real user upload, so it cannot loop. The clear path that
    # `change` used to cover is wired explicitly below.
    upload_io = dict(
        fn=handle_upload,
        inputs=[upload_box, username_box, last_ingested_state],
        outputs=[upload_status, last_ingested_state, upload_box],
        api_visibility="private",
    )
    upload_box.upload(**upload_io)
    username_box.submit(**upload_io)
    # Clearing by hand forgets what was ingested, so re-selecting the same file works.
    upload_box.clear(
        lambda: ("No file selected.", None),
        outputs=[upload_status, last_ingested_state],
        api_visibility="private",
    )
    send_btn.click(**chat_io)
    msg_box.submit(**chat_io)
    up_btn.click(lambda eid: send_feedback(eid, "up"), inputs=event_id_state, outputs=feedback_status, api_visibility="private")
    down_btn.click(lambda eid: send_feedback(eid, "down"), inputs=event_id_state, outputs=feedback_status, api_visibility="private")


if __name__ == "__main__":
    # One configuration, shared with api/main.py — see kbm/logsetup.py. This was a second
    # basicConfig with its own format and a hardcoded level; it now honours KBM_LOG_LEVEL
    # like the API does, which it never did before.
    configure_logging()

    # Say which API this UI is pointed at. The default is loopback, which is right on a
    # single pod and silently wrong on a CPU host that has no API — there, every action
    # fails as a connection error and nothing anywhere names the cause. startup.sh
    # --ui-only refuses to start on the default for the same reason; this covers the
    # `python app.py` path, which has no such gate.
    log.info("API: %s (token %s)", API_URL, "set" if API_TOKEN else "UNSET")
    if RUNPOD_POD_ID and RUNPOD_API_KEY:
        log.info("Wake armed: can start pod %s on demand.", RUNPOD_POD_ID)

    # APP_AUTH gates the front door of this UI; KBM_API_TOKEN gates the API behind it.
    # They are different locks on different doors and both need setting on a public pod
    # — a login page in front of an open API only protects the page.
    app.launch(
        server_name=APP_HOST,
        server_port=APP_PORT,
        share=False,
        auth=app_auth(),
        theme=gr.themes.Soft(),
        # Drops the "Use via API" footer link. Gradio 6 replaced launch(show_api=False)
        # with footer_links, which names the links to keep rather than the one to remove;
        # passing show_api here is a TypeError. This is cosmetic — what actually empties
        # /gradio_api/info is api_visibility="private" on the bindings above.
        footer_links=["gradio", "settings"],
    )
