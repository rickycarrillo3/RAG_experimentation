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

from config import APP_HOST, APP_PORT, app_auth

log = logging.getLogger(__name__)

API_URL = os.environ.get("KBM_API_URL", "http://127.0.0.1:8000")
API_TOKEN = os.environ.get("KBM_API_TOKEN", "")
# Generation of a full answer runs to tens of seconds; the default httpx timeout would
# abort mid-stream. Read timeout is the one that matters for SSE.
TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=600.0, pool=10.0)


def _headers() -> dict:
    return {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}


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
    instead of an empty box. Yields (status_text, last_ingested); `last_ingested` is what
    makes auto-fire safe — see the guard below.
    """
    username = (username or "").strip().lower()

    if pdf_file is None:
        # Also the "clear" path: forget what we ingested so re-selecting it works.
        yield "No file selected.", None
        return
    if not username:
        # Named recovery action, because this fires on file selection now: a user who
        # picks the file first would otherwise hit a dead end with nothing to re-trigger.
        yield "Enter your name above, then press Enter (or click Ingest / retry).", last_ingested
        return
    if pdf_file.name == last_ingested:
        # This function is wired to three triggers, one of which is the name box's submit
        # event, so it can fire repeatedly for one file. Without this guard every Enter
        # press would re-run Marker — minutes of GPU for a document already indexed.
        yield f"Already ingested {os.path.basename(pdf_file.name)}. Choose another file to add more.", last_ingested
        return

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            yield f"Uploading {os.path.basename(pdf_file.name)}...", last_ingested
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
                yield f"Could not upload {os.path.basename(pdf_file.name)} — {why}", last_ingested
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
                    yield prefix + job["detail"] + _diagnostic(job), pdf_file.name
                    return
                if status == "failed":
                    yield "❌ " + job["detail"] + _diagnostic(job), last_ingested
                    return

                if elapsed > UPLOAD_POLL_TIMEOUT:
                    yield (
                        f"Still working after {elapsed / 60:.0f} minutes. The import has "
                        f"not been cancelled — it should finish on its own.",
                        last_ingested,
                    )
                    return

                yield f"{status.title()}... ({elapsed / 60:.1f} min elapsed)", last_ingested
    except Exception as e:
        # httpx/JSON errors from the polling loop itself. The upload may well still be
        # running on the server, so do not claim it failed.
        log.warning("upload status polling failed", exc_info=e)
        yield (
            "Lost contact with the server while importing. The import may still be "
            "running — wait a moment, then press Ingest / retry to check.",
            last_ingested,
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
    fine-tune pairs eventually come from — see telemetry.py."""
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
with gr.Blocks(title="Math Tutor") as app:
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
            upload_btn = gr.Button("Ingest / retry")
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

    chat_io = dict(
        fn=handle_chat,
        inputs=[msg_box, chatbot, clean_history_state, username_box],
        outputs=[msg_box, chatbot, clean_history_state, event_id_state],
    )

    # Three triggers, one handler. Selecting a file starts ingestion immediately, which
    # is the common case; submitting the name box rescues the user who picked the file
    # before typing their name; the button stays as an explicit retry for when the API
    # was down. `change` (not `upload`) because it also fires on clear, which is what
    # resets last_ingested_state.
    upload_io = dict(
        fn=handle_upload,
        inputs=[upload_box, username_box, last_ingested_state],
        outputs=[upload_status, last_ingested_state],
    )
    upload_box.change(**upload_io)
    username_box.submit(**upload_io)
    upload_btn.click(**upload_io)
    send_btn.click(**chat_io)
    msg_box.submit(**chat_io)
    up_btn.click(lambda eid: send_feedback(eid, "up"), inputs=event_id_state, outputs=feedback_status)
    down_btn.click(lambda eid: send_feedback(eid, "down"), inputs=event_id_state, outputs=feedback_status)


if __name__ == "__main__":
    # APP_AUTH gates the front door of this UI; KBM_API_TOKEN gates the API behind it.
    # They are different locks on different doors and both need setting on a public pod
    # — a login page in front of an open API only protects the page.
    app.launch(
        server_name=APP_HOST,
        server_port=APP_PORT,
        share=False,
        auth=app_auth(),
        theme=gr.themes.Soft(),
    )
