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
import os
import time

import gradio as gr
import httpx

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

def handle_upload(pdf_file, username: str):
    username = (username or "").strip().lower()
    if not username:
        return "Please enter your name before uploading."
    if pdf_file is None:
        return "No file uploaded."

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            with open(pdf_file.name, "rb") as fh:
                r = client.post(
                    f"{API_URL}/upload",
                    headers=_headers(),
                    data={"user": username},
                    files={"file": (os.path.basename(pdf_file.name), fh, "application/pdf")},
                )
            if r.status_code != 200:
                return f"Upload rejected: {r.text}"
            job_id = r.json()["job_id"]

            # Poll rather than block the request: Marker is minutes, not seconds.
            while True:
                time.sleep(3)
                job = client.get(f"{API_URL}/jobs/{job_id}", headers=_headers()).json()
                if job["status"] == "done":
                    return job["detail"]
                if job["status"] == "failed":
                    return f"Ingestion failed: {job['detail']}"
    except Exception as e:
        return f"Ingestion failed: {e}"


# ── Chat ──────────────────────────────────────────────────────────────────────

def handle_chat(message: str, history: list, clean_history: list, username: str):
    """Stream one answer from the API, yielding Gradio state as tokens arrive.

    Yields (cleared input, display history, clean history, event_id). `clean_history`
    holds the plain answers with the sources footer stripped — that is what gets sent
    back as conversation context, so the model never re-reads its own citation noise.
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

    answer = ""
    sources_text = ""
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

                    if kind == "sources":
                        if data["sources"]:
                            sources_text = "\n\n**Sources retrieved:**\n" + "\n".join(
                                f"- {s['source']} (score={s['score']:.4f})" for s in data["sources"]
                            )
                    elif kind == "token":
                        answer += data["text"]
                        yield "", base + [_msg("assistant", answer)], clean_history, event_id
                    elif kind == "done":
                        event_id = data["event_id"]
                    elif kind == "error":
                        raise RuntimeError(data["message"])
    except Exception as e:
        answer = f"Error: {e}"
        sources_text = ""

    new_clean = clean_history + [_msg("user", message), _msg("assistant", answer)]
    yield "", base + [_msg("assistant", answer + sources_text)], new_clean, event_id


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

# theme goes on Blocks, not launch(). The previous version of this file passed it to
# launch(), which raises TypeError on gradio 6.18 — `python app.py` crashed on startup.
# The deprecation warning says to move it to launch() "in Gradio 6.0", but this build
# rejects it there, so Blocks is the only thing that actually works today.
with gr.Blocks(title="Math Tutor", theme=gr.themes.Soft()) as app:
    gr.Markdown("# Math Tutor\nYour personal math knowledge base. Upload your textbooks and ask anything.")

    clean_history_state = gr.State([])  # LLM-facing history, no sources noise
    event_id_state = gr.State(None)     # last answer's telemetry id, for feedback

    with gr.Row():
        username_box = gr.Textbox(label="Your name", placeholder="e.g. alice", scale=1)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload a document")
            upload_box = gr.File(label="PDF file", file_types=[".pdf"])
            upload_btn = gr.Button("Ingest document")
            upload_status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### Ask a question")
            chatbot = gr.Chatbot(type="messages", height=500, allow_tags=False, latex_delimiters=[
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

    upload_btn.click(handle_upload, inputs=[upload_box, username_box], outputs=upload_status)
    send_btn.click(**chat_io)
    msg_box.submit(**chat_io)
    up_btn.click(lambda eid: send_feedback(eid, "up"), inputs=event_id_state, outputs=feedback_status)
    down_btn.click(lambda eid: send_feedback(eid, "down"), inputs=event_id_state, outputs=feedback_status)


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
