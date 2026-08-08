"""
app.py - Gradio web UI for the math RAG system.

Run:
    python app.py

Then open http://localhost:7860 in your browser.
Each user logs in with a username — their documents are fully isolated.
"""

import os
import tempfile

import gradio as gr
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from extract import extract
from chunking import split_baseline
from ingest import (
    build_bm25,
    build_chroma,
    load_mmd_files,
)
import retrieval
from retrieval import BM25_DIR, OLLAMA_MODEL, load_embeddings, load_reranker

# Prompt order is load-bearing for latency: static text → history → context → question.
# Ollama caches the KV of the longest common prompt *prefix* between consecutive requests.
# Everything up to the first token that changes is free; everything after it is re-prefilled.
# The static block never changes and `history` only ever grows by appending, so both stay
# cached across turns. `context` is fresh every query, which is exactly why it must come
# last — it lives in the human message, after the system message. Putting it before the
# history (as this file used to) invalidated the cache at token ~150 and re-prefilled the
# whole ~3.4k-token prompt every single turn. See LATENCY.md.
SYSTEM_PROMPT = """You are an expert mathematician and dedicated teacher. Your deep love for mathematics drives you to help students not just find answers, but truly understand the underlying concepts and develop their own mathematical thinking.

When answering:
- Don't just solve the problem — explain the reasoning behind each step so the student understands why, not just how.
- If a student makes a conceptual error, gently point it out and guide them toward the correct understanding.
- Encourage curiosity: point out interesting patterns, connections to other concepts, or follow-up questions worth thinking about.
- Show your working step by step, but match the length of the answer to the question: a conceptual "why" question wants a short, clear explanation, not a full derivation. Stop once the student has what they asked for.
- Use LaTeX for all equations (e.g. $x^2$, \\frac{{a}}{{b}}).
- If context from uploaded documents is provided, prioritise it and cite the source. Otherwise answer from your own expertise.

Conversation so far:
{history}"""

HUMAN_PROMPT = """{context}

Question: {input}"""

# History is trimmed in blocks, not one message at a time — see _history_window().
# These count *messages* (a turn is two: student + tutor).
HISTORY_KEEP = 8    # smallest the window is ever trimmed back to
HISTORY_BLOCK = 8   # the window start only ever moves in steps of this

NUM_PREDICT = 350   # decode is ~60% of query latency and the solver will fill whatever it is given
KEEP_ALIVE = "30m"  # else Ollama unloads after 5 min idle and the next question pays a cold load

# Loaded once at startup, shared across all users (stateless)
embeddings = load_embeddings()
llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
    num_predict=NUM_PREDICT,
    keep_alive=KEEP_ALIVE,
)
reranker = load_reranker()
chain = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
]) | llm | StrOutputParser()


# ── Retrieval ──────────────────────────────────────────────────────────────────
# The pipeline itself (BM25 + dense → RRF → cross-encoder) lives in retrieval.py,
# shared with query.py and eval.py. Only the UI-specific bits are here.

def _has_index(user: str) -> bool:
    return os.path.exists(os.path.join(BM25_DIR, f"user_{user}.pkl"))


def retrieve(query: str, user: str) -> list[tuple[Document, float]]:
    return retrieval.retrieve(query, user, embeddings, reranker=reranker)


# ── Gradio handlers ────────────────────────────────────────────────────────────

def handle_upload(pdf_file, username: str) -> str:
    username = username.strip().lower()
    if not username:
        return "Please enter your name before uploading."
    if pdf_file is None:
        return "No file uploaded."

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            mmd_out_dir = os.path.join(tmp_dir, "extracted")
            mmd_path = extract(pdf_file.name, out_dir=mmd_out_dir)

            docs = load_mmd_files([mmd_path])
            chunks = split_baseline(docs)

            # Merge with existing BM25 index if user already has one
            if _has_index(username):
                _, existing_chunks = retrieval.load_bm25(username)
                chunks = existing_chunks + chunks

            build_bm25(chunks, username)
            build_chroma(chunks, username, embeddings)

        fname = os.path.basename(pdf_file.name)
        return f"'{fname}' ingested successfully. {len(chunks)} total chunks indexed for {username}."
    except Exception as e:
        return f"Ingestion failed: {e}"


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def _history_window(history: list) -> list:
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


def _format_history(history: list) -> str:
    lines = []
    for m in _history_window(history):
        role = "Student" if m["role"] == "user" else "Tutor"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines) if lines else "None yet."


def handle_chat(message: str, history: list, clean_history: list, username: str):
    """Stream one answer, yielding (cleared input, display history, clean history).

    A generator rather than a plain function: a full answer takes tens of seconds to
    decode, and waiting for all of it before painting anything is the single largest
    contributor to *perceived* latency. Streaming shows the first words in ~1s instead.
    Gradio treats a yielded tuple exactly like a returned one and repaints per yield.
    """
    username = username.strip().lower()
    if not username:
        yield "", history + [_msg("user", message), _msg("assistant", "Please enter your name first.")], clean_history
        return
    if not message.strip():
        yield "", history, clean_history
        return

    chat_history = _format_history(clean_history)
    base = history + [_msg("user", message)]

    # Paint the question and a placeholder before retrieval, so the UI reacts immediately.
    yield "", base + [_msg("assistant", "_Searching your documents…_")], clean_history

    answer = ""
    sources_text = ""
    try:
        if _has_index(username):
            results = retrieve(message, username)
            context = "Context from your documents:\n\n" + "\n\n".join(
                f"[Source: {os.path.basename(doc.metadata.get('source', 'unknown'))}]\n{doc.page_content}"
                for doc, _ in results
            )
            sources_text = "\n\n**Sources retrieved:**\n" + "\n".join(
                f"- {os.path.basename(doc.metadata.get('source', 'unknown'))} (rerank={score:.4f})"
                for doc, score in results
            )
        else:
            context = "No documents uploaded yet. Answer from your own expertise."

        for token in chain.stream({"context": context, "history": chat_history, "input": message}):
            answer += token
            yield "", base + [_msg("assistant", answer)], clean_history

    except Exception as e:
        answer = f"Error: {e}"
        sources_text = ""

    full_answer = answer + sources_text
    # clean_history stores only the plain answer (no sources) for LLM context
    new_clean = clean_history + [_msg("user", message), _msg("assistant", answer)]
    yield "", base + [_msg("assistant", full_answer)], new_clean


# ── UI layout ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Math Tutor") as app:
    gr.Markdown("# Math Tutor\nYour personal math knowledge base. Upload your textbooks and ask anything.")

    clean_history_state = gr.State([])  # LLM-facing history, no sources noise

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
            chatbot = gr.Chatbot(height=500, latex_delimiters=[
                {"left": "$$", "right": "$$", "display": True},
                {"left": "$", "right": "$", "display": False},
                {"left": "\\(", "right": "\\)", "display": False},
                {"left": "\\[", "right": "\\]", "display": True},
            ])
            msg_box = gr.Textbox(label="Your question", placeholder="e.g. How do I solve a quadratic equation?")
            send_btn = gr.Button("Send", variant="primary")

    upload_btn.click(handle_upload, inputs=[upload_box, username_box], outputs=upload_status)
    send_btn.click(handle_chat, inputs=[msg_box, chatbot, clean_history_state, username_box], outputs=[msg_box, chatbot, clean_history_state])
    msg_box.submit(handle_chat, inputs=[msg_box, chatbot, clean_history_state, username_box], outputs=[msg_box, chatbot, clean_history_state])


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
