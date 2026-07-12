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
from langchain_text_splitters import RecursiveCharacterTextSplitter

from extract import extract
from ingest import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    assign_chunk_ids,
    build_bm25,
    build_chroma,
    load_mmd_files,
)
import retrieval
from retrieval import BM25_DIR, OLLAMA_MODEL, load_embeddings, load_reranker

SYSTEM_PROMPT = """You are an expert mathematician and dedicated teacher. Your deep love for mathematics drives you to help students not just find answers, but truly understand the underlying concepts and develop their own mathematical thinking.

When answering:
- Don't just solve the problem — explain the reasoning behind each step so the student understands why, not just how.
- If a student makes a conceptual error, gently point it out and guide them toward the correct understanding.
- Encourage curiosity: point out interesting patterns, connections to other concepts, or follow-up questions worth thinking about.
- Always show full working step by step.
- Use LaTeX for all equations (e.g. $x^2$, \\frac{{a}}{{b}}).
- If context from uploaded documents is provided, prioritise it and cite the source. Otherwise answer from your own expertise.

{context}

Conversation so far:
{history}"""

HISTORY_TURNS = 6

# Loaded once at startup, shared across all users (stateless)
embeddings = load_embeddings()
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, num_predict=1024)
reranker = load_reranker()
chain = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
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
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            chunks = assign_chunk_ids(splitter.split_documents(docs))

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


def _format_history(history: list) -> str:
    recent = history[-HISTORY_TURNS:]
    lines = []
    for m in recent:
        role = "Student" if m["role"] == "user" else "Tutor"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines) if lines else "None yet."


def handle_chat(message: str, history: list, clean_history: list, username: str) -> tuple[str, list, list]:
    username = username.strip().lower()
    if not username:
        display = history + [_msg("user", message), _msg("assistant", "Please enter your name first.")]
        return "", display, clean_history
    if not message.strip():
        return "", history, clean_history

    chat_history = _format_history(clean_history)
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

        answer = chain.invoke({"context": context, "history": chat_history, "input": message})
        full_answer = answer + sources_text

    except Exception as e:
        answer = f"Error: {e}"
        full_answer = answer

    # clean_history stores only the plain answer (no sources) for LLM context
    new_clean = clean_history + [_msg("user", message), _msg("assistant", answer)]
    new_display = history + [_msg("user", message), _msg("assistant", full_answer)]
    return "", new_display, new_clean


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
