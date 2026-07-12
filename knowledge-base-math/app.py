"""
app.py - Gradio web UI for the math RAG system.

Run:
    python app.py

Then open http://localhost:7860 in your browser.
Each user logs in with a username — their documents are fully isolated.
"""

import os
import pickle
import tempfile
from collections import defaultdict

import gradio as gr
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from extract import extract
from ingest import build_bm25, build_chroma, load_mmd_files

CHROMA_DIR = "chroma_db"
BM25_DIR = "bm25_indexes"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
OLLAMA_MODEL = "t1c/deepseek-math-7b-rl:Q4"
TOP_K = 10
RERANK_TOP_C = 20  # candidate pool taken from RRF and fed to the reranker
TOP_N = 5
RRF_K = 60

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
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, num_predict=1024)
reranker = CrossEncoder(RERANK_MODEL)
chain = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
]) | llm | StrOutputParser()


# ── Retrieval helpers ──────────────────────────────────────────────────────────

def _bm25_path(user: str) -> str:
    return os.path.join(BM25_DIR, f"user_{user}.pkl")


def _has_index(user: str) -> bool:
    return os.path.exists(_bm25_path(user))


def _load_bm25(user: str) -> tuple[BM25Okapi, list[Document]]:
    with open(_bm25_path(user), "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"]


def _rrf(ranked_lists: list[list[Document]], k: int = RRF_K) -> list[tuple[Document, float]]:
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            key = doc.page_content[:120]
            scores[key] += 1 / (k + rank)
            doc_map[key] = doc
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [(doc_map[key], scores[key]) for key in sorted_keys]


def _rerank(query: str, candidates: list[tuple[Document, float]], top_n: int = TOP_N) -> list[tuple[Document, float]]:
    """Rescore RRF candidates with the cross-encoder and keep the top_n.

    The cross-encoder reads each (query, chunk) pair together in a single
    forward pass and emits a relevance logit; we sort by that and truncate.
    """
    if not candidates:
        return []
    pairs = [(query, doc.page_content) for doc, _ in candidates]
    scores = reranker.predict(pairs)
    reranked = sorted(zip((doc for doc, _ in candidates), scores),
                      key=lambda x: x[1], reverse=True)
    return [(doc, float(score)) for doc, score in reranked[:top_n]]


def retrieve(query: str, user: str) -> list[tuple[Document, float]]:
    bm25, chunks = _load_bm25(user)
    bm25_scores = bm25.get_scores(query.split())
    top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K]
    bm25_results = [chunks[i] for i in top_idx]

    store = Chroma(collection_name=f"user_{user}", embedding_function=embeddings, persist_directory=CHROMA_DIR)
    dense_results = store.similarity_search(query, k=TOP_K)

    merged = _rrf([bm25_results, dense_results])
    return _rerank(query, merged[:RERANK_TOP_C], TOP_N)


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
            splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
            chunks = splitter.split_documents(docs)

            # Merge with existing BM25 index if user already has one
            if _has_index(username):
                with open(_bm25_path(username), "rb") as f:
                    existing = pickle.load(f)
                chunks = existing["chunks"] + chunks

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
