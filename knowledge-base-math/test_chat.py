"""
test_chat.py - Ingest test.mmd then start an interactive CLI chat.

Usage:
    python test_chat.py
    python test_chat.py --retrieval-only   # skip LLM, print chunks only
"""

import argparse
import os
import sys

from langchain_huggingface import HuggingFaceEmbeddings

from ingest import ingest
from query import retrieve, build_answer_chain, format_context, print_retrieved, OLLAMA_MODEL
from langchain_ollama import ChatOllama

TEST_USER = "test"
TEST_DOC = "docs/extracted/test.mmd"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
HISTORY_TURNS = 6


def _format_history(history: list[tuple[str, str]]) -> str:
    recent = history[-HISTORY_TURNS:]
    lines = []
    for user_msg, bot_msg in recent:
        lines.append(f"Student: {user_msg}")
        lines.append(f"Tutor: {bot_msg}")
    return "\n".join(lines) if lines else "None yet."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(TEST_DOC):
        print(f"Test document not found: {TEST_DOC}")
        sys.exit(1)

    print(f"Ingesting {TEST_DOC} for user '{TEST_USER}'...")
    ingest(user=TEST_USER, inputs=[TEST_DOC])

    print(f"\nLoading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    answer_chain = None
    if not args.retrieval_only:
        print(f"Connecting to Ollama ({OLLAMA_MODEL})...")
        llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, num_predict=1024)
        answer_chain = build_answer_chain(llm)

    history = []
    mode = "retrieval-only" if args.retrieval_only else "full pipeline"
    print(f"\nReady [{mode}]. Type your question or 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        results = retrieve(question, user=TEST_USER, embeddings=embeddings)
        print_retrieved(results)

        if args.retrieval_only:
            continue

        context = format_context(results)
        chat_history = _format_history(history)
        answer = answer_chain.invoke({
            "context": f"Context from your documents:\n\n{context}" if context else "",
            "history": chat_history,
            "input": question,
        })

        print(f"\nTutor: {answer}\n")
        history.append((question, answer))


if __name__ == "__main__":
    main()
