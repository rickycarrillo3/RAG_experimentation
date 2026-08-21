"""
test_chat.py - Ingest test.mmd then start an interactive CLI chat.

Usage:
    python test_chat.py
    python test_chat.py --retrieval-only   # skip LLM, print chunks only
    python test_chat.py --selftest         # run the pure regression checks and exit

`--selftest` covers the two invariants that fail *silently* if broken, which is why they
are checked here rather than left to a manual pass: the prefill-echo filter (whose failure
mode is showing the student a duplicated answer) and chunk-id stability (whose failure mode
is every re-upload duplicating a document in the index instead of replacing it). Neither
needs a GPU, a model, or a network.
"""

import argparse
import os
import sys

from langchain_ollama import ChatOllama

from ingest import ingest
from kbm.config import OLLAMA_BASE_URL
from kbm.retrieval import load_embeddings
from query import (
    KEEP_ALIVE,
    NUM_PREDICT,
    OLLAMA_MODEL,
    build_answer_chain,
    format_context,
    print_retrieved,
    retrieve,
)

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


def selftest() -> int:
    """Pure checks: no model, no network, no GPU. Returns a process exit code."""
    import tempfile

    from api.chat import PrefillEcho
    from ingest import load_mmd_files, merge_chunks
    from kbm.chunking import split_baseline

    failures = []

    def check(name, cond):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        if not cond:
            failures.append(name)

    def echo_run(prefill, chunks):
        e = PrefillEcho(prefill)
        return "".join(e.feed(c) for c in chunks), e.mismatch

    print("PrefillEcho — strips the prefill Ollama echoes back on a continuation")
    check("first pass (no prefill) passes text through", echo_run("", ["a", "b"]) == ("ab", False))
    check("echo + new text in one chunk", echo_run("Hello world", ["Hello world and more"]) == (" and more", False))
    check("echo split across chunks", echo_run("Hello world", ["Hel", "lo ", "world", " tail"]) == (" tail", False))
    check("leading space preserved", echo_run(" The chain rule", [" The chain rule is"]) == (" is", False))
    # The one that matters: on divergence it must emit NOTHING, so the caller can abandon
    # the continuation rather than showing the answer twice.
    check("diverged echo -> mismatch, no output", echo_run("Hello world", ["Hello there"]) == ("", True))
    check("divergence caught before full length", echo_run("Hello world", ["Hex"]) == ("", True))

    print("chunk ids — must not depend on the (now temporary) directory")
    text = "The chain rule states that $$\\frac{d}{dx}f(g(x)) = f'(g(x))g'(x)$$.\n\n" * 20

    def chunks_from_tempdir():
        d = tempfile.mkdtemp(prefix="kbm_selftest_")
        path = os.path.join(d, "calculus.mmd")
        with open(path, "w") as fh:
            fh.write(text)
        docs = load_mmd_files([path])
        for doc in docs:
            doc.metadata["source"] = os.path.basename(path)
        return split_baseline(docs)

    a, b = chunks_from_tempdir(), chunks_from_tempdir()
    ids_a = [c.metadata["chunk_id"] for c in a]
    check("ids identical across two temp dirs", ids_a == [c.metadata["chunk_id"] for c in b])
    check("ids are basename-derived", ids_a[:1] == ["calculus.mmd::0"])
    check("re-upload replaces rather than duplicates", len(merge_chunks(a, b)) == len(a))

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
        return 1
    print("all checks passed")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--selftest", action="store_true",
                        help="Run the pure regression checks and exit (no model needed)")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(selftest())

    if not os.path.exists(TEST_DOC):
        print(f"Test document not found: {TEST_DOC}")
        sys.exit(1)

    print(f"Ingesting {TEST_DOC} for user '{TEST_USER}'...")
    ingest(user=TEST_USER, inputs=[TEST_DOC])

    print(f"\nLoading embedding model...")
    # Via load_embeddings, not a second HuggingFaceEmbeddings(...) — the smoke test is
    # only worth anything if it exercises the same loader (and so the same device
    # pinning and LaTeX-normalization behaviour) that app.py and query.py use.
    embeddings = load_embeddings(EMBED_MODEL)

    answer_chain = None
    if not args.retrieval_only:
        print(f"Connecting to Ollama ({OLLAMA_MODEL}) at {OLLAMA_BASE_URL}...")
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            num_predict=NUM_PREDICT,
            keep_alive=KEEP_ALIVE,
        )
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
        print("\nTutor: ", end="", flush=True)
        answer = ""
        for token in answer_chain.stream({
            "context": f"Context from your documents:\n\n{context}" if context else "",
            "history": chat_history,
            "input": question,
        }):
            answer += token
            print(token, end="", flush=True)
        print("\n")
        history.append((question, answer))


if __name__ == "__main__":
    main()
