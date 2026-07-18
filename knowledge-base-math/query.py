"""
query.py - CLI for the math RAG system.

Retrieval lives in retrieval.py (shared with app.py and eval.py); this file is the
interactive shell around it plus DeepSeek-Math generation.

Usage:
    python query.py --user alice                         # full pipeline (retrieval + LLM)
    python query.py --user alice --retrieval-only        # print chunks, skip LLM
    python query.py --user alice --no-bm25               # dense only (for comparison)
    python query.py --user alice --no-dense              # BM25 only (for comparison)
    python query.py --user alice --no-rerank             # skip the cross-encoder (RRF order)
"""

import argparse
import os
import sys

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieval import (
    EMBED_MODEL,
    OLLAMA_MODEL,
    RERANK_MODEL,
    format_context,
    load_embeddings,
    load_reranker,
    retrieve,
)

SYSTEM_PROMPT = """You are a patient math tutor helping a student or family member understand mathematics.

Rules:
- Answer only based on the context provided below.
- If the answer is not in the context, say: "I don't have that in my knowledge base."
- Always show your work step by step.
- Use LaTeX notation for all equations (e.g. $x^2$, \\frac{{a}}{{b}}).
- Cite the source document at the end of your answer.

Context:
{context}"""


# ── Generation ────────────────────────────────────────────────────────────────

def build_answer_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    return prompt | llm | StrOutputParser()


# ── CLI ───────────────────────────────────────────────────────────────────────

def print_retrieved(results, reranked: bool = True) -> None:
    label = "reranked" if reranked else "RRF merged"
    score_label = "rerank" if reranked else "RRF"
    print(f"\n{'─'*60}")
    print(f"Retrieved {len(results)} chunks ({label}):")
    for i, (doc, score) in enumerate(results, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        preview = doc.page_content[:150].replace("\n", " ")
        print(f"\n  [{i}] {score_label}={score:.4f} | {source}")
        print(f"      {preview}...")
    print(f"{'─'*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Query the math RAG system.")
    parser.add_argument("--user", required=True, help="Username (determines which index to search)")
    parser.add_argument("--retrieval-only", action="store_true", help="Print retrieved chunks only, skip LLM")
    parser.add_argument("--no-bm25", action="store_true", help="Disable BM25 (dense only)")
    parser.add_argument("--no-dense", action="store_true", help="Disable dense search (BM25 only)")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder rerank (RRF order only)")
    args = parser.parse_args()

    print(f"Loading embedding model '{EMBED_MODEL}'...")
    embeddings = load_embeddings()

    reranker = None
    if not args.no_rerank:
        print(f"Loading reranker model '{RERANK_MODEL}'...")
        reranker = load_reranker()

    answer_chain = None
    if not args.retrieval_only:
        print(f"Connecting to Ollama model '{OLLAMA_MODEL}'...")
        llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, num_predict=1024)
        answer_chain = build_answer_chain(llm)

    mode = "retrieval-only" if args.retrieval_only else "full pipeline"
    bm25_status = "off" if args.no_bm25 else "on"
    dense_status = "off" if args.no_dense else "on"
    rerank_status = "off" if args.no_rerank else "on"
    print(f"\nReady [{mode}] | BM25: {bm25_status} | Dense: {dense_status} | Rerank: {rerank_status}")
    print("Type your question (or 'quit' to exit).\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        try:
            results = retrieve(
                question,
                user=args.user,
                embeddings=embeddings,
                use_bm25=not args.no_bm25,
                use_dense=not args.no_dense,
                reranker=reranker,
                use_rerank=not args.no_rerank,
            )
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            break

        print_retrieved(results, reranked=not args.no_rerank)

        if args.retrieval_only:
            continue

        context = format_context(results)
        answer = answer_chain.invoke({"context": context, "input": question})
        sources = {os.path.basename(doc.metadata.get("source", "unknown")) for doc, _ in results}
        print(f"Answer:\n{answer}")
        print(f"\nSources: {', '.join(sources)}\n")


if __name__ == "__main__":
    main()
