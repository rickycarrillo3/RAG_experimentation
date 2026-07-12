"""
query.py - Hybrid BM25 + dense retrieval with RRF, then DeepSeek-Math-7B generation.

Usage:
    python query.py --user alice                         # full pipeline (retrieval + LLM)
    python query.py --user alice --retrieval-only        # print chunks, skip LLM
    python query.py --user alice --no-bm25               # dense only (for comparison)
    python query.py --user alice --no-dense              # BM25 only (for comparison)
"""

import argparse
import os
import pickle
import sys
from collections import defaultdict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

CHROMA_DIR = "chroma_db"
BM25_DIR = "bm25_indexes"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
OLLAMA_MODEL = "t1c/deepseek-math-7b-rl:Q4"
TOP_K = 10        # candidates from each retriever
RERANK_TOP_C = 20 # candidate pool taken from RRF and fed to the reranker
TOP_N = 5         # final chunks passed to LLM after rerank
RRF_K = 60        # RRF smoothing constant

SYSTEM_PROMPT = """You are a patient math tutor helping a student or family member understand mathematics.

Rules:
- Answer only based on the context provided below.
- If the answer is not in the context, say: "I don't have that in my knowledge base."
- Always show your work step by step.
- Use LaTeX notation for all equations (e.g. $x^2$, \\frac{{a}}{{b}}).
- Cite the source document at the end of your answer.

Context:
{context}"""


# ── Retrieval ──────────────────────────────────────────────────────────────────

def load_bm25(user: str) -> tuple[BM25Okapi, list[Document]]:
    pkl_path = os.path.join(BM25_DIR, f"user_{user}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"No BM25 index for user '{user}'. Run ingest.py first.")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"]


def load_chroma(user: str, embeddings) -> Chroma:
    return Chroma(
        collection_name=f"user_{user}",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def bm25_search(query: str, bm25: BM25Okapi, chunks: list[Document], k: int) -> list[Document]:
    scores = bm25.get_scores(query.split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in top_indices]


def dense_search(query: str, store: Chroma, k: int) -> list[Document]:
    return store.similarity_search(query, k=k)


def reciprocal_rank_fusion(ranked_lists: list[list[Document]], k: int = RRF_K) -> list[Document]:
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            key = doc.page_content[:120]
            scores[key] += 1 / (k + rank)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [(doc_map[key], scores[key]) for key in sorted_keys]


def load_reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL)


def rerank(
    query: str,
    candidates: list[tuple[Document, float]],
    reranker: CrossEncoder,
    top_n: int = TOP_N,
) -> list[tuple[Document, float]]:
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


def retrieve(
    query: str,
    user: str,
    embeddings,
    use_bm25: bool = True,
    use_dense: bool = True,
    reranker: CrossEncoder | None = None,
    use_rerank: bool = True,
) -> list[tuple[Document, float]]:
    ranked_lists = []

    if use_bm25:
        bm25, chunks = load_bm25(user)
        bm25_results = bm25_search(query, bm25, chunks, k=TOP_K)
        ranked_lists.append(bm25_results)

    if use_dense:
        store = load_chroma(user, embeddings)
        dense_results = dense_search(query, store, k=TOP_K)
        ranked_lists.append(dense_results)

    if not ranked_lists:
        raise ValueError("At least one of --no-bm25 or --no-dense must be left enabled.")

    merged = reciprocal_rank_fusion(ranked_lists)

    if use_rerank and reranker is not None:
        return rerank(query, merged[:RERANK_TOP_C], reranker, top_n=TOP_N)
    return merged[:TOP_N]


# ── Generation ────────────────────────────────────────────────────────────────

def build_answer_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    return prompt | llm | StrOutputParser()


def format_context(results: list[tuple[Document, float]]) -> str:
    parts = []
    for doc, _ in results:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


# ── CLI ───────────────────────────────────────────────────────────────────────

def print_retrieved(results: list[tuple[Document, float]], reranked: bool = True) -> None:
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
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    reranker = None
    if not args.no_rerank:
        print(f"Loading reranker model '{RERANK_MODEL}'...")
        reranker = load_reranker()

    llm = None
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
