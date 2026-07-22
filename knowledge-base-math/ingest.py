"""
ingest.py - Chunk, embed, and index .mmd files into per-user BM25 + ChromaDB indexes.

Run after extract.py:
    python ingest.py --user alice docs/extracted/textbook.mmd
    python ingest.py --user alice docs/extracted/  (ingests all .mmd in a directory)

Chunking and embedding are selectable so the retrieval sweep can build one index per
(chunking x embedding) combination:
    python ingest.py --user alice_eqm3 docs/extracted/ \
        --chunker eqaware --embed-model BAAI/bge-m3 --normalize-latex
"""

import argparse
import glob
import os
import pickle
import random
import sys
import time

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from chunking import CHUNKERS, assign_chunk_ids  # assign_chunk_ids re-exported for callers
from retrieval import EMBED_MODEL, load_embeddings

CHROMA_DIR = "chroma_db"
BM25_DIR = "bm25_indexes"


def load_mmd_files(paths: list[str]) -> list[Document]:
    documents = []
    for path in paths:
        loader = TextLoader(path, encoding="utf-8")
        documents.extend(loader.load())
    return documents


def build_bm25(chunks: list[Document], user: str) -> str:
    tokenized = [doc.page_content.split() for doc in chunks]
    bm25 = BM25Okapi(tokenized)

    os.makedirs(BM25_DIR, exist_ok=True)
    pkl_path = os.path.join(BM25_DIR, f"user_{user}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    return pkl_path


def build_chroma(chunks: list[Document], user: str, embeddings) -> Chroma:
    collection_name = f"user_{user}"
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    store.add_documents(chunks)
    return store


def resolve_paths(inputs: list[str]) -> list[str]:
    paths = []
    for inp in inputs:
        if os.path.isdir(inp):
            paths.extend(glob.glob(os.path.join(inp, "*.mmd")))
        elif inp.endswith(".mmd") and os.path.isfile(inp):
            paths.append(inp)
        else:
            print(f"Warning: skipping {inp} (not a .mmd file or directory)")
    return paths


def ingest(
    user: str,
    inputs: list[str],
    chunker: str = "baseline",
    embed_model: str = EMBED_MODEL,
    normalize_latex: bool = False,
) -> dict:
    """Build a user's BM25 + Chroma indexes. Returns build stats for the eval sweep."""
    if chunker not in CHUNKERS:
        raise ValueError(f"Unknown chunker '{chunker}'. Choose from: {', '.join(CHUNKERS)}")

    paths = resolve_paths(inputs)
    if not paths:
        print("No .mmd files found. Run extract.py first.")
        sys.exit(1)

    print(f"Loading {len(paths)} file(s)...")
    documents = load_mmd_files(paths)

    t0 = time.perf_counter()
    chunks = CHUNKERS[chunker](documents)
    print(f"Split into {len(chunks)} chunks with '{chunker}' chunker.")

    print(f"Building BM25 index for user '{user}'...")
    pkl_path = build_bm25(chunks, user)
    print(f"BM25 index saved to {pkl_path}")

    print(f"Loading embedding model '{embed_model}' (normalize_latex={normalize_latex})...")
    embeddings = load_embeddings(embed_model, normalize_latex=normalize_latex)

    print(f"Embedding chunks into ChromaDB collection 'user_{user}'...")
    build_chroma(chunks, user, embeddings)
    build_seconds = time.perf_counter() - t0
    print(f"ChromaDB collection 'user_{user}' ready.")

    print("\n--- Sample chunks (3 random) ---")
    for chunk in random.sample(chunks, min(3, len(chunks))):
        preview = chunk.page_content[:200].replace("\n", " ")
        print(f"  [{chunk.metadata.get('source', 'unknown')}] {preview}...")

    print(f"\nDone. {len(chunks)} chunks indexed for user '{user}' in {build_seconds:.1f}s.")
    return {
        "user": user,
        "chunker": chunker,
        "embed_model": embed_model,
        "normalize_latex": normalize_latex,
        "n_chunks": len(chunks),
        "build_seconds": round(build_seconds, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest .mmd files into per-user BM25 + ChromaDB indexes.")
    parser.add_argument("--user", required=True, help="Username (determines which collection to write to)")
    parser.add_argument("inputs", nargs="+", help=".mmd file(s) or directory containing .mmd files")
    parser.add_argument("--chunker", default="baseline", choices=list(CHUNKERS),
                        help="Chunking strategy (default: baseline)")
    parser.add_argument("--embed-model", default=EMBED_MODEL, help="HuggingFace embedding model")
    parser.add_argument("--normalize-latex", action="store_true",
                        help="LaTeX-normalize chunk text before embedding (page_content stays raw)")
    args = parser.parse_args()

    ingest(
        user=args.user,
        inputs=args.inputs,
        chunker=args.chunker,
        embed_model=args.embed_model,
        normalize_latex=args.normalize_latex,
    )
