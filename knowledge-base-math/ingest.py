"""
ingest.py - Chunk, embed, and index .mmd files into per-user BM25 + ChromaDB indexes.

Run after extract.py:
    python ingest.py --user alice docs/extracted/textbook.mmd
    python ingest.py --user alice docs/extracted/  (ingests all .mmd in a directory)
"""

import argparse
import glob
import os
import pickle
import random
import sys

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

CHROMA_DIR = "chroma_db"
BM25_DIR = "bm25_indexes"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80


def load_mmd_files(paths: list[str]) -> list[Document]:
    documents = []
    for path in paths:
        loader = TextLoader(path, encoding="utf-8")
        documents.extend(loader.load())
    return documents


def assign_chunk_ids(chunks: list[Document]) -> list[Document]:
    """Stamp a deterministic `<source>::<n>` id onto each chunk.

    The eval gold set labels questions by chunk id, so ids must be reproducible: the
    same .mmd re-ingested must yield the same ids, or every gold label silently rots.
    Counting per source (not globally) keeps ids stable when an unrelated file is added.
    """
    counters: dict[str, int] = {}
    for chunk in chunks:
        source = os.path.basename(chunk.metadata.get("source", "unknown"))
        n = counters.get(source, 0)
        chunk.metadata["chunk_id"] = f"{source}::{n}"
        counters[source] = n + 1
    return chunks


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


def ingest(user: str, inputs: list[str]) -> None:
    paths = resolve_paths(inputs)
    if not paths:
        print("No .mmd files found. Run extract.py first.")
        sys.exit(1)

    print(f"Loading {len(paths)} file(s)...")
    documents = load_mmd_files(paths)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = assign_chunk_ids(splitter.split_documents(documents))
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    print(f"Building BM25 index for user '{user}'...")
    pkl_path = build_bm25(chunks, user)
    print(f"BM25 index saved to {pkl_path}")

    print(f"Loading embedding model '{EMBED_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Embedding chunks into ChromaDB collection 'user_{user}'...")
    build_chroma(chunks, user, embeddings)
    print(f"ChromaDB collection 'user_{user}' ready.")

    print("\n--- Sample chunks (3 random) ---")
    for chunk in random.sample(chunks, min(3, len(chunks))):
        preview = chunk.page_content[:200].replace("\n", " ")
        print(f"  [{chunk.metadata.get('source', 'unknown')}] {preview}...")

    print(f"\nDone. {len(chunks)} chunks indexed for user '{user}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest .mmd files into per-user BM25 + ChromaDB indexes.")
    parser.add_argument("--user", required=True, help="Username (determines which collection to write to)")
    parser.add_argument("inputs", nargs="+", help=".mmd file(s) or directory containing .mmd files")
    args = parser.parse_args()

    ingest(user=args.user, inputs=args.inputs)
