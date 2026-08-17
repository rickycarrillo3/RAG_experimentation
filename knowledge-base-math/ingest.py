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
# Index locations come from retrieval.py, not a local copy: if ingest wrote to one
# directory while retrieval read from another, every query would return nothing and
# the indexes would look empty rather than misplaced.
from retrieval import BM25_DIR, CHROMA_DIR, EMBED_MODEL, chunk_id, load_embeddings


def load_mmd_files(paths: list[str]) -> list[Document]:
    documents = []
    for path in paths:
        loader = TextLoader(path, encoding="utf-8")
        documents.extend(loader.load())
    return documents


def merge_chunks(existing: list[Document], new: list[Document]) -> list[Document]:
    """Combine a user's existing chunks with freshly ingested ones, newest winning.

    Deduplicated by `chunk_id`, which is `<source>::<n>` — so re-uploading the *same*
    document replaces its chunks rather than appending a second copy of them. Without
    this, uploading `notes.pdf` twice left the user's index holding every one of its
    chunks twice: BM25 scored the same passage repeatedly and the dense top-k filled
    with copies, crowding out distinct candidates before the reranker ever saw them.

    Note this keys on chunk *id*, not content, so re-uploading an edited version of a
    document correctly replaces the old chunks at the same positions. Chunks past the
    end of a document that got shorter are the one case this cannot catch — they keep
    ids no new chunk claims. Re-ingesting from scratch is the fix if that matters.
    """
    by_id: dict[str, Document] = {chunk_id(d): d for d in existing}
    by_id.update({chunk_id(d): d for d in new})
    return list(by_id.values())


def build_bm25(chunks: list[Document], user: str) -> str:
    tokenized = [doc.page_content.split() for doc in chunks]
    bm25 = BM25Okapi(tokenized)

    os.makedirs(BM25_DIR, exist_ok=True)
    pkl_path = os.path.join(BM25_DIR, f"user_{user}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    return pkl_path


def build_chroma(chunks: list[Document], user: str, embeddings) -> Chroma:
    """Embed chunks into the user's collection, keyed by chunk_id so adds are idempotent.

    The ids are load-bearing, not a nicety. Callers pass the user's *whole* chunk list —
    a second upload merges the existing chunks with the new ones and rebuilds — and
    without explicit ids Chroma mints a fresh UUID per document, so every previously
    indexed chunk is inserted *again*. Measured before this fix: uploads of 66 then 4
    then 4 chunks left Chroma holding 66 → 136 → 210 entries instead of 66 → 70 → 74.

    That is quadratic growth in both storage and embedding time, and it degrades
    retrieval: the dense top-k fills with copies of one chunk, so fewer distinct
    candidates reach RRF and the reranker than TOP_K implies.

    chunk_id is `<source>::<n>` (stamped by chunking.assign_chunk_ids), unique across
    documents and stable across re-ingestion, so re-adding a chunk upserts in place.
    """
    store = Chroma(
        collection_name=f"user_{user}",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    # Defensive dedupe: Chroma rejects a batch containing the same id twice, so a caller
    # that merged carelessly would get a DuplicateIDError mid-ingest rather than a
    # sensible index. merge_chunks already guarantees this; belt and braces because the
    # failure lands in a background job where it is only visible as a failed upload.
    deduped = {chunk_id(doc): doc for doc in chunks}
    store.add_documents(list(deduped.values()), ids=list(deduped))
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
