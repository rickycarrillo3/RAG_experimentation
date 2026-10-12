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
import base64
import glob
import json
import os
import pickle
import random
import sys
import time

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from kbm.chunking import (  # assign_chunk_ids re-exported for callers
    CHUNKERS,
    assign_chunk_ids,
)
from kbm.images import add_images, caption_image

# Index locations come from kbm/retrieval.py, not a local copy: if ingest wrote to one
# directory while retrieval read from another, every query would return nothing and
# the indexes would look empty rather than misplaced.
from kbm.retrieval import BM25_DIR, CHROMA_DIR, EMBED_MODEL, chunk_id, load_embeddings


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


def load_vision_llm():
    """Lazily build the captioning ChatOllama client, or None if it cannot be reached.

    Captioning is best-effort: a vision model that is not pulled or an Ollama that is
    not running must degrade ingest to "no figures found", never fail the whole thing.
    Lazy-imported so a plain `python ingest.py` with no figures in any document never
    pays for langchain_ollama.
    """
    try:
        from langchain_ollama import ChatOllama

        from kbm.config import OLLAMA_BASE_URL, VISION_OLLAMA_MODEL, VISION_PROFILE

        return ChatOllama(
            model=VISION_OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            num_predict=VISION_PROFILE.num_predict,
            num_ctx=VISION_PROFILE.num_ctx,
            # Short — captioning is infrequent (once per figure, at ingest time), unlike
            # every-request generation, so it must not sit resident in VRAM alongside
            # the chat generator and reranker.
            keep_alive="2m",
        )
    except Exception as e:  # noqa: BLE001 - captioning is best-effort, see docstring
        print(f"WARNING: could not set up the vision model for captioning: {e}")
        return None


def build_image_sidecar(
    doc_source: str, mmd_path: str, doc_chunks: list[Document], vision_llm
) -> int:
    """Caption every raw image for one document and write its figure sidecar.

    doc_source: the chunk-id-compatible basename (WITH extension, e.g. "textbook.mmd") —
    the same string chunking.assign_chunk_ids uses for `source`.
    mmd_path: path to the .mmd this document was ingested from, used to find the raw
    image bytes extract.py wrote alongside it (<stem>.images.raw.json) — ingest.py never
    re-runs Marker/pymupdf4llm just to get images back.
    doc_chunks: this document's own chunks (already id-stamped), used to resolve
    nearby_chunk_id by nearest same-or-preceding page.

    Never raises: a captioning failure degrades a document to fewer findable figures,
    the same contract as kbm.memory.add(). Returns the number of figures captioned.
    """
    raw_path = os.path.splitext(mmd_path)[0] + ".images.raw.json"
    if not os.path.exists(raw_path):
        return 0
    try:
        with open(raw_path, encoding="utf-8") as f:
            raw_images = json.load(f)
    except (OSError, ValueError) as e:
        print(f"WARNING: could not read {raw_path}: {e}")
        return 0
    if not raw_images:
        return 0
    if vision_llm is None:
        print(f"Skipping {len(raw_images)} figure(s) in {doc_source} — no vision model available.")
        return 0

    # Sorted by page, so "nearest same-or-preceding page" below is a linear scan.
    paged_chunks = sorted(
        ((c.metadata["page"], c) for c in doc_chunks if c.metadata.get("page") is not None),
        key=lambda t: t[0],
    )

    def _nearby_chunk_id(page) -> str | None:
        if page is None or not paged_chunks:
            return None
        best = paged_chunks[0][1]
        for p, c in paged_chunks:
            if p <= page:
                best = c
            else:
                break
        return chunk_id(best)

    records = []
    for n, img in enumerate(raw_images):
        try:
            image_bytes = base64.b64decode(img["bytes_b64"])
        except (KeyError, ValueError) as e:
            print(f"WARNING: skipping unreadable image {n} in {doc_source}: {e}")
            continue
        caption = caption_image(image_bytes, vision_llm)
        if not caption:
            continue
        page = img.get("page")
        records.append({
            "image_id": f"{doc_source}::img{n}",
            "doc_name": doc_source,
            "page": page,
            "caption": caption,
            "nearby_chunk_id": _nearby_chunk_id(page),
        })

    add_images(doc_source, records)
    return len(records)


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
    caption_images: bool = True,
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

    # Figure sidecars, one per document — same point BM25+Chroma are built together, so
    # the three artifacts are built from the same chunk list rather than drifting apart.
    # Captioning happens here (ingest time), not in extract.py: extraction must keep
    # working with no Ollama server running, and this is the step that already needs one.
    n_figures = 0
    if caption_images:
        vision_llm = load_vision_llm()
        for path in paths:
            doc_source = os.path.basename(path)
            doc_chunks = [c for c in chunks if os.path.basename(c.metadata.get("source", "")) == doc_source]
            n_figures += build_image_sidecar(doc_source, path, doc_chunks, vision_llm)
        if n_figures:
            print(f"Captioned {n_figures} figure(s) across {len(paths)} document(s).")

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
        "n_figures": n_figures,
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
    parser.add_argument("--no-images", action="store_true",
                        help="Skip figure captioning (kbm/images.py), e.g. when no vision model is pulled")
    args = parser.parse_args()

    ingest(
        user=args.user,
        inputs=args.inputs,
        chunker=args.chunker,
        embed_model=args.embed_model,
        normalize_latex=args.normalize_latex,
        caption_images=not args.no_images,
    )
