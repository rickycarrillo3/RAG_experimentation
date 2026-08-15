"""
retrieval.py - The retrieval pipeline, in one place.

Hybrid BM25 + dense search, fused with Reciprocal Rank Fusion, then rescored by a
cross-encoder reranker:

    query ─┬─→ BM25        top-10 ─┐
           └─→ bge-small   top-10 ─┴─→ RRF ─→ top-20 ─→ cross-encoder ─→ top-5 ─→ LLM

This module is imported by query.py (CLI), app.py (web UI) and eval.py (benchmark).
It exists so those three cannot drift apart: an eval that measures a different pipeline
than the one you ship is worse than no eval at all.
"""

import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# Index locations come from config.py so they can be moved onto a pod's persistent
# volume without a code change. Re-exported here because app.py and eval.py import
# them from this module — there must stay exactly one definition (see ingest.py).
from config import BM25_DIR, CHROMA_DIR, resolve_device  # noqa: F401  (re-export)

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
OLLAMA_MODEL = "t1c/deepseek-math-7b-rl:Q4"

TOP_K = 10         # candidates from each retriever
RERANK_TOP_C = 20  # candidate pool taken from RRF and fed to the reranker
TOP_N = 5          # final chunks passed to the LLM
RRF_K = 60         # RRF smoothing constant


# ── Chunk identity ────────────────────────────────────────────────────────────

def chunk_id(doc: Document) -> str:
    """Stable id for a chunk, used to match retrieved chunks against gold labels.

    ingest.py stamps `chunk_id` into metadata. Indexes built before that existed
    fall back to a content prefix, which is what RRF already keys on.
    """
    cid = doc.metadata.get("chunk_id")
    if cid:
        return cid
    return f"legacy::{doc.page_content[:120]}"


# ── Loading ───────────────────────────────────────────────────────────────────

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


class NormalizingEmbeddings(Embeddings):
    """Wrap an embedder so text is LaTeX-normalized before it is encoded.

    Only the *vectors* see normalized text; page_content stays raw everywhere else.
    Because ingest and query both go through load_embeddings, the query is normalized
    with exactly the same function the documents were — no divergent second path.
    """

    def __init__(self, base: Embeddings, normalize):
        self.base = base
        self.normalize = normalize

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base.embed_documents([self.normalize(t) for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.base.embed_query(self.normalize(text))


def load_embeddings(model_name: str = EMBED_MODEL, normalize_latex: bool = False) -> Embeddings:
    # device is pinned only when CUDA is present; otherwise the key is omitted entirely
    # so sentence-transformers keeps auto-detecting (MPS on the Mac, CPU elsewhere).
    device = resolve_device()
    model_kwargs = {"device": device} if device else {}
    base = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )
    if normalize_latex:
        from latex_norm import normalize_latex as _normalize
        return NormalizingEmbeddings(base, _normalize)
    return base


def load_reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL, device=resolve_device())


# ── Stages ────────────────────────────────────────────────────────────────────

def bm25_search(query: str, bm25: BM25Okapi, chunks: list[Document], k: int) -> list[Document]:
    scores = bm25.get_scores(query.split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in top_indices]


def dense_search(query: str, store: Chroma, k: int) -> list[Document]:
    return store.similarity_search(query, k=k)


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = RRF_K,
    weights: list[float] | None = None,
) -> list[tuple[Document, float]]:
    """Fuse ranked lists by rank position only: score = sum of w * 1/(k + rank).

    Note this deliberately ignores each retriever's own score — RRF is ordinal.
    Discarding that magnitude is precisely the weakness the reranker exists to fix.

    `weights` (one per ranked list, default all 1.0) lets a weaker retriever be
    down-weighted rather than dropped outright — worth testing when one retriever is
    measurably dragging the fusion below its better input.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(f"Got {len(weights)} weights for {len(ranked_lists)} ranked lists.")

    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for weight, ranked_list in zip(weights, ranked_lists):
        for rank, doc in enumerate(ranked_list, start=1):
            key = doc.page_content[:120]
            scores[key] += weight * (1 / (k + rank))
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [(doc_map[key], scores[key]) for key in sorted_keys]


def rerank(
    query: str,
    candidates: list[tuple[Document, float]],
    reranker: CrossEncoder,
    top_n: int | None = TOP_N,
) -> list[tuple[Document, float]]:
    """Rescore candidates with the cross-encoder, sorted best-first.

    Unlike the bi-encoder, this reads (query, chunk) *together* in one forward pass,
    so it can judge specific relevance rather than topical similarity. Cost is linear
    in len(candidates), which is why it only ever sees the RRF shortlist.

    top_n=None returns every candidate rescored (used by eval to measure rank depth).
    """
    if not candidates:
        return []
    pairs = [(query, doc.page_content) for doc, _ in candidates]
    scores = reranker.predict(pairs)
    reranked = sorted(
        zip((doc for doc, _ in candidates), scores),
        key=lambda x: x[1],
        reverse=True,
    )
    out = [(doc, float(score)) for doc, score in reranked]
    return out if top_n is None else out[:top_n]


# ── Orchestration ─────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """One query's trip through the pipeline, with the intermediate stages kept.

    eval.py needs more than the final top-5: whether the gold chunk was even in the
    pre-rerank pool is what separates "the reranker demoted it" (fixable by a better
    reranker) from "retrieval never found it" (fixable only upstream).
    """
    final: list[tuple[Document, float]]       # top_n — what the LLM actually sees
    ranked: list[tuple[Document, float]]      # full ranked list, for rank-depth metrics
    candidates: list[tuple[Document, float]]  # pre-rerank RRF pool
    timings: dict[str, float] = field(default_factory=dict)  # seconds, per stage


def retrieve_detailed(
    query: str,
    user: str,
    embeddings,
    use_bm25: bool = True,
    use_dense: bool = True,
    reranker: CrossEncoder | None = None,
    use_rerank: bool = True,
    top_n: int = TOP_N,
    top_k: int = TOP_K,
    rerank_top_c: int = RERANK_TOP_C,
    rrf_weights: list[float] | None = None,
    bm25_index: tuple[BM25Okapi, list[Document]] | None = None,
    store: Chroma | None = None,
) -> RetrievalResult:
    """Run the pipeline and keep every stage, with per-stage wall-clock timings.

    top_k is exposed because it caps everything downstream: RRF can only ever emit
    up to top_k * 2 candidates, so raising rerank_top_c above that does nothing. To
    actually widen the reranker's pool you must raise top_k as well.

    bm25_index / store may be passed in pre-loaded. The CLI reloads them per query
    (fine, interactive); eval runs hundreds of queries and must not pay that each time.
    """
    ranked_lists: list[list[Document]] = []
    timings: dict[str, float] = {}

    if use_bm25:
        t0 = time.perf_counter()
        bm25, chunks = bm25_index if bm25_index is not None else load_bm25(user)
        ranked_lists.append(bm25_search(query, bm25, chunks, k=top_k))
        timings["bm25"] = time.perf_counter() - t0

    if use_dense:
        t0 = time.perf_counter()
        s = store if store is not None else load_chroma(user, embeddings)
        ranked_lists.append(dense_search(query, s, k=top_k))
        timings["dense"] = time.perf_counter() - t0

    if not ranked_lists:
        raise ValueError("At least one of BM25 / dense retrieval must be enabled.")

    t0 = time.perf_counter()
    merged = reciprocal_rank_fusion(ranked_lists, weights=rrf_weights)
    timings["rrf"] = time.perf_counter() - t0

    candidates = merged[:rerank_top_c]

    if use_rerank and reranker is not None:
        t0 = time.perf_counter()
        ranked = rerank(query, candidates, reranker, top_n=None)
        timings["rerank"] = time.perf_counter() - t0
    else:
        ranked = merged

    return RetrievalResult(
        final=ranked[:top_n],
        ranked=ranked,
        candidates=candidates,
        timings=timings,
    )


def retrieve(
    query: str,
    user: str,
    embeddings,
    use_bm25: bool = True,
    use_dense: bool = True,
    reranker: CrossEncoder | None = None,
    use_rerank: bool = True,
) -> list[tuple[Document, float]]:
    """Top-N chunks for a query. The interface query.py and app.py use."""
    return retrieve_detailed(
        query,
        user,
        embeddings,
        use_bm25=use_bm25,
        use_dense=use_dense,
        reranker=reranker,
        use_rerank=use_rerank,
    ).final


def format_context(results: list[tuple[Document, float]]) -> str:
    parts = []
    for doc, _ in results:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)
