"""
kbm/images.py - Per-document figure sidecars: captions for images embedded in a
textbook, and lookup of them at query time.

Implements the "document-embedded figures" half of Information_files/IMAGE_DESIGN.md.
(The other half — chat-uploaded photos — is not stored at all; see api/chat.py's
is_image_turns/most_recent_image/resolve_image_reference, which work on the client-held
`history` directly, with no persistence.)

Storage is one JSONL file per DOCUMENT (not per user, unlike kbm/memory.py), deliberately
the same idiom: a lock, whole-file writes, exceptions swallowed, so a failed caption or a
missing sidecar can never take down ingest or a query. One record per figure:

    {"image_id": "<basename>::img<n>", "doc_name": "<basename>", "page": <int|None>,
     "caption": "<text>", "nearby_chunk_id": "<chunk_id|None>"}

`image_id`/`doc_name` use the exact same `source` string kbm/chunking.assign_chunk_ids
uses for chunk ids — the basename WITH its extension (e.g. "textbook.mmd"), so the two id
spaces share the `<basename>::` prefix but can never collide (different counters, "img"
vs no infix). The sidecar's own filename drops the extension for readability.

RECALL PATH — recall_figures() mirrors kbm.memory.recall()'s exact shape: load every
sidecar for the user's indexed documents, score captions against the question with the
cross-encoder (reusing kbm.retrieval.rerank), keep what clears the floor. No new
embedding model, no vector store — Information_files/IMAGE_DESIGN.md's "why not a vision
vector store yet" argues this directly: a reranker pass over a short list of captions,
same mechanism kbm/memory.py already uses for curated facts.

Imports nothing from api/ (same rule as kbm/memory.py) and keeps the reranker import
lazy, so importing this module does not drag in torch.
"""

from __future__ import annotations

import json
import logging
import os
import threading

from kbm.config import IMAGES_DIR

log = logging.getLogger(__name__)

# Same rationale as kbm/memory._lock: concurrency is a handful of family members plus
# one ingest worker, so a lock plus whole-file writes is enough and needs no database.
_lock = threading.Lock()

# Non-pinned-equivalent cap: figures kept after relevance ranking. There is no "pinned"
# concept here (every figure is equally eligible), unlike memory's curated facts.
MAX_FIGURES = 3

# Fallback relevance floor, mirroring kbm.memory._DEFAULT_FLOOR for the same reason: the
# live value is KBM_RELEVANCE_FLOOR (api/settings.py), and kbm/ must not import api/.
_DEFAULT_FLOOR = 0.15


def _stem(doc_source: str) -> str:
    """Basename without its extension, for the sidecar's on-disk filename only.

    `doc_source` is the chunk-id-compatible string (basename WITH extension, e.g.
    "textbook.mmd") — only the filename drops the extension; image_id/doc_name do not.
    """
    return os.path.splitext(os.path.basename(doc_source))[0]


def _path(doc_source: str, images_dir: str = IMAGES_DIR) -> str:
    return os.path.join(images_dir, f"{_stem(doc_source)}.images.jsonl")


def all_images(doc_source: str, images_dir: str = IMAGES_DIR) -> list[dict]:
    """Every figure record for one document. [] if no sidecar exists."""
    path = _path(doc_source, images_dir)
    if not os.path.exists(path):
        return []
    out: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    log.warning("skipping malformed image record for %s", doc_source)
                    continue
                if isinstance(rec, dict) and rec.get("image_id"):
                    out.append(rec)
    except OSError as e:
        log.warning("could not read image sidecar for %s: %s", doc_source, e)
        return []
    return out


def add_images(doc_source: str, records: list[dict], images_dir: str = IMAGES_DIR) -> None:
    """Replace a document's sidecar wholesale with `records`.

    Full-file replace, not append — a document is re-captioned on every (re-)ingest, same
    reasoning as build_chroma's upsert-by-id: a partial figure list from a half-finished
    caption pass is worse than none, and a stale figure from a document that shrank must
    not survive a re-ingest. Never raises (kbm/memory.py's contract, same reason: a
    sidecar write failing must not fail the ingest it is part of).
    """
    try:
        os.makedirs(images_dir, exist_ok=True)
        lines = [json.dumps(r, ensure_ascii=False) for r in records]
        with _lock, open(_path(doc_source, images_dir), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
    except Exception as e:  # noqa: BLE001 - deliberately swallowed, see docstring
        log.warning("dropped image sidecar write for %s: %s", doc_source, e)


def caption_image(image_bytes: bytes, vision_llm) -> str:
    """One captioning call via a pre-loaded vision ChatOllama client. Never raises.

    Lazy-imports langchain_core, same reason kbm/memory.py lazy-imports kbm.retrieval:
    callers that never caption (ingest.py --no-images, test_chat.py --selftest) should
    not pay for it. Returns "" on any failure — a dropped caption degrades a document to
    fewer findable figures, not a failed ingest or a failed chat turn.
    """
    try:
        import base64

        from langchain_core.messages import HumanMessage

        b64 = base64.b64encode(image_bytes).decode("ascii")
        message = HumanMessage(content=[
            {"type": "text", "text": "Describe this image in one or two sentences, "
                                      "focusing on any mathematical content (equations, "
                                      "diagrams, graphs, labelled figures)."},
            {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
        ])
        response = vision_llm.invoke([message])
        return str(response.content).strip()
    except Exception as e:  # noqa: BLE001 - captioning must never break ingest or chat
        log.warning("captioning failed: %s", e)
        return ""


def figures_near_chunk(
    chunk_id: str, doc_source: str, images_dir: str = IMAGES_DIR
) -> list[dict]:
    """Figures whose nearby_chunk_id == chunk_id — the "see Figure 3" linkage."""
    return [r for r in all_images(doc_source, images_dir) if r.get("nearby_chunk_id") == chunk_id]


def recall_figures(
    user_sources: list[str],
    question: str,
    reranker=None,
    floor: float = _DEFAULT_FLOOR,
    max_figures: int = MAX_FIGURES,
    images_dir: str = IMAGES_DIR,
) -> list[dict]:
    """The figures to inject for `question`, best first. Mirrors kbm.memory.recall().

    Loads every sidecar for `user_sources` (the user's currently-indexed document
    basenames), scores captions against the question with the cross-encoder (reusing
    kbm.retrieval.rerank — no vector store), keeps those clearing `floor`, caps at
    `max_figures`. [] when there is no reranker, no documents have figures, or nothing
    clears the floor. Never raises.
    """
    try:
        all_records: list[dict] = []
        for source in user_sources:
            all_records.extend(all_images(source, images_dir))
        if not all_records or reranker is None:
            return []

        from langchain_core.documents import Document

        from kbm.retrieval import rerank

        scored = rerank(
            question,
            [(Document(page_content=r.get("caption", "")), 0.0) for r in all_records],
            reranker,
            top_n=None,
        )
        by_caption = {r.get("caption", ""): r for r in all_records}
        kept: list[dict] = []
        for doc, score in scored:
            if score < floor:
                continue
            rec = by_caption.get(doc.page_content)
            if rec is not None:
                kept.append(rec)
            if len(kept) >= max_figures:
                break
        return kept
    except Exception as e:  # noqa: BLE001 - figure recall must never break a request
        log.warning("figure recall failed: %s", e)
        return []
