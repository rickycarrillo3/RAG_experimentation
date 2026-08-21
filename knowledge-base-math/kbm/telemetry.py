"""
kbm/telemetry.py - Append-only usage log.

One JSON object per query. Two payoffs beyond dashboards, both of which need the data
to already exist by the time you want them:

1. **Real questions replace synthetic ones.** EVALUATION.md §8 names "LLM-generated
   questions are not how your family talks" as a limitation the protocol cannot fix
   from the inside. Logged questions are the fix — gold set v4 should be drawn from
   them rather than from qwen2 writing questions about chunks.
2. **Fine-tuning data.** Thumbs-up (question, retrieved chunk) pairs are exactly the
   contrastive pairs an embedding fine-tune needs — far cheaper than touching the 7B
   generator. See EVALUATION.md §10.9.

Usernames are hashed. The questions are a teenager's homework and the log is a file on
a rented box; there is no reason for it to carry names.
"""

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone

from kbm.config import TELEMETRY_PATH, TELEMETRY_SALT

# Writes are appends of a single line under a lock. Concurrency here is a handful of
# family members, so a lock plus line-buffered appends is sufficient and keeps the log
# readable by `tail`, `jq`, and pandas without a database.
_lock = threading.Lock()


def user_hash(user: str) -> str:
    return hashlib.sha256(f"{TELEMETRY_SALT}:{user}".encode()).hexdigest()[:16]


def new_event_id() -> str:
    return uuid.uuid4().hex


def log_query(
    event_id: str,
    user: str,
    question: str,
    mode: str,
    sources: list[dict],
    timings: dict,
    model: str,
    n_completion_chars: int,
    error: str | None = None,
    *,
    truncated: bool = False,
    continuations: int = 0,
    tool_calls: int = 0,
    tool_errors: int = 0,
    protocol: str = "none",
    tool_counts: dict | None = None,
    search_queries: list | None = None,
    late_sources: list | None = None,
) -> None:
    _append({
        "kind": "query",
        "event_id": event_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "user_hash": user_hash(user),
        "question": question,
        "mode": mode,
        "sources": sources,
        "timings": timings,
        "model": model,
        "n_completion_chars": n_completion_chars,
        # Whether the answer hit KBM_NUM_PREDICT, and how many resumes it took to finish.
        # Logged because nothing else in this project can answer "is 350 the wrong cap?" —
        # LATENCY.md's advice is "raise it if answers are visibly truncated", which needs
        # someone to notice. These two fields turn that into a query over the log.
        "truncated": truncated,
        "continuations": continuations,
        # Tool-integrated reasoning: how many Python programs the model ran, and how many
        # of those failed. The pair is the only place the running system records whether
        # the sandbox is earning its keep — a corpus of real questions where tool_calls is
        # always 0 says the model does not reach for it, and a high tool_errors rate says
        # it reaches for things the sandbox policy refuses. Neither is visible in an
        # offline benchmark, which is the same argument that put this module here.
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        # Which tool arm actually ran: "tools", "tir" or "none". Logged rather than
        # inferred from `model`, because it is the product of a profile, two environment
        # variables and a startup capability probe — and "which arm produced this answer?"
        # is the first question to ask of any row during the generator bake-off.
        "protocol": protocol,
        # Per-tool call counts, e.g. {"search_documents": 2, "run_python": 1}. One field
        # rather than a counter per tool, so adding a tool does not change this schema.
        "tool_counts": tool_counts or {},
        # The search queries THE MODEL WROTE, when it decided the question needed another
        # look at the documents. The most valuable field added here.
        #
        # EVALUATION.md §8 names "LLM-generated questions are not how your family talks"
        # as the limitation the protocol cannot fix from the inside, and §10.9 makes the
        # logged questions the fix. This is a second and different signal: the gap between
        # what the student typed and the words that actually found the right chunk is
        # exactly the training pair a query-rewriting step or an embedding fine-tune
        # needs, and — like everything else in this file — it cannot be backfilled.
        #
        # It is the model's phrasing, not the student's, so it carries no PII that the
        # `question` field above does not already carry.
        "search_queries": search_queries or [],
        # Chunks a mid-answer search retrieved, UNFILTERED and with scores, for the same
        # reason the `sources` field above is unfiltered: the floor can be re-applied
        # offline, but a chunk that was dropped and never logged is gone. Without this,
        # re-calibrating KBM_RELEVANCE_FLOOR from real traffic would silently cover only
        # the retrievals the server initiated.
        "late_sources": late_sources or [],
        "error": error,
    })


def log_feedback(event_id: str, rating: str, note: str = "") -> None:
    """Feedback is a separate record keyed by event_id, not a mutation of the query row.

    An append-only log stays append-only: rewriting an earlier line to attach a rating
    would mean rewriting the file, and would lose *when* the rating arrived.
    """
    _append({
        "kind": "feedback",
        "event_id": event_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "rating": rating,
        "note": note,
    })


def _append(record: dict) -> None:
    # Telemetry must never take down a query. A failed write is logged to stderr and
    # dropped — losing an analytics row is strictly better than losing a student's answer.
    try:
        os.makedirs(os.path.dirname(TELEMETRY_PATH) or ".", exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with _lock, open(TELEMETRY_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:  # noqa: BLE001 - deliberately swallowed, see above
        print(f"[telemetry] dropped {record.get('kind')} event: {e}")
