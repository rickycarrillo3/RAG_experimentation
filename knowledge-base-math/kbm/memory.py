"""
kbm/memory.py - Curated per-user memory: what the tutor knows about a student.

A small, slow-changing profile that carries between conversations — a student's year
group, their goals, the mistakes they keep making, how they like things explained. It is
NOT conversation history (that is still client-supplied per request, api/schemas.py) and
it is NOT a transcript store.

Storage is one append-only JSONL file per user, `$MEMORY_DIR/user_<name>.jsonl`, deliberately
the same idiom as kbm/telemetry.py — a lock, line appends, exceptions swallowed — so it
stays greppable with `jq`/pandas and a failed write can never take down a request. One
record per line:

    {"id": "<uuid4 hex>", "ts": "<iso8601 UTC>", "user": "<name>",
     "fact": "<text>", "pinned": <bool>, "source": "<cli|...>"}

`user` is stored in cleartext — the filename already carries it, and unlike the telemetry
log this file is per-user local state, not a shareable analytics artefact.

WRITE PATH — v1 is human-curated only. Facts are added by a person via `python -m kbm.memory`
(a parent or the student themselves), never by the generator. Deciding *what is worth
remembering* is a judgment neither shipped model is trained for: deepseek-math cannot call
tools at all, and a small instruct model is measured over-eager at discretionary calls
(kbm/tools/agent.py). `add()` is the single documented entry point any future automatic
capture (a post-turn extractor, a `remember` tool) must call, so that recall and the
storage format never have to change to gain one.

RECALL PATH — `recall()` returns the facts to inject for a given question: all pinned
facts, plus the non-pinned ones the cross-encoder reranker (already loaded by the API)
judges relevant, trimmed to a token budget. See api/chat.format_memory for where the
result lands in the prompt, and kbm/config.MEMORY_TOKENS for the budget.

Imports nothing from api/ (same rule as kbm/telemetry.py) and keeps the reranker import
lazy, so `python -m kbm.memory` does not drag in torch.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import uuid
from datetime import datetime, timezone

from kbm.config import MEMORY_DIR

log = logging.getLogger(__name__)

# Same rationale as kbm/telemetry._lock: concurrency is a handful of family members, so a
# lock plus line-buffered appends is enough and needs no database.
_lock = threading.Lock()

# Non-pinned facts kept after relevance ranking, before the token budget is applied.
MAX_RETRIEVED = 5

# chars-per-token used to apply MEMORY_TOKENS without importing a tokenizer onto the hot
# path. Deliberately crude and slightly generous, so the estimate errs toward trimming.
_CHARS_PER_TOKEN = 4

# Fallback relevance floor for recall(). The live value is KBM_RELEVANCE_FLOOR, which
# lives in api/settings.py; kbm/ must not import from api/, so the API passes it in and
# this default only covers the CLI/test callers that never rank anything anyway.
_DEFAULT_FLOOR = 0.15


def _safe_user(user: str) -> str:
    """Lowercased, stripped, and safe to drop into a filename.

    The API hands this an already-normalized user (api/deps.normalize_user); this repeats
    the parts that matter for a path so the CLI is held to the same rule. Not a full copy
    of normalize_user — kbm/ cannot import it — just the filename-safety subset.
    """
    u = (user or "").strip().lower()
    if not u or u in (".", "..") or any(c in u for c in "/\\\0"):
        raise ValueError(f"unusable user name: {user!r}")
    return u


def _path(user: str) -> str:
    return os.path.join(MEMORY_DIR, f"user_{_safe_user(user)}.jsonl")


def _norm(text: str) -> str:
    """Whitespace- and case-insensitive form, for near-duplicate detection."""
    return " ".join(text.split()).casefold()


def all_facts(user: str) -> list[dict]:
    """Every stored fact for a user, oldest first. [] if the file does not exist."""
    path = _path(user)
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
                    log.warning("skipping malformed memory line for user=%s", user)
                    continue
                if isinstance(rec, dict) and rec.get("fact"):
                    out.append(rec)
    except OSError as e:
        log.warning("could not read memory for user=%s: %s", user, e)
        return []
    return out


def add(user: str, fact: str, *, pinned: bool = False, source: str = "cli") -> bool:
    """Append one fact. Returns False if it was blank or a near-duplicate of an existing one.

    THE AUTOMATION HOOK. Any future auto-capture — a post-turn extractor, a `remember`
    tool — calls this and nothing else; the record shape and the recall path do not know
    or care where a fact came from beyond the `source` tag.

    Never raises: a memory write must not break whatever asked for it (kbm/telemetry.py's
    contract, same reason).
    """
    fact = (fact or "").strip()
    if not fact:
        return False
    try:
        if any(_norm(f["fact"]) == _norm(fact) for f in all_facts(user)):
            return False
        record = {
            "id": uuid.uuid4().hex,
            "ts": datetime.now(timezone.utc).isoformat(),
            "user": _safe_user(user),
            "fact": fact,
            "pinned": bool(pinned),
            "source": source,
        }
        os.makedirs(MEMORY_DIR, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with _lock, open(_path(user), "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return True
    except Exception as e:  # noqa: BLE001 - deliberately swallowed, see docstring
        log.warning("dropped memory add for user=%s: %s", user, e)
        return False


def _rewrite(user: str, facts: list[dict]) -> None:
    """Replace the file with `facts`. The only non-append operation; CLI edits only."""
    os.makedirs(MEMORY_DIR, exist_ok=True)
    lines = [json.dumps(f, ensure_ascii=False) for f in facts]
    with _lock, open(_path(user), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def forget(user: str, fact_id: str) -> bool:
    """Drop one fact by id. Returns False if no fact had that id."""
    facts = all_facts(user)
    kept = [f for f in facts if f.get("id") != fact_id]
    if len(kept) == len(facts):
        return False
    _rewrite(user, kept)
    return True


def set_pinned(user: str, fact_id: str, pinned: bool) -> bool:
    """Pin or unpin one fact by id. Returns False if no fact had that id."""
    facts = all_facts(user)
    hit = False
    for f in facts:
        if f.get("id") == fact_id:
            f["pinned"] = bool(pinned)
            hit = True
    if hit:
        _rewrite(user, facts)
    return hit


def recall(user, question, budget_tokens, reranker=None, floor: float = _DEFAULT_FLOOR):
    """The facts to inject for `question`, best first, already trimmed to the budget.

    Pinned facts always come first, in the order they were added. Non-pinned facts are
    scored against the question by the cross-encoder (reusing kbm.retrieval.rerank) and
    only those clearing `floor` are kept, up to MAX_RETRIEVED. The combined list is then
    trimmed from the end until it fits `budget_tokens` (pinned facts are dropped last).

    Returns a list[str] of fact texts — the rendering and the section header live in
    api/chat.format_memory. Empty list when memory is off, the user has none, or nothing
    cleared the floor. Never raises.
    """
    if not budget_tokens or budget_tokens <= 0:
        return []
    try:
        facts = all_facts(user)
        if not facts:
            return []

        pinned = [f["fact"] for f in facts if f.get("pinned")]
        rest = [f["fact"] for f in facts if not f.get("pinned")]

        retrieved: list[str] = []
        if rest and reranker is not None:
            # Lazy: kbm.retrieval pulls in sentence-transformers, which the CLI must not.
            from langchain_core.documents import Document

            from kbm.retrieval import rerank

            scored = rerank(
                question,
                [(Document(page_content=t), 0.0) for t in rest],
                reranker,
                top_n=None,
            )
            retrieved = [
                doc.page_content for doc, score in scored if score >= floor
            ][:MAX_RETRIEVED]

        ordered = pinned + retrieved
        return _fit_budget(ordered, budget_tokens)
    except Exception as e:  # noqa: BLE001 - recall must never break a request
        log.warning("memory recall failed for user=%s: %s", user, e)
        return []


def _fit_budget(facts: list[str], budget_tokens: int) -> list[str]:
    """Drop facts from the end until the rendered block fits the token budget."""
    cap = budget_tokens * _CHARS_PER_TOKEN
    kept = list(facts)
    while kept and len("\n".join(f"- {f}" for f in kept)) > cap:
        kept.pop()
    return kept


# ── CLI ───────────────────────────────────────────────────────────────────────
# print(), not logging: this is terminal output a person is reading, same call as
# ingest.py / query.py / ops.backup_indexes.main().

def _cmd_add(args) -> int:
    ok = add(args.user, args.fact, pinned=args.pin, source="cli")
    if ok:
        print(f"added{' (pinned)' if args.pin else ''}: {args.fact}")
        return 0
    print("not added (blank or already present)")
    return 1


def _cmd_list(args) -> int:
    facts = all_facts(args.user)
    if not facts:
        print(f"(no memory for {args.user})")
        return 0
    for f in facts:
        flag = "P" if f.get("pinned") else " "
        print(f"[{flag}] {f['id']}  {f['fact']}")
    print(f"\n{len(facts)} fact(s), {sum(1 for f in facts if f.get('pinned'))} pinned")
    return 0


def _cmd_forget(args) -> int:
    if forget(args.user, args.id):
        print(f"forgot {args.id}")
        return 0
    print(f"no fact with id {args.id}")
    return 1


def _cmd_pin(args) -> int:
    want = args.command == "pin"
    if set_pinned(args.user, args.id, want):
        print(f"{'pinned' if want else 'unpinned'} {args.id}")
        return 0
    print(f"no fact with id {args.id}")
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m kbm.memory",
        description="Curated per-user memory the tutor carries between conversations.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="add a fact")
    p_add.add_argument("user")
    p_add.add_argument("fact")
    p_add.add_argument("--pin", action="store_true", help="always inject this fact")
    p_add.set_defaults(func=_cmd_add)

    p_list = sub.add_parser("list", help="show all facts for a user")
    p_list.add_argument("user")
    p_list.set_defaults(func=_cmd_list)

    p_forget = sub.add_parser("forget", help="delete a fact by id")
    p_forget.add_argument("user")
    p_forget.add_argument("id")
    p_forget.set_defaults(func=_cmd_forget)

    for name in ("pin", "unpin"):
        p = sub.add_parser(name, help=f"{name} a fact by id")
        p.add_argument("user")
        p.add_argument("id")
        p.set_defaults(func=_cmd_pin)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
