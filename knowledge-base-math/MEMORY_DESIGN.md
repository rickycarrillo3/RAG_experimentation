# Cross-conversation memory — decision record

`kbm/memory.py` gives the tutor a small, durable profile of each student that survives
between conversations: year group, goals, the mistakes they keep making, how they like
things explained. This is the record of what was decided and why. Read it before adding an
automatic-capture path or moving the store to a database.

## The problem it solves

Conversation history is client-supplied per `POST /chat` and forgotten when the stream
ends (`api/schemas.py`, `api/chat.format_history`). There is no session id and no
server-side state; `user` is the only identity, and it just selects a document index.
Reloading the UI starts from nothing. So the tutor cannot know on Tuesday what it learned
about a student on Monday.

## What v1 is

- **Storage: one append-only JSONL file per user**, `$MEMORY_DIR/user_<name>.jsonl`
  (`MEMORY_DIR` defaults to `$DATA_DIR/memory`). Same idiom as `kbm/telemetry.py` — a
  `threading.Lock`, line appends, every exception swallowed and logged. One record per
  line: `{id, ts, user, fact, pinned, source}`. `user` is cleartext (the filename carries
  it already, and this is per-user local state, not a shareable analytics log — the
  opposite of telemetry's `user_hash`).
- **Write path: human-curated only.** `python -m kbm.memory add|list|forget|pin <user>`.
  A parent or the student maintains the list.
- **Recall: pinned + relevant.** `recall()` returns every pinned fact plus the non-pinned
  facts the cross-encoder reranker scores at or above `KBM_RELEVANCE_FLOOR` for the
  current question (reusing `kbm.retrieval.rerank`), then trims the combined list to a
  token budget (pinned dropped last). `api/chat.format_memory` renders the result into the
  `{memory}` slot of the mode-rules block, immediately before `Conversation so far:`.
- **Budget gate: `mem_tokens` per model profile** (`kbm/llm_profiles.py`), overridable
  with `KBM_MEMORY_TOKENS`. **0 for every 4096-token model** (deepseek-math, Qwen2.5-Math)
  — their prompt already reaches ~3.4k with retrieved context and history (`LATENCY.md`),
  and Ollama answers an overflow by dropping the system prompt from the left. **400 for
  qwen3** (8192). So recall is effectively a qwen3/agent-mode feature; the CLI write path
  works under any generator.
- **Backed up.** `ops/backup_indexes.py` archives `memory/` alongside the indexes. It is
  the only copy — uploaded PDFs are not retained and `/workspace` is one delete away from
  gone.
- **Observable.** `DoneEvent.memories_recalled` and the telemetry `query` record carry how
  many facts were injected.

## Why the model does not decide what to remember

The obvious design — a `remember` tool the generator calls mid-answer — asks the model to
make a write-time editorial judgment ("is this fact durable?"). That was rejected:

- deepseek-math (the default) cannot call tools at all — `ollama show` reports
  `Capabilities: completion`. It is a solver, not an instruction-follower.
- qwen3 can call tools, but small instruct models are measured **over-eager** at
  discretionary calls (`kbm/tools/agent.py` docstring, the `list_documents` regression).
  "Should I persist this?" is fuzzier than "do I need to compute?" and has no feedback
  signal to correct it.
- The judgment the pipeline *is* good at is "is this relevant to the question", and that
  one is made at **recall** time by the reranker, not at write time.

So v1 keeps the human in the write loop. `kbm.memory.add()` is the single entry point any
future automatic capture must call — the record shape and the recall path do not need to
change to gain one.

## Why not a database (or a Chroma collection)

Family scale is a handful of users with tens of facts each; recall injects at most a
handful. "All pinned, then rerank the rest, then truncate" needs no vector index. JSONL is
greppable with `jq`/pandas, matches the project's stated "no relational DB / Redis"
convention, and is already the right shape to mine for fine-tuning data later.

**v2, if a user's fact count ever outgrows the injection cap:** promote the non-pinned
facts to a per-user `mem_<name>` Chroma + BM25 pair and route them through
`kbm.retrieval.retrieve()`. `load_chroma`/`load_bm25` already accept preloaded handles and
`ingest.py`'s upsert-by-stable-id pattern is reusable. Not needed yet; JSONL stays the
single source of truth until then.

## KV-cache placement

Prompt order is `static rules → memory → history → (human: context → question)`. Memory is
byte-stable within a conversation (curated writes do not happen mid-chat) and only varies
by user, so `static + memory + history` stays cached turn to turn. It goes in as a
`ChatPromptTemplate` **variable**, not literal template text, because a fact can contain
LaTeX braces (`\frac{d}{dx}`) that would raise at format time as literals — the
`kbm/tools/agent.py` "NO BRACES" lesson. When `MEMORY_TOKENS == 0` the variable is `""`
and the rendered prompt is byte-identical to the pre-feature prompt.

## Out of scope for v1

- Any model-driven write — autonomous `remember` tool, mechanical turn capture, post-turn
  LLM extraction. All deferred behind `memory.add()`.
- Session/conversation summaries ("last time we did the chain rule"). v1 is a flat fact
  list with no conversation boundaries.
- The `mem_<name>` Chroma collection (v2, above).
- A frontend for viewing/editing memories — `app.py` is untouched; inspection is
  `python -m kbm.memory list <user>`.
- Parent/child access controls beyond that CLI command.
- Recall on a 4096-token model by default (opt in with `KBM_MEMORY_TOKENS`).
- `kbm/tools/tir.py` and `kbm/tools/agent.py` — unchanged.
