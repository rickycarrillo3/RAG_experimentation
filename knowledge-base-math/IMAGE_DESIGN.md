# Image metadata & indirect reference — decision record

This records a plan for letting the tutor resolve indirect references to images —
"the most recent photo", "the one before that", "the figure from chapter 3" — across the
two places an image can come from: a photo a student sends mid-chat, and a figure embedded
in an ingested textbook. **Nothing here is implemented yet.** Read this before adding
automatic image logging, a vision-embedding store, or a vision-capable generator.

## The problem it solves

There is no image support in the serving path today. The prior vision work lives on
`archive/experiments` and is not reused here — this is a fresh design against the current
architecture. `api/chat.py`'s `history` only carries text turns (`Message.content` is a
string, `format_history` renders `Student:`/`Tutor:` lines), and `kbm/retrieval.py`/Chroma
has no image modality at all: nothing to search, and no slot to put a reference in.

The nearest existing precedent is `chat.is_follow_up` / `chat.retrieval_query`
(`api/chat.py:629-682`) — the heuristic that notices a question is elliptical ("why?",
"explain that again") and borrows the previous student message so retrieval doesn't run on
stopwords alone. That mechanism solves the *text* anaphor case, and the shape of its
solution (walk recent turns, borrow context, don't touch the prompt-prefix text itself) is
what this design reuses — but it does not do anything for images, and nothing today tracks
which turn was an image upload at all.

## What v1 is

Two image sources, kept separate, because their retention story differs.

### Chat-uploaded images

A student photographs a problem and sends it mid-conversation.

- **No bytes retained**, matching the existing rule for uploaded PDFs: the source file and
  anything derived from it live in the request's temp dir and are deleted when the job ends
  (`DEPLOYMENT.md §7`, `ARCHITECTURE.md §5` — "Uploaded PDFs... not retained; held in a temp
  dir for the length of the ingest"). An uploaded image gets the same treatment: caption it
  once, discard the pixels.
- The caption (plus any OCR'd text a vision step naturally produces) becomes a normal
  **turn artifact** — folded into `history` the way a text message is, tagged with its
  `turn_index`/`ts` so later turns can find it. No new persisted store for this case.
- "Most recent" / "the one before that" resolves by walking `history` backward, filtering to
  image turns, and picking by position — the same shape as `retrieval_query`'s "borrow the
  last student message" logic, just filtered to a different turn type and living alongside
  it rather than inside it. A query that names a description ("the triangle picture")
  resolves by scoring the small set of in-conversation captions, not a new full reranker
  pass — there are at most a few images per conversation, so no infra is needed here either.

### Document-embedded figures

A figure inside an ingested textbook (from `extract.py`/`ingest.py`).

- These persist as part of the retained corpus, so a real sidecar index is fine — there's no
  retention tension the way there is for a chat upload.
- One JSONL sidecar per document, `<basename>.images.jsonl`, next to the extracted `.mmd`.
  One record per figure: `{image_id, doc_name, page, caption, nearby_chunk_id}`.
  `image_id` is `<basename>::img<n>`, mirroring the chunk-id convention
  `assign_chunk_ids` already uses (`kbm/chunking.py:36-47`, `f"{source}::{n}"` — chunks and
  figures share the `<basename>::` prefix but different counters, so the two id spaces never
  collide). `nearby_chunk_id` links a figure to the chunk that surrounds it, so a chunk
  saying "see Figure 3" can pull the figure's caption into context alongside its own text.
- Built at ingest time as a new step inside `ingest.py`, not a separate script — same reason
  the two indexes (`BM25`, Chroma) are built together there rather than in two tools that can
  drift out of sync.
- Content lookup ("the figure from chapter 3") reuses `kbm.retrieval.rerank`
  (`kbm/retrieval.py:169`) against the stored captions — the exact mechanism
  `kbm/memory.py`'s `recall()` already uses to score curated facts against a question, no
  new embedding model or vector store.

## Why not persist chat-upload images

Keeping uploaded image bytes around (even just for the life of a conversation, on disk)
would be a new exception to an existing invariant — "uploaded documents are not retained"
(`DEPLOYMENT.md §7`, `ARCHITECTURE.md §5`) — for a need that within-conversation,
caption-in-`history` recency already covers. If a real need for the original pixels later
(re-captioning, showing the image back to the student) ever shows up, that's the point to
revisit this, not before.

## Why not a vision vector store yet

A second embedding/vector index just for figure captions isn't justified by anything
measured. Per-document figure counts are small, captions are short strings, and
`kbm.retrieval.rerank` already does bounded-candidate cross-encoder reranking elsewhere in
the repo (`kbm/memory.py`'s `recall()`) with no dedicated store behind it. This mirrors
`MEMORY_DESIGN.md`'s own "why not a database" call — JSONL plus a rerank pass over a small
list, promote to a real index only if the fact/caption count ever outgrows that. Not needed
yet.

## Out of scope for v1

- **Cross-session image recall** ("the photo I sent last week"). That needs a persisted
  per-user image log that survives across conversations, which runs straight into
  `kbm/memory.py`'s rule that writes to a student's durable profile are human-curated only —
  see `MEMORY_DESIGN.md`'s "Why the model does not decide what to remember". Auto-logging
  every uploaded image would be exactly the automatic-capture path that document argues
  against. If this is wanted later, it's a deliberate, named exception to that rule — not a
  default extension of it.
- **Any vision-capable generator wiring.** No model is chosen here. The current generators
  (deepseek-math, qwen3:8b) are text-only; captioning needs an OSS vision model served via
  Ollama and a corresponding `kbm/llm_profiles.py` entry. This doc only names the gap.
- **OCR beyond whatever the captioning step itself produces.** No separate OCR pipeline.
- **A frontend for browsing image metadata.** `app.py` is untouched by this design.
