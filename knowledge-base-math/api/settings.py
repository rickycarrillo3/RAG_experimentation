"""
api/settings.py - Environment-driven configuration for the API service.

Everything the pod needs to differ from a laptop lives here, so deployment is a
matter of environment variables rather than edited source.
"""

import os

from config import (  # noqa: F401  (re-exported for callers)
    DATA_DIR,
    KEEP_ALIVE,
    NUM_CTX,
    NUM_PREDICT,
    THINK,
    TIR_ENABLED,
    TOOLS_ENABLED,
)

# NOTE: deployment knobs shared with the CLI and the Gradio client — DATA_DIR,
# CHROMA_DIR, BM25_DIR, OLLAMA_BASE_URL, APP_HOST/PORT, APP_AUTH, REQUIRE_GPU — live in
# config.py, NOT here. This module holds only what is specific to the HTTP service.
# An earlier version of this file read its own `KBM_DATA_DIR`, which meant setting the
# documented `DATA_DIR` moved the indexes for ingest and query but not for the API.

# ── Auth ──────────────────────────────────────────────────────────────────────
# A single shared secret, checked as `Authorization: Bearer <token>`. This is NOT
# per-user auth — the per-username index isolation behind it is unchanged, and is
# still just a string, not an identity. What this buys is the difference between
# "private URL" and "readable by anything that finds the host".
API_TOKEN = os.environ.get("KBM_API_TOKEN", "").strip()

# Whether to serve the interactive OpenAPI docs (/docs, /redoc, /openapi.json).
#
# Those routes are mounted by FastAPI on the *app*, while require_token is a dependency of
# the *router* — so they are not behind the token and never were. Anyone who reaches the
# host reads the complete shape of the service: every path, every field of every schema.
#
# Default: on when there is no token (a laptop, where /docs is part of the documented
# workflow), off as soon as one is set (a deployed pod). Set KBM_ENABLE_DOCS=1 to get them
# back on a pod — over an SSH tunnel, not a public port.
ENABLE_DOCS = (
    os.environ.get("KBM_ENABLE_DOCS", "").strip().lower() in ("1", "true", "yes")
    or not API_TOKEN
)

# ── Retrieval ─────────────────────────────────────────────────────────────────
# Cross-encoder relevance floor. A retrieved chunk is used as context only if it scores
# at least this much, and the answer is `grounded` only if at least one chunk does
# (api/chat.py:select_context). Below it we answer in `general` mode rather than
# pretending an irrelevant chunk is a source.
#
# SCALE: sentence_transformers.CrossEncoder applies the model's activation function,
# and bge-reranker-v2-m3 carries a Sigmoid — so predict() returns (0, 1), NOT raw
# logits. Do not set this to a logit-like value; 0.0 would make every score pass and
# abstention impossible.
#
# CALIBRATED (evaluation/calibrate_floor.py) — the previous 0.01 was a guess from a
# handful of pairs, and measurement showed it never abstained at all. Two datasets,
# because they bound the answer from opposite sides:
#
#   Our own corpus, real pipeline, questions the book does not cover (n=19 on-topic
#   from the gold set, 18 off-topic):
#       off-topic  max 0.0395   median 0.0037
#       on-topic   min 0.5965   median 0.9883
#   An empty gap of more than an order of magnitude. Anything inside it gives 100%
#   correct abstention at 0% false abstention; 0.01 sat BELOW the off-topic max and
#   caught only 67% of them. 0.15 is the geometric midpoint of that gap: ~3.8x above
#   the worst off-topic question, ~4x below the weakest real one. The margin is
#   deliberately larger on the on-topic side, because that side is the biased one —
#   gold-set questions were generated from their own chunks and so score high by
#   construction (EVALUATION.md §3), while a student's own phrasing will score lower.
#
#   ARQMath-1 (34,813 human-graded pairs, MIRB `hcju/mseqa`): pairwise AUC of just
#   0.60 for relevant-vs-irrelevant, and the median HUMAN-JUDGED-IRRELEVANT pair still
#   scores 0.63. That is not a reason to raise the floor — those are pooled hard
#   negatives, documents about the right topic that merely fail to answer the question.
#   It is a statement of what this floor CANNOT do: it separates on-topic from
#   off-topic well, and "on-topic but unhelpful" barely at all. Abstention protects
#   against the wrong book, not against the wrong paragraph of the right book.
#
# Re-calibrate against real logged questions once telemetry has them (EVALUATION.md
# §10.9) — 19 questions from one chapter is a bracket, not a distribution.
RELEVANCE_FLOOR = float(os.environ.get("KBM_RELEVANCE_FLOOR", "0.15"))

# ── Generation ────────────────────────────────────────────────────────────────
# NUM_PREDICT / KEEP_ALIVE are re-exported from config.py, not redeclared here. They
# used to exist in three places (here, query.py, test_chat.py), so the cap could be
# raised for the API while the CLI silently kept its own hardcoded 350.
#
# How many times the server may resume a generation that stopped because it hit
# NUM_PREDICT. deepseek-math is a chain-of-thought solver that fills whatever budget it
# is given, so a cap alone guarantees a cut-off answer sooner or later; continuing is
# what makes the answer finish. Measured on the Mac: resuming re-prefills only the
# partial answer (~160 ms for 350 tokens, because it appends to the END of the prompt
# and the cached prefix survives — see LATENCY.md) against ~13 s to decode the same
# tokens. So the cap bounds worst-case answer length, and costs almost nothing in
# prefill. 0 disables continuation and falls back to labelling the answer truncated.
MAX_CONTINUATIONS = int(os.environ.get("KBM_MAX_CONTINUATIONS", "2"))

# ── Tool-integrated reasoning ─────────────────────────────────────────────────
# Whether the generator may run Python, and the shape of its window, are properties of
# the model — they come from llm_profiles via config.py (KBM_LLM_MODEL / KBM_TIR /
# KBM_NUM_CTX), not from here. What is service policy, and therefore lives here:
#
# Wall-clock budget for one sandbox execution. Generous enough for a sympy integral on a
# cold import (~0.5s measured, but the pod shares CPU with the reranker) and short enough
# that a runaway loop costs one answer, not the evening. sandbox.py backs it with an
# RLIMIT_CPU that a signal-ignoring program cannot outlast.
SANDBOX_TIMEOUT_S = float(os.environ.get("KBM_SANDBOX_TIMEOUT", "10"))

# Show the student the raw ```python block instead of just the computed result.
#
# Default off, deliberately. The audience is a family member asking about a derivative:
# they want the reasoning and the number, and eight lines of sympy in the middle of the
# explanation is noise they have to learn to skip. The result itself is always shown —
# hiding the arithmetic entirely would make a wrong sandbox answer indistinguishable
# from a hallucination, which is the one failure mode this feature exists to remove.
#
# Set it to 1 to watch what the model is actually running. Same operator escape hatch as
# KBM_SHOW_DIAGNOSTICS, and the same reason: the information is never destroyed, only
# not rendered by default.
SHOW_TOOL_CODE = os.environ.get("KBM_SHOW_TOOL_CODE", "").strip().lower() in ("1", "true", "yes")

# ── Telemetry ─────────────────────────────────────────────────────────────────
TELEMETRY_PATH = os.environ.get(
    "KBM_TELEMETRY_PATH", os.path.join(DATA_DIR, "telemetry", "events.jsonl")
)
# Salt for hashing usernames before they are written to the event log. Set this on
# the pod; the default makes hashes non-portable between machines, which is fine —
# they only ever need to be consistent within one deployment.
TELEMETRY_SALT = os.environ.get("KBM_TELEMETRY_SALT", "kbm-local-dev-salt")

# ── Idle stop ─────────────────────────────────────────────────────────────────
# Minutes without a /chat request before the pod stops itself. This is what keeps
# an on-demand GPU near $17/mo instead of ~$115/mo always-on. Disabled (0) unless
# the RunPod credentials are also present.
IDLE_STOP_MINUTES = int(os.environ.get("KBM_IDLE_STOP_MINUTES", "0"))
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "").strip()
RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID", "").strip()
