"""
api/settings.py - Environment-driven configuration for the API service.

Everything the pod needs to differ from a laptop lives here, so deployment is a
matter of environment variables rather than edited source.
"""

import os

from config import (  # noqa: F401  (re-exported for callers)
    DATA_DIR,
    KEEP_ALIVE,
    NUM_PREDICT,
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

# ── Retrieval ─────────────────────────────────────────────────────────────────
# Cross-encoder relevance floor. If the best retrieved chunk scores below this, the
# corpus probably does not cover the question and we answer in `general` mode rather
# than pretending an irrelevant chunk is a source.
#
# SCALE: sentence_transformers.CrossEncoder applies the model's activation function,
# and bge-reranker-v2-m3 carries a Sigmoid — so predict() returns (0, 1), NOT raw
# logits. Do not set this to a logit-like value; 0.0 would make every score pass and
# abstention impossible. Measured on this corpus:
#     unrelated text  ~1e-5 .. 1.5e-3
#     weakly on-topic ~0.19
#     the right chunk ~0.94
# 0.01 sits in the gap, but it is a STARTING GUESS from a handful of pairs, not a
# calibrated value. The `no_answer` slice of gold set v3 exists to settle it — sweep
# this threshold against that slice and set it from data before quoting any
# abstention number.
RELEVANCE_FLOOR = float(os.environ.get("KBM_RELEVANCE_FLOOR", "0.01"))

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
