"""
api/settings.py - Environment-driven configuration for the API service.

Everything the pod needs to differ from a laptop lives here, so deployment is a
matter of environment variables rather than edited source.
"""

import os

# ── Auth ──────────────────────────────────────────────────────────────────────
# A single shared secret, checked as `Authorization: Bearer <token>`. This is NOT
# per-user auth — the per-username index isolation behind it is unchanged, and is
# still just a string, not an identity. What this buys is the difference between
# "private URL" and "readable by anything that finds the host".
API_TOKEN = os.environ.get("KBM_API_TOKEN", "").strip()

# ── Storage ───────────────────────────────────────────────────────────────────
# On the pod these point at the network volume so the corpus and indexes never
# live on a personal disk. See retrieval.py, which reads the same variables — the
# API must not introduce a second opinion about where the data is.
DATA_DIR = os.environ.get("KBM_DATA_DIR", ".")

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
NUM_PREDICT = int(os.environ.get("KBM_NUM_PREDICT", "350"))
KEEP_ALIVE = os.environ.get("KBM_KEEP_ALIVE", "30m")

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
