"""
kbm/config.py - Everything about this deployment that changes between machines.

The pipeline itself is machine-independent; what differs between a Mac laptop and a
RunPod GPU pod is *where things live* (indexes, model caches), *what serves the LLM*
(local Ollama vs. a separate host), and *who is allowed to connect*. Those knobs are
collected here rather than scattered as literals, so moving to a new box is an
environment change and not a code edit.

Every default reproduces the original local-Mac behaviour exactly, so importing this
module changes nothing until an env var is actually set.

Env vars
--------
    DATA_DIR          base dir for the indexes            (default: ".")
    CHROMA_DIR        override the Chroma dir outright    (default: $DATA_DIR/chroma_db)
    BM25_DIR          override the BM25 dir outright      (default: $DATA_DIR/bm25_indexes)
    OLLAMA_BASE_URL   where Ollama is served              (default: http://localhost:11434)
    APP_HOST          Gradio bind address                 (default: 0.0.0.0)
    APP_PORT          Gradio port                         (default: 7860)
    APP_AUTH          "user:pass" pairs, comma-separated  (default: unset = no auth)
    REQUIRE_GPU       1 = refuse to run on CPU            (default: unset = allow CPU)
    KBM_LLM_MODEL     the generator's Ollama tag          (default: deepseek-math)
    KBM_NUM_PREDICT   decode cap, in tokens               (default: per llm_profiles)
    KBM_NUM_CTX       context window, in tokens           (default: per llm_profiles)
    KBM_TIR           1/0 = force the Python sandbox on/off (default: per llm_profiles)
    KBM_KEEP_ALIVE    how long Ollama holds the weights   (default: 30m)
"""

import os

from kbm.llm_profiles import profile_for

# ── Where the indexes live ─────────────────────────────────────────────────────
# On a pod these must sit on the persistent volume, or every pod restart silently
# starts from an empty knowledge base. DATA_DIR is the one knob that moves both.
DATA_DIR = os.environ.get("DATA_DIR", ".")
CHROMA_DIR = os.environ.get("CHROMA_DIR") or os.path.join(DATA_DIR, "chroma_db")
BM25_DIR = os.environ.get("BM25_DIR") or os.path.join(DATA_DIR, "bm25_indexes")

# ── Where Ollama is ────────────────────────────────────────────────────────────
# Defaults to the local daemon, which is the pod layout (app and Ollama on the same
# box, sharing the GPU). Set it to point the app at a separate inference host.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Which generator, and what it needs ─────────────────────────────────────────
# This lived as a literal in kbm/retrieval.py, which was fine while there was one model and
# it needed nothing special. It is here now because it is a deployment knob like every
# other one in this file, and because swapping generators is a thing we do on purpose
# (EVALUATION.md §12) rather than a code edit per experiment. kbm/retrieval.py re-exports it,
# so every existing `from retrieval import OLLAMA_MODEL` keeps working.
#
# The default stays deepseek-math until the college-tier bake-off says otherwise. A
# measured swap or none at all.
OLLAMA_MODEL = os.environ.get("KBM_LLM_MODEL", "t1c/deepseek-math-7b-rl:Q4").strip()

# Everything else about the model — window size, decode budget, whether it can drive the
# Python sandbox — comes from its profile, so naming a model configures it. Env vars
# still win: the table supplies each model's right default, not a policy.
PROFILE = profile_for(OLLAMA_MODEL)


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes"):
        return True
    if raw in ("0", "false", "no"):
        return False
    return default


# ── Generation ─────────────────────────────────────────────────────────────────
# These lived in three places at once (api/settings.py, query.py, and test_chat.py via
# query.py), one of which hardcoded 350 — so raising the cap fixed the API and silently
# left the CLI alone. Defined once here, imported everywhere, per CLAUDE.md.
#
# NUM_PREDICT is the direct lever on decode time, which LATENCY.md measures as the
# dominant cost (~38 ms/token on the Mac, ~10x less on the pod). It is a *cap*, not a
# target: hitting it means the answer was cut off, which api/routes.py now detects via
# Ollama's done_reason and continues from. Do not raise this to "fix" truncation — it
# moves the cliff without removing it.
#
# The default is now the model's, not a constant: 350 is what deepseek needs and roughly
# a third of what a TIR trace needs, since that spends tokens on reasoning, then a
# program, then reasoning about the result.
NUM_PREDICT = int(os.environ.get("KBM_NUM_PREDICT") or PROFILE.num_predict)

# Ollama does NOT read the window size from the model — it applies its own default and
# silently shifts the context from the left when a prompt overflows. Left-shifting eats
# the system prompt first, so an over-long TIR trace does not error, it just quietly
# stops the model being a tutor. Setting this explicitly is what makes the overflow
# bounded and diagnosable instead.
NUM_CTX = int(os.environ.get("KBM_NUM_CTX") or PROFILE.num_ctx)

# Which tool protocol the generator gets, and the one place that decides it.
#
# There are two, they are different mechanisms, and a model must never be given both:
#
#   kbm/tools/tir.py    a TEXT protocol — the model writes a ```python block, generation stops at
#             the ```output stop word, and the result is spliced into the same turn.
#             Works on a completion-only model, because it is just text.
#   kbm/tools/agent.py  NATIVE tool calling — a JSON schema goes out with the request and the model
#             answers with a structured tool_calls array in a new turn. Needs a tools
#             template in the model, and reaches retrieval as well as the sandbox.
#
# llm_profiles carries them as independent capabilities because qwen3 genuinely has both.
# Handing a model both at once is what must not happen: it would be given a `tools` array
# AND a ```output stop word, and it will interleave the two mid-answer — unreadable for
# the student, and un-attributable for the eval, which could no longer say which protocol
# produced an answer. So the exclusivity lives here, in one line, downstream of every
# environment variable. KBM_TOOLS=1 KBM_TIR=1 resolves to tools.
#
# Native tools win the tie because they are the strictly larger capability: the sandbox is
# reachable either way, and only kbm/tools/agent.py reaches retrieval.
TOOLS_ENABLED = _flag("KBM_TOOLS", PROFILE.tools)

# Whether the generator may run Python via kbm/tools/tir.py + kbm/tools/sandbox.py. Off for any model without
# TIR training: asked for a ```python block, deepseek-math writes prose about Python.
# Also off whenever native tools are on — see above.
TIR_ENABLED = False if TOOLS_ENABLED else _flag("KBM_TIR", PROFILE.tir)

# Qwen3-style thinking mode. None means the model has none and the knob is not sent.
THINK = PROFILE.think
# Keep this >= KBM_IDLE_STOP_MINUTES: a keep-alive shorter than the idle window unloads
# the model while the pod keeps billing, so the next question pays a reload for nothing.
KEEP_ALIVE = os.environ.get("KBM_KEEP_ALIVE", "30m")

# ── Telemetry ──────────────────────────────────────────────────────────────────
# These live here rather than in api/settings.py because kbm/telemetry.py needs them and
# kbm/ must not import from api/. That import used to run the other way — a library
# module reaching into the HTTP service's settings, while api/routes.py imported the
# library — which made `telemetry` un-importable from anything that was not the service.
# api/settings.py re-exports both, so nothing that reads them had to change.
#
# The path follows DATA_DIR for the same reason the indexes do: on the pod the container
# filesystem is wiped on restart, and an event log that cannot be backfilled is exactly
# the thing not to lose (EVALUATION.md §10.9).
TELEMETRY_PATH = os.environ.get(
    "KBM_TELEMETRY_PATH", os.path.join(DATA_DIR, "telemetry", "events.jsonl")
)
# Salt for hashing usernames before they are written to the event log. Set this on
# the pod; the default makes hashes non-portable between machines, which is fine —
# they only ever need to be consistent within one deployment.
TELEMETRY_SALT = os.environ.get("KBM_TELEMETRY_SALT", "kbm-local-dev-salt")


# ── Serving ────────────────────────────────────────────────────────────────────
APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
APP_PORT = int(os.environ.get("APP_PORT", "7860"))


def app_auth() -> list[tuple[str, str]] | None:
    """Parse APP_AUTH into Gradio's auth format, or None if unset.

    Format: "alice:secret,bob:hunter2". Returned as a list of tuples, which is what
    `Blocks.launch(auth=...)` expects; None disables the login page entirely.

    This is a front door lock, not real authentication — it gates *access to the app*,
    but the per-user document isolation behind it is still just a lowercased username
    string typed into a textbox. Anyone who gets in can read any user's documents by
    typing their name. Worth knowing before sharing the URL beyond the family.
    """
    raw = os.environ.get("APP_AUTH", "").strip()
    if not raw:
        return None

    pairs = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(
                f"APP_AUTH entry {entry!r} is not 'user:pass'. "
                "Format: APP_AUTH='alice:secret,bob:hunter2'"
            )
        user, password = entry.split(":", 1)
        if not user or not password:
            raise ValueError(f"APP_AUTH entry {entry!r} has an empty username or password.")
        pairs.append((user, password))

    if not pairs:
        return None
    return pairs


# ── Device ─────────────────────────────────────────────────────────────────────
# The failure this guards against: a CPU fallback looks *identical* to success. The
# app starts, answers questions, and is merely 10-40x slower — which on a rented GPU
# is money spent on nothing. So the device is resolved once, logged loudly, and
# REQUIRE_GPU=1 turns a silent downgrade into a startup crash.

_device: str | None = None
_device_resolved = False


def resolve_device() -> str | None:
    """The device to pin models to: "cuda" if present, else None (auto-detect).

    Returning None rather than "cpu"/"mps" off-GPU is deliberate — it leaves
    sentence-transformers' own detection in charge, which is what already happens on
    the Mac (where it picks MPS). Only the CUDA case is forced, because that is the
    one we need to be sure actually happened.

    Raises RuntimeError when REQUIRE_GPU=1 and no CUDA device is visible.
    """
    global _device, _device_resolved
    if _device_resolved:
        return _device

    require_gpu = os.environ.get("REQUIRE_GPU", "").strip() in ("1", "true", "yes")

    try:
        import torch
        if torch.cuda.is_available():
            _device = "cuda"
            name = torch.cuda.get_device_name(0)
            print(f"[config] CUDA available — pinning models to GPU: {name}")
        else:
            _device = None
            if require_gpu:
                raise RuntimeError(
                    "REQUIRE_GPU=1 but torch.cuda.is_available() is False. "
                    "Models would silently run on CPU (10-40x slower). "
                    "Check the pod has a GPU attached and that torch was installed "
                    "with CUDA support, or unset REQUIRE_GPU to run on CPU anyway."
                )
            print("[config] No CUDA device — falling back to CPU/MPS auto-detect (slow on large models).")
    except ImportError:
        # torch missing entirely: nothing to pin to, and REQUIRE_GPU cannot be honoured.
        if require_gpu:
            raise RuntimeError("REQUIRE_GPU=1 but torch is not installed.")
        _device = None
        print("[config] torch not installed — leaving device selection to sentence-transformers.")

    _device_resolved = True
    return _device
