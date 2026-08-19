"""
config.py - Everything about this deployment that changes between machines.

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
    KBM_NUM_PREDICT   decode cap, in tokens               (default: 350)
    KBM_KEEP_ALIVE    how long Ollama holds the weights   (default: 30m)
"""

import os

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
NUM_PREDICT = int(os.environ.get("KBM_NUM_PREDICT", "350"))
# Keep this >= KBM_IDLE_STOP_MINUTES: a keep-alive shorter than the idle window unloads
# the model while the pod keeps billing, so the next question pays a reload for nothing.
KEEP_ALIVE = os.environ.get("KBM_KEEP_ALIVE", "30m")

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
