"""
ops/runpod.py - The RunPod REST calls that start and stop a pod.

Two callers, on two different hosts, and that is the whole reason this file exists:

    ops/idle_stop.py   runs inside the API process on the GPU pod and STOPS it.
    app.py             runs on the always-on CPU pod and STARTS it, when a family
                       member asks a question and the generator is not there.

Sleep and wake are the same endpoint with a different verb, so they belong in the same
place. Left where it was, `RUNPOD_API` would now be declared twice on two machines, and
the pair would drift the way CHROMA_DIR/BM25_DIR did before kbm/config.py (CLAUDE.md:
"when you add a path, a model, or a host, define it once and import it").

**Credentials are arguments, not imports.** ops/ is documented as a client of api/
(kbm/__init__.py), and api.settings is where RUNPOD_API_KEY/RUNPOD_POD_ID live — but
api/ does not ship in the CPU image at all, so importing them here would make this
module unimportable on the exact host that needs to start the pod. Each caller reads
its own configuration and passes it in; this file knows only HTTP.

⚠️ RUNPOD_API_KEY controls the whole RunPod account — start, stop, terminate, spend.
Putting it on the CPU host is a deliberate trade (DEPLOYMENT.md §5, "the wake gap"):
it is what turns a dead URL into a progress bar, and it is why the CPU pod must have
APP_AUTH set. It is not a credential to put in anything a family member can read.
"""

import httpx

RUNPOD_API = "https://rest.runpod.io/v1"

# Generous, and on purpose. These calls are control-plane operations that happen at most
# a few times a day; a timeout here costs a retry, while a timeout that fires early on a
# slow-but-succeeding start would have us report failure for a pod that is coming up.
TIMEOUT_SECONDS = 30.0


def _call(action: str, pod_id: str, api_key: str) -> None:
    r = httpx.post(
        f"{RUNPOD_API}/pods/{pod_id}/{action}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=TIMEOUT_SECONDS,
    )
    r.raise_for_status()


def stop_pod(pod_id: str, api_key: str) -> None:
    """Stop the pod. Stop, not terminate — the volume and its ~15GB of model weights
    must survive, or every wake re-downloads them (and the indexes are the only copy
    of the family's documents, ARCHITECTURE.md §5)."""
    _call("stop", pod_id, api_key)


def start_pod(pod_id: str, api_key: str) -> None:
    """Start the pod. Returns as soon as RunPod accepts the request, which is well
    before anything is serving: a start recreates the container from the image, so the
    caller still has to wait for the processes. Poll GET /healthz for `model_loaded`
    rather than assuming this call means ready — DEPLOYMENT.md §5, "wake is two things".
    """
    _call("start", pod_id, api_key)
