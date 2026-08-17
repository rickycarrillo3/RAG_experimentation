"""
ops/idle_stop.py - Stop the pod when nobody is using it.

This is not a nicety. An always-on 24GB GPU is ~$115/mo; the same card started on
demand for ~3.5 hrs/day is ~$17/mo. The entire budget rests on the pod actually being
stopped, and nobody remembers to stop it manually every evening.

Enabled only when KBM_IDLE_STOP_MINUTES > 0 and the RunPod credentials are present, so
a laptop run can never accidentally try to stop something.
"""

import sys
import threading
import time

import httpx

from api.settings import IDLE_STOP_MINUTES, RUNPOD_API_KEY, RUNPOD_POD_ID

CHECK_INTERVAL_SECONDS = 60
RUNPOD_API = "https://rest.runpod.io/v1"


def _stop_pod() -> None:
    """Ask RunPod to stop this pod. Stop, not terminate — the volume and its 15GB of
    model weights must survive, or every wake re-downloads them."""
    r = httpx.post(
        f"{RUNPOD_API}/pods/{RUNPOD_POD_ID}/stop",
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        timeout=30.0,
    )
    r.raise_for_status()


def _watch() -> None:
    from api import routes

    idle_seconds = IDLE_STOP_MINUTES * 60
    while True:
        time.sleep(CHECK_INTERVAL_SECONDS)
        idle_for = time.monotonic() - routes.last_chat_at
        if idle_for < idle_seconds:
            continue

        # An ingest can run for many minutes without any /chat traffic. Stopping the pod
        # mid-Marker would lose the work and leave a half-built index.
        if any(j.status.value in ("queued", "running") for j in routes._jobs.values()):
            print("[idle_stop] idle, but an ingest job is active — deferring.", file=sys.stderr)
            continue

        print(f"[idle_stop] idle {idle_for / 60:.1f} min — stopping pod {RUNPOD_POD_ID}.", file=sys.stderr)
        try:
            _stop_pod()
            return
        except Exception as e:  # noqa: BLE001 - retry on the next tick rather than dying
            print(f"[idle_stop] stop failed, will retry: {e}", file=sys.stderr)


def start_watchdog() -> None:
    if not (RUNPOD_API_KEY and RUNPOD_POD_ID):
        print(
            "[idle_stop] KBM_IDLE_STOP_MINUTES is set but RUNPOD_API_KEY/RUNPOD_POD_ID "
            "are not — the pod will NOT stop itself and will bill continuously.",
            file=sys.stderr,
        )
        return
    threading.Thread(target=_watch, name="idle-stop", daemon=True).start()
    print(f"[idle_stop] watchdog armed: stop after {IDLE_STOP_MINUTES} min idle.")
