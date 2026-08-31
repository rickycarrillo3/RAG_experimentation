"""
ops/idle_stop.py - Stop the pod when nobody is using it.

This is not a nicety. An always-on 24GB GPU is ~$115/mo; the same card started on
demand for ~3.5 hrs/day is ~$17/mo. The entire budget rests on the pod actually being
stopped, and nobody remembers to stop it manually every evening.

Enabled only when KBM_IDLE_STOP_MINUTES > 0 and the RunPod credentials are present, so
a laptop run can never accidentally try to stop something.
"""

import logging
import threading
import time

from api.settings import IDLE_STOP_MINUTES, RUNPOD_API_KEY, RUNPOD_POD_ID
from ops.runpod import stop_pod

# This runs inside the API process — api/main.py's lifespan starts the thread — so
# kbm/logsetup.py's configuration is already installed by the time anything here emits.
# It also settles an inconsistency: four of these five lines went to stderr and the
# "watchdog armed" one went to stdout, so `2>/dev/null` hid some of the watchdog's
# output and not the rest.
log = logging.getLogger(__name__)

CHECK_INTERVAL_SECONDS = 60


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
            log.info("idle, but an ingest job is active — deferring.")
            continue

        log.info("idle %.1f min — stopping pod %s.", idle_for / 60, RUNPOD_POD_ID)
        try:
            stop_pod(RUNPOD_POD_ID, RUNPOD_API_KEY)
            return
        except Exception as e:  # noqa: BLE001 - retry on the next tick rather than dying
            log.warning("stop failed, will retry: %s", e)


def start_watchdog() -> None:
    if not (RUNPOD_API_KEY and RUNPOD_POD_ID):
        log.warning(
            "KBM_IDLE_STOP_MINUTES is set but RUNPOD_API_KEY/RUNPOD_POD_ID "
            "are not — the pod will NOT stop itself and will bill continuously."
        )
        return
    threading.Thread(target=_watch, name="idle-stop", daemon=True).start()
    log.info("watchdog armed: stop after %s min idle.", IDLE_STOP_MINUTES)
