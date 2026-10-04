"""
kbm/logsetup.py - The one place logging is configured.

There were two configurations before this, and they disagreed. api/main.py set a level
from KBM_LOG_LEVEL with a "%(levelname)s:     %(name)s - %(message)s" format; app.py set
a hardcoded INFO with a "[app] %(message)s" format, and only under `if __name__ ==
"__main__"`, so an imported app.py had no configuration at all. Two spellings of one
decision is the same trap CLAUDE.md records for CHROMA_DIR and for the decode budget:
the reader and the writer drift, and raising the level in one place silently leaves the
other alone.

This lives in kbm/ rather than api/ for exactly the reason TELEMETRY_PATH does — see the
comment in kbm/config.py: app.py and ops/ need it, and kbm/ must not import from api/.

It also imports nothing from this repo, not even kbm.config. That is deliberate: the
CPU/UI image (docker/Dockerfile.cpu) copies an explicit, short file list and carries no
torch and no pipeline, so anything this module reached for would have to be copied in
beside it.

Env vars
--------
    KBM_LOG_LEVEL     DEBUG/INFO/WARNING/ERROR   (default: INFO)
"""

import logging
import os

# Named rather than inlined so the two entry points cannot drift apart again.
FORMAT = "%(levelname)s:     %(name)s - %(message)s"
DEFAULT_LEVEL = "INFO"


def configure_logging() -> None:
    """Give the application's loggers a handler and a level.

    Without this the root logger has none, so Python falls back to its
    handler-of-last-resort — which emits WARNING and above and drops INFO entirely. Every
    `log.info` in api/routes.py was therefore invisible, including the sandbox-failure
    line whose own comment says it is logged, and every tool call. Errors still appeared,
    which is exactly why nobody noticed the rest was missing.

    basicConfig and not dictConfig: uvicorn owns its own loggers and this must not fight
    them. It only installs a handler on the ROOT logger, which is what
    `logging.getLogger(__name__)` in these packages resolves to. KBM_LOG_LEVEL=WARNING
    restores the old quiet.

    Idempotent, because basicConfig is: it returns without doing anything if the root
    logger already has a handler. So calling it from more than one entry point is safe,
    and the first caller wins.
    """
    logging.basicConfig(
        level=os.environ.get("KBM_LOG_LEVEL", DEFAULT_LEVEL).upper(),
        format=FORMAT,
    )
