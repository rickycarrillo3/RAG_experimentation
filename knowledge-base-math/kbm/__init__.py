"""kbm - the library the entry points and the API are built from.

The split is between things that are *imported* and things that are *run*. This package
holds the first: configuration, retrieval, chunking, LaTeX normalization, telemetry, and
the tool protocols under `kbm.tools`. The scripts a person invokes — extract.py,
ingest.py, query.py, app.py, test_chat.py, prefetch_models.py — stay at the project root,
which is why every command in CLAUDE.md still reads `python ingest.py …` and not
`python -m something.ingest`.

Nothing here imports from `api/`, which is the layering this package exists to make
visible: `api/` is a client of the library, and `ops/` is a client of `api/`.
"""
