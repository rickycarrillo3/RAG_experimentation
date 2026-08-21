"""
prefetch_models.py - Download every model the app needs, before anyone uses it.

The problem this solves is a cold `/workspace`. Three model sets live in the
HuggingFace cache, and they are fetched at very different moments:

    bge-small embedder    ~130MB   at app.py import      (startup blocks, silently)
    bge-reranker-v2-m3    ~2.2GB   at app.py import      (startup blocks, silently)
    Marker / surya        ~3-4GB   on the FIRST UPLOAD   (nothing warms it at all)

The first two only make startup look like it hung. Marker is the real trap: its
models are built inside extract_marker() (extract.py), which only runs when someone
uploads a PDF — and app.py's handle_upload wraps that in a try/except that reports
any failure as "Ingestion failed: <e>". So on a fresh pod, the first family PDF
either stalls for minutes or surfaces a multi-GB download as a generic error.

Fetching through the pipeline's own loaders (rather than snapshot_download) is
deliberate: it pulls exactly the files the app will use, and doubles as a "do these
actually load on this machine?" check — which is also where a broken CUDA install
shows up, before the family is waiting on it.

Usage:
    python prefetch_models.py                          # embedder + reranker + Marker
    python prefetch_models.py --skip-marker            # the ~3-4GB Marker set only
    python prefetch_models.py --embed-model BAAI/bge-m3 --embed-model BAAI/bge-small-en-v1.5
"""

import argparse
import os
import sys
import time


def _hr() -> None:
    print("─" * 72)


def prefetch_embedders(models: list[str]) -> list[str]:
    """Load each embedder once so its files land in the HF cache. Returns failures."""
    from kbm import retrieval

    failures = []
    for name in dict.fromkeys(models):  # dedupe, preserve order
        print(f"  ↓ embedder {name}")
        t0 = time.perf_counter()
        try:
            retrieval.load_embeddings(name)
            print(f"    ok ({time.perf_counter() - t0:.1f}s)")
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            failures.append(f"embedder {name}")
    return failures


def prefetch_reranker() -> list[str]:
    from kbm import retrieval

    print(f"  ↓ reranker {retrieval.RERANK_MODEL}")
    t0 = time.perf_counter()
    try:
        retrieval.load_reranker()
        print(f"    ok ({time.perf_counter() - t0:.1f}s)")
        return []
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {e}")
        return [f"reranker {retrieval.RERANK_MODEL}"]


def prefetch_marker() -> list[str]:
    """Build Marker's model dict, which is what pulls the surya weights.

    This is the whole reason the script exists — nothing else in the serving path
    touches these models until a user uploads a PDF.
    """
    print("  ↓ Marker / surya models (~3-4GB on a cold cache)")
    t0 = time.perf_counter()
    try:
        from marker.models import create_model_dict
        create_model_dict()
        print(f"    ok ({time.perf_counter() - t0:.1f}s)")
        return []
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {e}")
        print("    PDF upload will fall back to pymupdf4llm, which mangles equations.")
        return ["marker/surya"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-download the HuggingFace models the app needs.",
    )
    parser.add_argument(
        "--embed-model", action="append", dest="embed_models", metavar="NAME",
        help="Embedding model to prefetch (repeatable; defaults to retrieval.EMBED_MODEL)",
    )
    parser.add_argument("--skip-marker", action="store_true",
                        help="Skip Marker/surya (the largest set; not needed for query-only use)")
    parser.add_argument("--skip-reranker", action="store_true", help="Skip the cross-encoder")
    args = parser.parse_args()

    # Imported after arg parsing so --help stays instant.
    from kbm import retrieval
    from kbm.config import resolve_device

    embed_models = args.embed_models or [retrieval.EMBED_MODEL]

    _hr()
    print("Prefetching models")
    print(f"  HF_HOME       = {os.environ.get('HF_HOME', '(default: ~/.cache/huggingface)')}")
    print(f"  device        = {resolve_device() or 'auto (CPU/MPS)'}")
    _hr()

    t0 = time.perf_counter()
    failures = prefetch_embedders(embed_models)
    if not args.skip_reranker:
        failures += prefetch_reranker()
    if not args.skip_marker:
        failures += prefetch_marker()

    _hr()
    if failures:
        print(f"Done in {time.perf_counter() - t0:.1f}s with {len(failures)} failure(s):")
        for f in failures:
            print(f"  ✗ {f}")
        print("\nThese would otherwise have failed later, mid-use. Fix before serving.")
        return 1

    print(f"All models present ({time.perf_counter() - t0:.1f}s). Nothing will download mid-use.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
