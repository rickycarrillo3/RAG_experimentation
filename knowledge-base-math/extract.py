"""
extract.py - PDF extraction for math documents.

Marker is the only extractor: it is the only one that produces faithful LaTeX, and
this is a math RAG system. If Marker fails, extraction FAILS — there is no fallback.

pymupdf4llm used to be the fallback (and, before that, the choice for "non-math"
PDFs). Both were removed. A fallback that flattens equations to Unicode indexes
"successfully" and quietly ruins retrieval for that document, which is worse than an
error the family can see and retry. And "non-math" could not be detected: pymupdf4llm
never emits LaTeX, and on OpenStax Calculus Vol 1 the equations are embedded images,
so neither LaTeX-token nor Unicode-glyph checks found any math in a calculus textbook.

Also extracts page-bearing figures (Information_files/IMAGE_DESIGN.md), as two
artifacts alongside the .mmd:
  - a `<!-- page:N -->` sentinel inserted into the .mmd text at every page boundary,
    so a page number survives into chunk metadata (kbm/chunking.py strips it before
    indexing — it must never reach BM25/dense search or the LLM context).
  - `<stem>.images.raw.json`, the raw image bytes (base64) + page numbers. ingest.py
    reads this to caption figures without re-running Marker — extract.py and ingest.py
    are separate CLI steps connected only by files on disk, and extraction must keep
    working with no Ollama server running (it is tested independently, and the frozen
    eval corpus in docs/extracted/ is built this way).

Usage:
    python extract.py docs/raw/textbook.pdf
"""

import argparse
import base64
import io
import json
import os
import re
import sys


EXTRACTED_DIR = "docs/extracted"

# Marker's own `_page_<N>_...` image filenames encode the page an image came from.
_MARKER_IMAGE_PAGE_RE = re.compile(r"page_(\d+)")


def _raw_images_path(mmd_path: str) -> str:
    stem = os.path.splitext(mmd_path)[0]
    return f"{stem}.images.raw.json"


def _write_raw_images(mmd_path: str, images: list[dict]) -> None:
    """Write `<stem>.images.raw.json` next to the .mmd: [{"page": int|None, "bytes_b64": str}].

    Always written, even when `images` is empty, so ingest.py can tell "no figures in
    this document" apart from "extract.py predates this feature" by the file's absence.
    """
    with open(_raw_images_path(mmd_path), "w", encoding="utf-8") as f:
        json.dump(images, f)


def extract_marker(pdf_path: str, out_dir: str) -> tuple[str, list[dict]]:
    """Run Marker on a PDF. Returns (.mmd path, raw images [{"page", "bytes_b64"}]).

    Marker converts the PDF to Markdown with LaTeX-faithful equations. Models
    are downloaded to the HuggingFace cache on first run and reused thereafter;
    on a GPU pod this is fast, on CPU it is slow but correct.
    """
    # Imported lazily so importing this module (e.g. by api/routes.py) stays cheap.
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    os.makedirs(out_dir, exist_ok=True)
    # paginate_output inserts a `{N}-------...` delimiter line between pages, which is
    # the only reliable source of page boundaries Marker's own API offers — its combined
    # markdown has no other per-page structure. Best-effort: an older marker-pdf without
    # this config key still extracts, just without page numbers for this document.
    try:
        converter = PdfConverter(
            artifact_dict=create_model_dict(), config={"paginate_output": True}
        )
    except Exception as e:  # noqa: BLE001 - an older marker-pdf without this config key
        print(f"WARNING: could not request paginated output ({e}); page numbers for "
              f"figures in this document will be unavailable.")
        converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(pdf_path)
    text, _, images = text_from_rendered(rendered)

    # Marker's page-break line looks like "{3}------------------------------------------".
    # Replace it with our own sentinel, the one format chunking.py parses.
    text = re.sub(r"\{(\d+)\}-+", lambda m: f"<!-- page:{m.group(1)} -->", text)

    raw_images: list[dict] = []
    for filename, pil_image in (images or {}).items():
        page_match = _MARKER_IMAGE_PAGE_RE.search(filename)
        page = int(page_match.group(1)) if page_match else None
        try:
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            raw_images.append({
                "page": page,
                "bytes_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
            })
        except Exception as e:  # noqa: BLE001 - one bad image must not fail extraction
            print(f"WARNING: could not encode image {filename}: {e}")

    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(out_dir, f"{stem}.mmd")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path, raw_images


def extract(pdf_path: str, out_dir: str = EXTRACTED_DIR) -> str:
    """Extract a PDF with Marker; return the .mmd path. Raises if Marker fails.

    Also writes `<stem>.images.raw.json` beside the .mmd for ingest.py's figure sidecar.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"[marker] Extracting {os.path.basename(pdf_path)} (LaTeX-faithful; slow on CPU)...")
    try:
        out_path, raw_images = extract_marker(pdf_path, out_dir)
    except Exception as e:
        raise RuntimeError(f"Marker failed: {type(e).__name__}: {e}") from e
    _write_raw_images(out_path, raw_images)
    print(f"Saved to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PDF to markdown for RAG ingestion.")
    parser.add_argument("pdf", help="Path to the PDF file")
    args = parser.parse_args()

    try:
        result = extract(args.pdf)
        print(f"\nDone. Output: {result}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
