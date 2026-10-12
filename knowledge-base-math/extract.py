"""
extract.py - PDF extraction for math documents.

Marker is the default: it is the only extractor here that produces faithful LaTeX,
and this is a math RAG system. pymupdf4llm is used only when explicitly asked for
(--force-pymupdf) or as a fallback when Marker fails.

There is deliberately NO math auto-detection. The old `_has_math()` sampled the PDF
with pymupdf4llm and grepped the text for LaTeX tokens (`\frac`, `$$`, …), which
cannot work: pymupdf4llm never emits LaTeX. Measured on OpenStax Calculus Vol 1 —
a *calculus textbook* — that check finds zero LaTeX tokens and routes the document
to the wrong extractor. Unicode-glyph detection fails on the same document too,
because the equations are embedded images that pymupdf hands to Tesseract. You
cannot tell whether a PDF contains math by reading text that has already lost it.

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
    python extract.py docs/raw/textbook.pdf --force-pymupdf
"""

import argparse
import base64
import io
import json
import os
import re
import sys
from dataclasses import dataclass, field

import pymupdf4llm


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
    # Imported lazily so --force-pymupdf runs don't pay Marker's heavy import.
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
    # Replace it with our own sentinel so chunking.py has one marker format to parse
    # regardless of which extractor produced the .mmd.
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


def extract_pymupdf(pdf_path: str, out_dir: str) -> tuple[str, list[dict]]:
    """Extract with pymupdf4llm and save as .mmd (plain markdown, no LaTeX fix).

    Returns (.mmd path, raw images [{"page", "bytes_b64"}]) — same shape as
    extract_marker, so ingest.py sees one uniform interface regardless of extractor.
    """
    os.makedirs(out_dir, exist_ok=True)
    image_dir = os.path.join(out_dir, "_pymupdf_images")
    pages = pymupdf4llm.to_markdown(
        pdf_path, page_chunks=True, write_images=True,
        image_path=image_dir, image_format="png",
    )

    parts = []
    raw_images: list[dict] = []
    for i, p in enumerate(pages):
        page_no = p.get("metadata", {}).get("page", i)
        parts.append(f"<!-- page:{page_no} -->\n\n{p['text']}")
        for img in p.get("images") or []:
            path = img.get("file")
            if not path or not os.path.exists(path):
                continue
            try:
                with open(path, "rb") as f:
                    raw_images.append({
                        "page": page_no,
                        "bytes_b64": base64.b64encode(f.read()).decode("ascii"),
                    })
            except OSError as e:
                print(f"WARNING: could not read extracted image {path}: {e}")
    text = "\n\n".join(parts)

    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(out_dir, f"{stem}.mmd")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path, raw_images


@dataclass(frozen=True)
class ExtractResult:
    """What came out, and which extractor produced it.

    The second field is the point. `extract()` used to return a bare path on both the
    Marker path and the pymupdf4llm fallback, so no caller could tell them apart — and
    since the fallback does not raise, an ingest whose equations had been flattened to
    Unicode reported "Indexed. N total chunks" like any success. A return type that
    cannot express degradation guarantees the caller will report success.
    """

    path: str
    extractor: str                      # "marker" | "pymupdf4llm"
    marker_error: str | None = None     # why Marker was skipped, when it was tried and failed
    images: list[dict] = field(default_factory=list)  # [{"page": int|None, "bytes_b64": str}]

    @property
    def degraded(self) -> bool:
        """True when the text contains no LaTeX, whatever the reason.

        `--force-pymupdf` is a deliberate choice rather than a failure, but the chunks
        are equally unusable for math either way, so it counts as degraded too. Why it
        happened is `marker_error`'s job, not this flag's.
        """
        return self.extractor != "marker"


def extract_detailed(
    pdf_path: str, force_pymupdf: bool = False, out_dir: str = EXTRACTED_DIR
) -> ExtractResult:
    """Extract, reporting which extractor won. See `extract()` for the plain-path form.

    Mirrors the retrieve()/retrieve_detailed() split in kbm/retrieval.py: the simple entry
    point stays simple, and the caller that needs to act on the details asks for them.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not force_pymupdf:
        print(f"[marker] Extracting {os.path.basename(pdf_path)} (LaTeX-faithful; slow on CPU)...")
        try:
            out_path, raw_images = extract_marker(pdf_path, out_dir)
            print(f"Saved to {out_path}")
            _write_raw_images(out_path, raw_images)
            return ExtractResult(path=out_path, extractor="marker", images=raw_images)
        except Exception as e:
            # pymupdf flattens equations, so this is a degraded result, not an equivalent one.
            print(f"WARNING: Marker failed ({e}).")
            print("Falling back to pymupdf4llm — EQUATIONS WILL BE MANGLED. Re-run once Marker works.")
            marker_error = str(e)
    else:
        marker_error = None

    print(f"[pymupdf4llm] Extracting {os.path.basename(pdf_path)}...")
    out_path, raw_images = extract_pymupdf(pdf_path, out_dir)
    print(f"Saved to {out_path}")
    _write_raw_images(out_path, raw_images)
    return ExtractResult(
        path=out_path, extractor="pymupdf4llm", marker_error=marker_error, images=raw_images
    )


def extract(pdf_path: str, force_pymupdf: bool = False, out_dir: str = EXTRACTED_DIR) -> str:
    """Extract and return the .mmd path. Unchanged signature for the CLI and eval scripts."""
    return extract_detailed(pdf_path, force_pymupdf=force_pymupdf, out_dir=out_dir).path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PDF to markdown for RAG ingestion.")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--force-pymupdf", action="store_true", help="Skip Marker, use pymupdf4llm directly")
    args = parser.parse_args()

    try:
        result = extract(args.pdf, force_pymupdf=args.force_pymupdf)
        print(f"\nDone. Output: {result}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
