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

Usage:
    python extract.py docs/raw/textbook.pdf
    python extract.py docs/raw/textbook.pdf --force-pymupdf
"""

import argparse
import os
import sys
from dataclasses import dataclass

import pymupdf4llm


EXTRACTED_DIR = "docs/extracted"


def extract_marker(pdf_path: str, out_dir: str) -> str:
    """Run Marker on a PDF and return the path to the .mmd output file.

    Marker converts the PDF to Markdown with LaTeX-faithful equations. Models
    are downloaded to the HuggingFace cache on first run and reused thereafter;
    on a GPU pod this is fast, on CPU it is slow but correct.
    """
    # Imported lazily so --force-pymupdf runs don't pay Marker's heavy import.
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    os.makedirs(out_dir, exist_ok=True)
    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(pdf_path)
    text, _, _ = text_from_rendered(rendered)

    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(out_dir, f"{stem}.mmd")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def extract_pymupdf(pdf_path: str, out_dir: str) -> str:
    """Extract with pymupdf4llm and save as .mmd (plain markdown, no LaTeX fix)."""
    os.makedirs(out_dir, exist_ok=True)
    pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    text = "\n\n".join(p["text"] for p in pages)

    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(out_dir, f"{stem}.mmd")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


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
            out_path = extract_marker(pdf_path, out_dir)
            print(f"Saved to {out_path}")
            return ExtractResult(path=out_path, extractor="marker")
        except Exception as e:
            # pymupdf flattens equations, so this is a degraded result, not an equivalent one.
            print(f"WARNING: Marker failed ({e}).")
            print("Falling back to pymupdf4llm — EQUATIONS WILL BE MANGLED. Re-run once Marker works.")
            marker_error = str(e)
    else:
        marker_error = None

    print(f"[pymupdf4llm] Extracting {os.path.basename(pdf_path)}...")
    out_path = extract_pymupdf(pdf_path, out_dir)
    print(f"Saved to {out_path}")
    return ExtractResult(path=out_path, extractor="pymupdf4llm", marker_error=marker_error)


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
