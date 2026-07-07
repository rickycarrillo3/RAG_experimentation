"""
extract.py - PDF extraction for math documents.

Uses Nougat for math-heavy PDFs (produces proper LaTeX).
Falls back to pymupdf4llm for non-math or lightweight PDFs.

Usage:
    python extract.py docs/raw/textbook.pdf
    python extract.py docs/raw/textbook.pdf --force-pymupdf
"""

import argparse
import os
import subprocess
import sys
import pymupdf4llm


EXTRACTED_DIR = "docs/extracted"
LATEX_PATTERNS = [
    r"\frac", r"\sum", r"\int", r"\sqrt", r"\begin{",
    r"\alpha", r"\beta", r"\theta", r"\pi", r"\infty",
    r"$$", r"\leq", r"\geq", r"\nabla", r"\partial",
]


def _has_math(pdf_path: str, sample_pages: int = 5) -> bool:
    """Sample the first few pages with pymupdf4llm and check for LaTeX signals."""
    pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    sample = " ".join(p["text"] for p in pages[:sample_pages])
    return any(pattern in sample for pattern in LATEX_PATTERNS)


def extract_nougat(pdf_path: str, out_dir: str) -> str:
    """Run Nougat on a PDF and return the path to the .mmd output file."""
    os.makedirs(out_dir, exist_ok=True)
    result = subprocess.run(
        ["nougat", pdf_path, "-o", out_dir, "--no-skipping"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Nougat failed:\n{result.stderr}")

    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    mmd_path = os.path.join(out_dir, f"{stem}.mmd")
    if not os.path.exists(mmd_path):
        raise FileNotFoundError(f"Nougat ran but {mmd_path} was not created.")
    return mmd_path


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


def extract(pdf_path: str, force_pymupdf: bool = False, out_dir: str = EXTRACTED_DIR) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")


    if force_pymupdf:
        print(f"[pymupdf4llm] Extracting {os.path.basename(pdf_path)}...")
        out_path = extract_pymupdf(pdf_path, out_dir)
        print(f"Saved to {out_path}")
        return out_path

    print(f"Sampling PDF to detect math content...")
    is_math = _has_math(pdf_path)

    if is_math:
        print(f"Math detected — using Nougat for proper LaTeX extraction.")
        try:
            out_path = extract_nougat(pdf_path, out_dir)
            print(f"Saved to {out_path}")
            return out_path
        except (RuntimeError, FileNotFoundError) as e:
            print(f"Nougat failed ({e}), falling back to pymupdf4llm.")

    print(f"[pymupdf4llm] Extracting {os.path.basename(pdf_path)}...")
    out_path = extract_pymupdf(pdf_path, out_dir)
    print(f"Saved to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PDF to markdown for RAG ingestion.")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--force-pymupdf", action="store_true", help="Skip Nougat, use pymupdf4llm directly")
    args = parser.parse_args()

    try:
        result = extract(args.pdf, force_pymupdf=args.force_pymupdf)
        print(f"\nDone. Output: {result}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
