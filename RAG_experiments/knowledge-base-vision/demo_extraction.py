"""
demo_extraction.py - Extract a single PDF page via vision model and print the result.
Use this to verify LaTeX equation extraction before running a full ingest.

Usage:
    python demo_extraction.py [pdf_path] [page_number]

Defaults to docs/Probability_theory_durrett.pdf, page 10.
"""

import sys
import os
import base64
import requests
import pymupdf

VISION_MODEL = "llava"
OLLAMA_URL = "http://localhost:11434/api/generate"
DPI = 150

EXTRACT_PROMPT = (
    "Extract all text from this page exactly as it appears. "
    "Render any mathematical equations or formulas in LaTeX, "
    "using $ for inline math and $$ for display math on its own line. "
    "Return only the extracted text with no commentary or explanation."
)


def extract_page(pdf_path, page_number):
    doc = pymupdf.open(pdf_path)
    if page_number < 1 or page_number > doc.page_count:
        print(f"Page {page_number} out of range (1–{doc.page_count}).")
        sys.exit(1)

    page = doc[page_number - 1]
    mat = pymupdf.Matrix(DPI / 72, DPI / 72)
    pix = page.get_pixmap(matrix=mat)
    b64 = base64.b64encode(pix.tobytes("png")).decode()

    print(f"Sending page {page_number} of '{os.path.basename(pdf_path)}' to {VISION_MODEL}...")
    payload = {
        "model": VISION_MODEL,
        "prompt": EXTRACT_PROMPT,
        "images": [b64],
        "stream": False,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()["response"]


if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("docs", "Probability_theory_durrett.pdf")
    page_number = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)

    text = extract_page(pdf_path, page_number)
    print("\n" + "=" * 60)
    print(f"EXTRACTED TEXT (page {page_number})")
    print("=" * 60)
    print(text)
