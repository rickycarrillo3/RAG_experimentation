"""
kbm/latex_norm.py - Turn LaTeX into readable text before embedding.

A general text embedder's tokenizer has never meaningfully seen LaTeX: `\\theta`,
`\\mathbb{E}`, `_{t+1}` shred into backslash/brace subword junk, so the equation's
vector encodes punctuation, not math. `normalize_latex` runs the text through
pylatexenc, which renders commands to their readable form (`\\theta`→`θ`, `\\cos`→
`cos`, `\\frac{a}{b}`→`a/b`) — the same glyph-level representation prose-trained
models actually handle.

Used ONLY to compute embeddings (see retrieval.NormalizingEmbeddings). The stored
`page_content` stays raw LaTeX, so BM25, the reranker, the LLM context, and the eval
overlap-matcher all still see the original.
"""

import re

from pylatexenc.latex2text import LatexNodes2Text

_converter = LatexNodes2Text(strict_latex_spaces=False, keep_comments=False)
_whitespace = re.compile(r"\s+")


def normalize_latex(text: str) -> str:
    """Render LaTeX to readable text; collapse whitespace. Never raises.

    Plain prose passes through essentially unchanged. Malformed LaTeX (pylatexenc can
    choke on it) falls back to the original text rather than dropping the chunk.
    """
    try:
        rendered = _converter.latex_to_text(text)
    except Exception:
        return text
    return _whitespace.sub(" ", rendered).strip()
