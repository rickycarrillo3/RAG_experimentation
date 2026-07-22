"""
chunking.py - Splitting strategies for .mmd (Markdown+LaTeX) documents.

The baseline `RecursiveCharacterTextSplitter` is equation-blind: it splits on
`\\n\\n` / `\\n` / ` ` and will happily cut a `$$…$$` block in half, so half an
equation lands in one chunk and half in the next — both embed as nonsense. These
strategies keep equations intact so we can measure whether that matters:

    baseline          - the current splitter, kept for A/B (equation-blind)
    eqaware           - never split inside $$…$$ or $…$; equations stay atomic
    eqaware_context   - eqaware, plus every equation travels with the sentence
                        before and after it, so an equation is never embedded naked

All three return id-stamped chunks (via assign_chunk_ids) so ids stay reproducible
within a strategy. Ids are NOT comparable across strategies — eval.py matches by
content overlap for exactly that reason.
"""

import copy
import os
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# Display `$$…$$` (may span newlines) OR inline `$…$` (single line, no nested $).
# Display is listed first so the alternation prefers it over two inline matches.
EQUATION_RE = re.compile(r"\$\$.*?\$\$|\$[^$\n]+?\$", re.DOTALL)


# ── Chunk identity ────────────────────────────────────────────────────────────

def assign_chunk_ids(chunks: list[Document]) -> list[Document]:
    """Stamp a deterministic `<source>::<n>` id onto each chunk.

    The eval gold set labels questions by chunk id, so ids must be reproducible: the
    same .mmd re-ingested must yield the same ids, or every gold label silently rots.
    Counting per source (not globally) keeps ids stable when an unrelated file is added.
    """
    counters: dict[str, int] = {}
    for chunk in chunks:
        source = os.path.basename(chunk.metadata.get("source", "unknown"))
        n = counters.get(source, 0)
        chunk.metadata["chunk_id"] = f"{source}::{n}"
        counters[source] = n + 1
    return chunks


# ── Segmentation ──────────────────────────────────────────────────────────────

def segment_text(text: str) -> list[tuple[str, str]]:
    """Split text into an ordered list of (kind, span) where kind is 'prose' or 'eq'.

    Equation spans keep their `$` delimiters and are never broken downstream.
    """
    segments: list[tuple[str, str]] = []
    pos = 0
    for m in EQUATION_RE.finditer(text):
        if m.start() > pos:
            segments.append(("prose", text[pos:m.start()]))
        segments.append(("eq", m.group()))
        pos = m.end()
    if pos < len(text):
        segments.append(("prose", text[pos:]))
    return segments


_SENTENCE_RE = re.compile(r"[^.!?\n]*[.!?\n]|\S[^.!?\n]*$")


def _sentences(prose: str) -> list[str]:
    """Coarse sentence/line split — enough to grab the clause around an equation."""
    return [s for s in (m.group().strip() for m in _SENTENCE_RE.finditer(prose)) if s]


def _tail(text: str, overlap: int) -> str:
    """Trailing ~overlap chars of `text`, cut at a word boundary, to seed the next chunk.

    The cut must never land *inside* an equation, or the overlap seed would re-introduce
    the very split these chunkers exist to prevent — so if it does, advance past the end
    of that equation.
    """
    if overlap <= 0 or len(text) <= overlap:
        return text
    start = len(text) - overlap
    for m in EQUATION_RE.finditer(text):
        if m.start() < start < m.end():
            start = m.end()
            break
    frag = text[start:]
    sp = frag.find(" ")
    return frag[sp + 1:] if sp != -1 else frag


# ── Packers (return chunk strings for one document) ───────────────────────────

def _prose_pieces(prose: str, size: int, overlap: int) -> list[str]:
    """Break a long prose span with the baseline splitter; short spans pass through."""
    prose = prose.strip()
    if not prose:
        return []
    if len(prose) <= size:
        return [prose]
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return [p for p in splitter.split_text(prose) if p.strip()]


def _pack(units: list[tuple[str, str]], size: int, overlap: int) -> list[str]:
    """Greedily pack ('eq'|'prose', text) units into ≈size chunks.

    'eq' units are atomic — never split, even if a single equation exceeds `size`.
    'prose' units may be broken. Each new chunk is seeded with the previous chunk's
    tail (~overlap chars) for continuity.
    """
    chunks: list[str] = []
    cur = ""

    def flush():
        nonlocal cur
        if cur.strip():
            chunks.append(cur.strip())
        cur = _tail(cur.strip(), overlap)

    for kind, text in units:
        pieces = [text] if kind == "eq" else _prose_pieces(text, size, overlap)
        for piece in pieces:
            if cur.strip() and len(cur) + len(piece) + 1 > size:
                flush()
            cur = f"{cur} {piece}".strip() if cur.strip() else piece
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def _bundle_with_context(segments: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Turn each equation into an atomic unit carrying its neighbouring sentences.

    prev sentence + equation + next sentence become one 'eq' (unsplittable) unit, so
    the equation is never embedded without the prose that gives it meaning. Prose not
    adjacent to an equation stays a normal, splittable 'prose' unit.
    """
    units: list[tuple[str, str]] = []
    i = 0
    while i < len(segments):
        kind, text = segments[i]
        if kind != "eq":
            # Prose not adjacent to an equation is still content — keep it as a unit.
            units.append(("prose", text))
            i += 1
            continue
        # Consume the trailing sentence of the preceding prose unit, if any.
        prefix = ""
        if units and units[-1][0] == "prose":
            sents = _sentences(units[-1][1])
            if sents:
                prefix = sents[-1]
                remainder = " ".join(sents[:-1])
                if remainder:
                    units[-1] = ("prose", remainder)
                else:
                    units.pop()
        # Consume the leading sentence of the following prose segment; the rest stays in
        # place so the next loop iteration appends it as its own prose unit (no loss).
        suffix = ""
        if i + 1 < len(segments) and segments[i + 1][0] == "prose":
            sents = _sentences(segments[i + 1][1])
            if sents:
                suffix = sents[0]
                segments[i + 1] = ("prose", " ".join(sents[1:]))
        bundle = " ".join(p for p in (prefix, text, suffix) if p).strip()
        units.append(("eq", bundle))
        i += 1
    return units


# ── Public splitters (list[Document] -> list[Document]) ───────────────────────

def _rechunk(docs: list[Document], make_chunks) -> list[Document]:
    out: list[Document] = []
    for doc in docs:
        for text in make_chunks(doc.page_content):
            out.append(Document(page_content=text, metadata=copy.deepcopy(doc.metadata)))
    return assign_chunk_ids(out)


def split_baseline(docs: list[Document]) -> list[Document]:
    """The current pipeline's splitter, equation-blind. Kept for A/B comparison."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return assign_chunk_ids(splitter.split_documents(docs))


def split_equation_aware(docs: list[Document]) -> list[Document]:
    """Pack prose and equations to ≈CHUNK_SIZE, never splitting an equation."""
    return _rechunk(docs, lambda t: _pack(segment_text(t), CHUNK_SIZE, CHUNK_OVERLAP))


def split_equation_aware_with_context(docs: list[Document]) -> list[Document]:
    """Like eqaware, but each equation carries its adjacent sentences into its chunk."""
    return _rechunk(
        docs,
        lambda t: _pack(_bundle_with_context(segment_text(t)), CHUNK_SIZE, CHUNK_OVERLAP),
    )


CHUNKERS = {
    "baseline": split_baseline,
    "eqaware": split_equation_aware,
    "eqaware_context": split_equation_aware_with_context,
}
