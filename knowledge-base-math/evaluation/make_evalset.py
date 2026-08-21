"""
make_evalset.py - Build the gold set: question -> the chunk that should be retrieved.

Samples answerable chunks from a user's index, asks an instruct model to write the
question a student would ask that each chunk answers, and rejects questions that leak
the chunk's own wording.

    (run from knowledge-base-math/)
    python evaluation/make_evalset.py --user calctest --n 50
    python evaluation/make_evalset.py --user calctest --n 50 --model qwen2:7b

Why the filtering exists
------------------------
A question generated *from* a chunk inherits that chunk's vocabulary. Such a question is
findable by string-matching alone, which flatters BM25 and makes the reranker look
useless. The first gold set built by this script (before filtering) had 18/50 questions
with >=60% word overlap with their own chunk, and a mean overlap of 0.43 — it was, in
effect, a spelling test.

Two defences, because they catch different failures:
  1. An eligibility gate on the CHUNK. Some chunks cannot produce a fair question at all
     — figure captions, reference lists, bare number tables. The old gold set contained
     "What do the straight orange lines in the top row of plots represent?", which no
     retrieval system should be expected to answer. No prompt can fix a bad target.
  2. A leakage filter on the QUESTION, with retries.

Output is still a DRAFT: read evaluation/goldset_review.md (worst offenders first) before
trusting the numbers. See EVALUATION.md.
"""

import argparse
import json
import os
import random
import re
import sys

# This lives in evaluation/; the pipeline modules (kbm, …) are one level up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_ollama import ChatOllama

from kbm.config import OLLAMA_BASE_URL
from kbm.retrieval import chunk_id, load_bm25

EVAL_DIR = "evaluation"
GOLDSET_PATH = os.path.join(EVAL_DIR, "goldset.jsonl")
REVIEW_PATH = os.path.join(EVAL_DIR, "goldset_review.md")
QUESTION_MODEL = "qwen2:7b"  # an *instruct* model; deepseek-math is a solver, not a writer

PROMPT_VERSION = 3  # bump when PROMPT or leak_score changes, so scores stay comparable

LEAK_THRESHOLD = 0.6  # max fraction of a question's DISTINCTIVE words that may come from the chunk
MAX_ATTEMPTS = 3

MATH_SIGNALS = [
    "$$", "\\frac", "\\int", "\\sum", "\\sqrt", "\\begin{", "\\lim",
    "\\partial", "\\nabla", "\\alpha", "\\beta", "\\theta", "\\mathbb",
]

# Question words so common they say nothing about leakage.
STOPWORDS = set("""
what how why does do is are was were the a an of in to for and or with that this it on as by
its their our we you can could would should when which who whom where if then than there here
be been being have has had at from into about over under between among not no yes any some
""".split())

# Phrasing that only makes sense if you are LOOKING AT the chunk.
REFERENTIAL = re.compile(
    r"\b(this|the)\s+(passage|text|document|excerpt|benchmark|table|figure|plot|graph|chart|"
    r"paper|section|chapter|example|experiment|study|image)\b"
    r"|\bas (shown|described|given|seen|illustrated)\b"
    r"|\b(shown|described|given|illustrated|mentioned|listed) (in|above|below|here)\b"
    r"|\b(the )?(top|bottom|middle|left|right) (row|column|panel|plot|graph)\b"
    r"|\baccording to (the|this)\b",
    re.I,
)


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower())} - STOPWORDS


def build_corpus_df(chunks: list) -> tuple[dict[str, int], int]:
    """Document frequency of every word across the corpus."""
    df: dict[str, int] = {}
    for c in chunks:
        for w in words(c.page_content):
            df[w] = df.get(w, 0) + 1
    return df, len(chunks)


def distinctive(w: str, df: dict[str, int], n_docs: int, max_df_ratio: float = 0.10) -> bool:
    """Is this word specific to a few chunks, rather than corpus-wide vocabulary?"""
    return df.get(w, 0) <= max(1, int(n_docs * max_df_ratio))


def leak_score(question: str, chunk_text: str, df: dict, n_docs: int) -> float:
    """Fraction of the question's DISTINCTIVE words that were lifted from the chunk.

    Naive word overlap does not work here, and getting this wrong is the difference
    between a filter that helps and one that just deletes good questions. In a calculus
    corpus, "derivative", "function" and "rule" appear in nearly every chunk: a question
    using them is not leaking, it is simply *on topic*. Raw overlap punishes that, and it
    scored honest questions at 30-40% while barely separating them from real copying.

    So only *distinctive* words count — those appearing in <=10% of chunks. Those are the
    ones a student could not have guessed without seeing the chunk, and reusing them is
    what makes a question findable by string-matching alone.

    A question with no distinctive words at all is asking in plain language: leak 0.0.
    """
    q_distinct = {w for w in words(question) if distinctive(w, df, n_docs)}
    if not q_distinct:
        return 0.0
    return len(q_distinct & words(chunk_text)) / len(q_distinct)


def is_mathy(text: str) -> bool:
    return any(sig in text for sig in MATH_SIGNALS)


def is_answerable(text: str) -> bool:
    """Can this chunk plausibly be the answer to a standalone question?

    Rejects chunks that exist only as scaffolding around content: figure captions,
    reference lists, exercise-answer keys, and blocks that are mostly digits/symbols.
    A chunk that fails here would generate an unanswerable question and then punish
    the retriever for not finding it.
    """
    t = text.strip()
    if len(t) < 200:
        return False

    letters = sum(c.isalpha() for c in t)
    if letters / len(t) < 0.45:  # number tables, bare equation dumps, symbol soup
        return False

    # Needs some actual prose: a couple of reasonably long sentences.
    sentences = [s for s in re.split(r"[.!?]\s", t) if len(s.split()) >= 6]
    if len(sentences) < 2:
        return False

    head = t[:120].lower()
    if re.match(r"^\s*(figure|fig\.|table|exhibit|chart)\s*\d", head):
        return False
    if re.match(r"^\s*(references|bibliography|index|contents|appendix)\b", head):
        return False
    # Reference-list bodies: many "Author, Year" patterns.
    if len(re.findall(r"\b(19|20)\d{2}\b", t)) >= 4 and letters / len(t) < 0.7:
        return False
    return True


PROMPT = """You are helping build a benchmark for a math tutoring search engine.

Below is a passage from a math textbook. Write ONE question that a student would
naturally ask, which this passage answers.

Hard rules:
- The question must STAND ALONE. A student who has never seen this passage must be able
  to ask it. Never refer to "the passage", "the text", "this example", "the figure",
  "the table", or anything "shown above/below".
- Use the student's OWN WORDS. Do not copy distinctive phrases or notation from the
  passage. If you reuse its exact wording, the question is useless to us — it tests
  string-matching, not understanding.
- Ask about the concept, rule, or result — something a student would actually want to know.
- Output ONLY the question. No preamble, no quotes, no numbering.

Passage:
---
{chunk}
---

Question:"""

RETRY_SUFFIX = """

Your previous attempt reused too many words from the passage, or referred to it directly:
  "{previous}"

Rewrite it: same underlying question, but phrased the way a student would ask it BEFORE
ever seeing this passage. Use plain, everyday wording. Output ONLY the question."""


def clean_question(raw: str) -> str:
    q = raw.strip().split("\n")[0].strip()
    q = re.sub(r"^(question|q)\s*[:.\-]\s*", "", q, flags=re.I)
    return q.strip().strip('"').strip()


def reject_reason(question: str, chunk_text: str, df: dict, n_docs: int) -> str | None:
    if len(question) < 15 or "?" not in question:
        return "not a question"
    if REFERENTIAL.search(question):
        return "refers to the chunk itself"
    score = leak_score(question, chunk_text, df, n_docs)
    if score >= LEAK_THRESHOLD:
        return f"leaks chunk wording ({score:.0%} of distinctive words)"
    return None


def sample_chunks(chunks: list, n: int, seed: int = 0) -> list:
    """Sample n *answerable* chunks, preferring math-bearing ones."""
    rng = random.Random(seed)
    usable = [c for c in chunks if is_answerable(c.page_content)]
    mathy = [c for c in usable if is_mathy(c.page_content)]
    prose = [c for c in usable if not is_mathy(c.page_content)]
    rng.shuffle(mathy)
    rng.shuffle(prose)

    want_math = min(len(mathy), int(n * 0.7))
    picked = mathy[:want_math] + prose[: n - want_math]
    if len(picked) < n:
        picked += mathy[want_math : want_math + (n - len(picked))]
    rng.shuffle(picked)
    return picked[:n]


def generate_question(llm, chunk_text: str, df: dict, n_docs: int) -> tuple[str | None, str | None]:
    """Generate a question, retrying when it leaks. Returns (question, reject_reason)."""
    previous = None
    for _ in range(MAX_ATTEMPTS):
        prompt = PROMPT.format(chunk=chunk_text)
        if previous:
            prompt += RETRY_SUFFIX.format(previous=previous)
        try:
            question = clean_question(llm.invoke(prompt).content)
        except Exception as e:
            return None, f"generation error: {e}"

        reason = reject_reason(question, chunk_text, df, n_docs)
        if reason is None:
            return question, None
        previous = question
    return None, reason


def write_review(rows: list[dict], path: str) -> None:
    """Human review file: worst-leakage questions first, so 20 minutes goes where it counts."""
    rows = sorted(rows, key=lambda r: -r["leak_score"])
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Gold set review\n\n")
        f.write(f"{len(rows)} questions, worst leakage first. ")
        f.write("A high `leak_score` means the question reuses the chunk's own words, so it can be\n")
        f.write("found by string-matching alone — that flatters BM25 and understates the reranker.\n\n")
        f.write("Rewrite anything that reads like a lookup key rather than a student's question,\n")
        f.write("then delete the corresponding line from `goldset.jsonl` (or edit `q` in place).\n\n---\n\n")
        for r in rows:
            f.write(f"## `{r['gold_chunk_id']}` — leak {r['leak_score']:.0%}\n\n")
            f.write(f"**Q:** {r['q']}\n\n")
            snippet = r["chunk_text"][:300].replace("\n", " ")
            f.write(f"> {snippet}...\n\n")


def main():
    parser = argparse.ArgumentParser(description="Generate a gold eval set from an ingested index.")
    parser.add_argument("--user", required=True, help="Username whose index to sample from")
    parser.add_argument("--n", type=int, default=50, help="Number of question/chunk pairs")
    parser.add_argument("--model", default=QUESTION_MODEL, help="Ollama instruct model")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (keep fixed for comparable runs)")
    parser.add_argument("--out", default=GOLDSET_PATH, help="Output .jsonl path")
    args = parser.parse_args()

    _, chunks = load_bm25(args.user)
    eligible = [c for c in chunks if is_answerable(c.page_content)]
    print(f"Index for '{args.user}': {len(chunks)} chunks, {len(eligible)} answerable "
          f"({len(chunks) - len(eligible)} rejected as captions/tables/reference lists).")

    # Leakage is measured against the corpus: a word is only "lifted from the chunk" if it
    # is rare across the corpus. Common topic vocabulary ("derivative") is not leakage.
    df, n_docs = build_corpus_df(chunks)

    picked = sample_chunks(chunks, args.n, seed=args.seed)
    n_math = sum(1 for c in picked if is_mathy(c.page_content))
    print(f"Sampled {len(picked)} chunks ({n_math} math-bearing).")

    llm = ChatOllama(model=args.model, base_url=OLLAMA_BASE_URL, temperature=0.4, num_predict=100)
    print(f"Generating questions with '{args.model}' (rejecting leaks, up to {MAX_ATTEMPTS} attempts each)...\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows, dropped = [], 0
    for i, chunk in enumerate(picked, 1):
        question, reason = generate_question(llm, chunk.page_content, df, n_docs)
        if question is None:
            dropped += 1
            print(f"  [{i}/{len(picked)}] DROPPED — {reason}")
            continue

        rows.append({
            "q": question,
            "gold_chunk_id": chunk_id(chunk),
            "source": os.path.basename(chunk.metadata.get("source", "unknown")),
            "is_math": is_mathy(chunk.page_content),
            "leak_score": round(leak_score(question, chunk.page_content, df, n_docs), 3),
            "model": args.model,
            "prompt_version": PROMPT_VERSION,
            "chunk_text": chunk.page_content,
        })
        print(f"  [{i}/{len(picked)}] leak={rows[-1]['leak_score']:.0%}  {question}")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_review(rows, REVIEW_PATH)

    mean_leak = sum(r["leak_score"] for r in rows) / len(rows) if rows else 0.0
    print(f"\nWrote {len(rows)} pairs to {args.out}  ({dropped} dropped after {MAX_ATTEMPTS} attempts)")
    print(f"Mean leak score: {mean_leak:.2f}  (fraction of DISTINCTIVE question words lifted from the chunk)")
    print(f"\nNow read {REVIEW_PATH} — worst offenders first — and rewrite anything that still")
    print("reads like a lookup key. That pass is what makes the ABSOLUTE numbers trustworthy;")
    print("without it, only the deltas between configs mean anything.")


if __name__ == "__main__":
    main()
