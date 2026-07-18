"""
make_evalset.py - Build the gold set: question -> the chunk that should be retrieved.

Samples chunks from a user's index (biased toward math-bearing ones), asks an instruct
model to write the question a student would ask that each chunk answers, and writes
eval/goldset.jsonl. The chunk it was generated from is the gold label.

    python make_evalset.py --user test --n 50
    python make_evalset.py --user test --n 50 --model qwen2:7b

IMPORTANT: the output is a DRAFT. Read eval/goldset.jsonl and delete or rewrite the
questions that merely parrot the chunk — those inflate every score, BM25's most of all.
See EVALUATION.md for why this step is not optional.
"""

import argparse
import json
import os
import random
import re

from langchain_ollama import ChatOllama

from retrieval import chunk_id, load_bm25

EVAL_DIR = "eval"
GOLDSET_PATH = os.path.join(EVAL_DIR, "goldset.jsonl")
QUESTION_MODEL = "qwen2:7b"  # an *instruct* model; deepseek-math is a solver, not a writer

# Signals that a chunk actually carries math, not just prose about math.
MATH_SIGNALS = [
    "$$", "\\frac", "\\int", "\\sum", "\\sqrt", "\\begin{", "\\lim",
    "\\partial", "\\nabla", "\\alpha", "\\beta", "\\theta", "\\mathbb",
]

PROMPT = """You are helping build a benchmark for a math tutoring search engine.

Below is a passage from a math textbook. Write ONE question that a student would
naturally ask, which this passage answers.

Rules:
- Ask it the way a student would, in their own words.
- Do NOT copy phrases or notation verbatim from the passage — a good question tests
  whether a search engine can *find* this passage, not whether it can string-match it.
- Do NOT refer to "the passage", "the text", or "the document".
- Ask about the specific concept or result, not something generic.
- Output ONLY the question. No preamble, no quotes, no numbering.

Passage:
---
{chunk}
---

Question:"""


def is_mathy(text: str) -> bool:
    return any(sig in text for sig in MATH_SIGNALS)


def sample_chunks(chunks: list, n: int, seed: int = 0) -> list:
    """Sample n chunks, preferring math-bearing ones and skipping tiny fragments.

    Biasing toward math is deliberate: chunks of pure prose would let the eval look
    healthy while saying nothing about the case the system exists to handle.
    """
    rng = random.Random(seed)
    usable = [c for c in chunks if len(c.page_content.strip()) >= 200]
    mathy = [c for c in usable if is_mathy(c.page_content)]
    prose = [c for c in usable if not is_mathy(c.page_content)]

    rng.shuffle(mathy)
    rng.shuffle(prose)

    # Aim for ~70% math-bearing, backfilling from prose if there isn't enough math.
    want_math = min(len(mathy), int(n * 0.7))
    picked = mathy[:want_math] + prose[: n - want_math]
    if len(picked) < n:
        picked += mathy[want_math : want_math + (n - len(picked))]
    rng.shuffle(picked)
    return picked[:n]


def clean_question(raw: str) -> str:
    q = raw.strip().split("\n")[0].strip()
    q = re.sub(r'^(question|q)\s*[:.\-]\s*', '', q, flags=re.I)
    return q.strip().strip('"').strip()


def main():
    parser = argparse.ArgumentParser(description="Generate a gold eval set from an ingested index.")
    parser.add_argument("--user", required=True, help="Username whose index to sample from")
    parser.add_argument("--n", type=int, default=50, help="Number of question/chunk pairs")
    parser.add_argument("--model", default=QUESTION_MODEL, help="Ollama instruct model for question generation")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (keep fixed for comparable runs)")
    parser.add_argument("--out", default=GOLDSET_PATH, help="Output .jsonl path")
    args = parser.parse_args()

    _, chunks = load_bm25(args.user)
    print(f"Index for '{args.user}' has {len(chunks)} chunks.")

    picked = sample_chunks(chunks, args.n, seed=args.seed)
    n_math = sum(1 for c in picked if is_mathy(c.page_content))
    print(f"Sampled {len(picked)} chunks ({n_math} math-bearing).")

    llm = ChatOllama(model=args.model, temperature=0.3, num_predict=100)
    print(f"Generating questions with '{args.model}'...\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(picked, 1):
            try:
                raw = llm.invoke(PROMPT.format(chunk=chunk.page_content)).content
            except Exception as e:
                print(f"  [{i}/{len(picked)}] generation failed: {e}")
                continue

            question = clean_question(raw)
            if len(question) < 15 or "?" not in question:
                print(f"  [{i}/{len(picked)}] skipped (not a question): {question[:60]!r}")
                continue

            f.write(json.dumps({
                "q": question,
                "gold_chunk_id": chunk_id(chunk),
                "source": os.path.basename(chunk.metadata.get("source", "unknown")),
                "is_math": is_mathy(chunk.page_content),
                "chunk_text": chunk.page_content,
            }, ensure_ascii=False) + "\n")
            written += 1
            print(f"  [{i}/{len(picked)}] {question}")

    print(f"\nWrote {written} pairs to {args.out}")
    print("\nNEXT STEP, and it matters: open the file and read the questions. Delete or")
    print("rewrite any that quote the chunk verbatim — they flatter BM25 and make the")
    print("whole eval optimistic. See EVALUATION.md.")


if __name__ == "__main__":
    main()
