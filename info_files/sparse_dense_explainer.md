# Sparse vs Dense Retrieval: How BM25 Works and How We Merge Them

---

## 1. Why BM25 is "Sparse"

The word "sparse" refers to the vector representation of a document.

When you index a document collection, every method ultimately converts documents into numbers so they can be compared. The question is: what shape do those numbers take?

### The vocabulary space

Imagine your entire document collection uses 50,000 unique words. Every document can be represented as a vector of length 50,000 — one slot per word in the vocabulary.

**Sparse** means: most of those 50,000 slots are zero.

A chunk about quadratic equations might only contain 80 unique words. So 49,920 of the 50,000 slots are zero. Only the slots for words that actually appear in the chunk have non-zero values. That's a sparse vector.

```
vocabulary:  ["a", "algebra", "apple", "banana", "calculus", "derivative", ...]
                                                                  ↑
chunk text:  "the derivative of x^2 is 2x"

vector:      [0,   0,         0,       0,        0,          3.7,          ...]
```

Most values are zero. The vector is sparse.

**Dense** (what bge-small produces) means the opposite: every one of the 384 dimensions has a non-zero value. Every dimension encodes some abstract learned feature of the text's meaning. Nothing is zero.

```
dense vector: [0.23, -0.81, 0.14, 0.67, -0.32, 0.91, ...]  ← all 384 values filled
```

---

## 2. How BM25 Works

BM25 (Best Match 25) is the formula that decides what value to put in each non-zero slot.

It's a refined version of a simple intuition: **a word that appears many times in a document is probably important to that document, but only if it's rare across all documents.**

### The two core ideas

**Term Frequency (TF):** If "derivative" appears 5 times in a chunk, that chunk is probably very relevant to a query about derivatives. More occurrences = more relevant.

**Inverse Document Frequency (IDF):** If "derivative" appears in 2 out of 1000 chunks, it's a meaningful signal. But if "the" appears in 998 out of 1000 chunks, finding "the" in a chunk tells you nothing. IDF penalizes common words and rewards rare ones.

BM25 multiplies these together with two refinements:

### The BM25 formula

```
BM25(chunk, query_word) = IDF(word) × (TF × (k1 + 1)) / (TF + k1 × (1 - b + b × len/avglen))
```

Breaking it down:
- **IDF(word)** = log((N - n + 0.5) / (n + 0.5)) where N = total chunks, n = chunks containing the word. Rare words get high IDF, common words get low IDF.
- **TF** = count of the word in this chunk
- **k1** (~1.2–2.0) = controls how much repeated occurrences keep boosting the score. Without this, a word appearing 100 times would score 100× a word appearing once. k1 dampens that.
- **b** (~0.75) and **len/avglen** = length normalization. A long chunk naturally contains more words, so raw TF would unfairly favor long chunks. This divides by chunk length relative to average.

### A concrete example

Query: `"quadratic formula"`
Chunks:
- Chunk A: "The quadratic formula is x = (-b ± √(b²-4ac)) / 2a. It solves quadratic equations."
- Chunk B: "quadratic quadratic quadratic quadratic quadratic" (artificially repeated)
- Chunk C: "Linear equations have one solution. The solution depends on the coefficients."

Without BM25 (raw TF):
- Chunk B wins because "quadratic" appears 5 times

With BM25:
- Chunk A wins — "quadratic" appears twice AND "formula" appears once, IDF rewards both, length normalization is fair
- Chunk B is penalized — the k1 dampening stops pure repetition from dominating
- Chunk C scores low — neither "quadratic" nor "formula" appears

**This is why BM25 is good at exact match retrieval.** It is built around the presence and frequency of the exact query words.

### What BM25 cannot do

If the query is `"find the roots of a polynomial"` and the chunk says `"solve for x in a quadratic equation"`, BM25 scores this near zero — "roots", "polynomial" don't appear in the chunk. It has no understanding that these mean the same thing.

That's the job of dense retrieval.

---

## 3. How Dense Retrieval Works (Brief Recap)

`bge-small` passes text through a neural network and produces a 384-dimensional vector where **meaning** is encoded geometrically. "Find the roots of a polynomial" and "solve for x in a quadratic equation" end up close in this 384-dimensional space because the model was trained on millions of (query, relevant-passage) pairs and learned to put semantically similar texts near each other.

The cost: it doesn't care about exact words. If the query is `"quadratic formula"` and the chunk literally says `"quadratic formula"` 10 times, dense retrieval might rank it lower than a chunk that discusses the concept more broadly. BM25 would never make that mistake.

---

## 4. How We Merge Them (RRF in Detail)

You now have two ranked lists from two different retrieval methods. The goal is to produce one final ranked list that captures the strengths of both.

### Why you can't just add scores

BM25 scores look like: `12.4, 8.1, 3.2, 1.1, ...`
Dense cosine similarity scores look like: `0.87, 0.83, 0.71, 0.68, ...`

These scales are completely different. Adding them directly (12.4 + 0.87 = 13.27) is meaningless — BM25 would dominate simply because its numbers are larger, not because it's more relevant.

You could normalize them (min-max scaling to 0–1) but this introduces a new problem: a single outlier chunk with an unusually high BM25 score compresses all other scores toward zero, distorting the ranking.

### RRF: rank-based fusion

RRF throws away the scores entirely and only uses **position in the ranked list**.

For each chunk, compute:

```
RRF_score = Σ  1 / (k + rank)
```

Sum over every retrieval list the chunk appears in. k=60 is a constant that prevents rank 1 from having an outsized advantage over rank 2.

### Step-by-step example

Query: `"how to solve a quadratic equation"`

**BM25 results** (exact keyword match):
1. Chunk A — "quadratic equation has two solutions..."
2. Chunk C — "solving quadratic equations by factoring..."
3. Chunk E — "the quadratic formula gives..."

**Dense results** (semantic match):
1. Chunk C — semantically closest to the query
2. Chunk B — "finding roots of second-degree polynomials..."
3. Chunk A — also relevant

**RRF calculation (k=60):**

| Chunk | BM25 rank | Dense rank | RRF score |
|---|---|---|---|
| Chunk A | 1 | 3 | 1/61 + 1/63 = 0.0164 + 0.0159 = **0.0323** |
| Chunk C | 2 | 1 | 1/62 + 1/61 = 0.0161 + 0.0164 = **0.0325** |
| Chunk E | 3 | not found | 1/63 + 0 = **0.0159** |
| Chunk B | not found | 2 | 0 + 1/62 = **0.0161** |

**Final ranking:** C → A → B → E

Chunk C wins because it ranked highly in both lists — strong signal from two independent sources. Chunk A is close behind for the same reason. Chunk B (only dense) and Chunk E (only BM25) score lower.

### The code

```python
from collections import defaultdict

def reciprocal_rank_fusion(ranked_lists: list[list], k: int = 60) -> list:
    scores = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = doc.page_content[:100]  # use content as key
            scores[doc_id] += 1 / (k + rank)

    # Sort by RRF score descending
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

    # Map back to documents
    all_docs = {doc.page_content[:100]: doc for lst in ranked_lists for doc in lst}
    return [all_docs[doc_id] for doc_id in sorted_ids if doc_id in all_docs]


# Usage
bm25_results = bm25_search(query, top_k=10)     # list of Documents
dense_results = chroma_store.similarity_search(query, k=10)  # list of Documents

merged = reciprocal_rank_fusion([bm25_results, dense_results])
top_5 = merged[:5]  # pass these to the LLM
```

---

## 5. Summary: Why Each Method Exists

| | BM25 (sparse) | bge-small (dense) | Together (hybrid) |
|---|---|---|---|
| Representation | Sparse vector (50k dims, mostly zeros) | Dense vector (384 dims, all filled) | Both |
| Finds | Exact keyword matches | Semantic / conceptual matches | Both |
| Misses | Synonyms, paraphrases | Exact rare terms | Very little |
| Speed | ~2ms | ~12ms | ~14ms |
| Cost | Free, no model | Small model, CPU | Both cheap |
| Math example wins | `"quadratic formula"` → chunks containing exactly those words | `"find roots of polynomial"` → chunks about solving equations | Either phrasing retrieves the right chunk |
