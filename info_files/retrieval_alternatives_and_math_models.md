# Retrieval Alternatives & What Makes a Model Good at Math

---

## Part 1: Faster, Cheaper Retrieval Alternatives

The slowness and cost in standard RAG retrieval usually comes from two things:
1. **Embedding inference at query time** — running a neural network every time someone asks a question (geenrating query embedding)
2. **Large vector index search** — scanning many high-dimensional vectors (comparing query embedding to vector db)

Here are the main alternatives, ordered from cheapest/fastest to more capable.

---

### Option A: BM25 (Keyword Search)

**What it is:** A classical text ranking algorithm (the backbone of Elasticsearch and most search engines). No neural network involved — it scores documents based on term frequency and document length normalization.

**Speed:** Near-instant. Runs on CPU. No model inference.
**Cost:** Free. No GPU, no embedding API calls.
**Quality:** Surprisingly good for math. Math queries often contain exact terms (`derivative`, `quadratic`, `integral`) that BM25 handles perfectly. It struggles with paraphrasing but math language is actually fairly standardized.

**When it fails:** `"find the roots"` won't match a chunk that says `"solve for x"` — no semantic understanding.

**How to use it:**
```python
from rank_bm25 import BM25Okapi

corpus = [chunk.split() for chunk in your_chunks]
bm25 = BM25Okapi(corpus)
scores = bm25.get_scores(query.split())
```

**Best for:** Fast prototyping, low-resource environments, or as one half of a hybrid system.

---

### Option B: TF-IDF + Cosine Similarity

**What it is:** A classic sparse vector representation. Each document becomes a vector of term weights (term frequency × inverse document frequency). Similarity is measured by cosine distance between sparse vectors.

**Speed:** Very fast. All CPU, no neural inference.
**Cost:** Free.
**Quality:** Similar to BM25, slightly weaker. Less tuned for ranking than BM25.

**When to use it over BM25:** Rarely — BM25 is almost always better. TF-IDF is simpler to explain and implement from scratch, which matters if you're learning the fundamentals.

---

### Option C: Sparse Neural Embeddings (SPLADE)

**What it is:** A learned sparse representation model. Unlike dense embeddings (768-dim float vectors), SPLADE produces sparse vectors — most dimensions are zero, only a few hundred are non-zero. This makes them fast to store and search, like keyword search, but with semantic understanding.

**Speed:** Fast at search time (sparse math is cheap). Slightly slower than BM25 at index time (still need a model pass once).
**Cost:** One-time inference per chunk at index time. Zero cost at query time if you pre-compute query expansions.
**Quality:** Better than BM25 for semantic queries. Competitive with dense retrieval on many benchmarks.

**Best for:** When you want semantic understanding but can't afford dense retrieval latency.

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

# naver/splade-cocondenser-ensembledistil is a good free starting point
tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")
```

---

### Option D: Smaller / Quantized Dense Embedding Models

If you want to keep dense (semantic) retrieval but reduce cost and latency, the answer is usually **use a smaller embedding model**.

| Model | Dimensions | Speed | Quality |
|---|---|---|---|
| `bge-large-en-v1.5` | 1024 | Slow | Highest |
| `bge-base-en-v1.5` | 768 | Medium | Good |
| `bge-small-en-v1.5` | 384 | Fast | Good enough |
| `all-MiniLM-L6-v2` | 384 | Very fast | Decent |
| `nomic-embed-text` | 768 | Fast | Surprisingly strong |

`bge-small` on CPU runs in ~5–15ms per query — fast enough for a family-use system. Going from `bge-large` to `bge-small` is often the single biggest latency win with minimal quality loss.

---

### Option E: Approximate Nearest Neighbor (ANN) Search

**What it is:** Standard vector search does an exact comparison against every vector in the index. ANN search trades a small amount of accuracy for massive speed gains by using index structures like HNSW (Hierarchical Navigable Small World graphs) or IVF (Inverted File Index).

**Speed:** 10–100x faster than exact search on large corpora.
**Cost:** Free. Built into every major vector DB.
**Quality loss:** Minimal — typically <1% recall loss with proper tuning.

Most vector DBs already use ANN by default:
- **FAISS** (Facebook) — best for local use, highly configurable
- **Qdrant** — good local + cloud option, HNSW-based
- **Chroma** — easiest to set up locally, good for prototyping

For a family-use system with a few thousand math documents, this probably isn't the bottleneck — but it matters if you scale up to full textbooks.

---

### Option F: Hybrid BM25 + Small Dense Model (Recommended)

The best balance for your use case: run BM25 to get fast keyword candidates, run a small dense embedding model for semantic coverage, merge with Reciprocal Rank Fusion (RRF).

```
Query
  ├─ BM25 → top 20 chunks (fast, free)
  ├─ bge-small embedding → top 20 chunks (fast, small model)
  └─ RRF merge → top 5 chunks → LLM
```

**Why this wins for math:** Math queries often have both exact terms (BM25 excels) and semantic meaning (dense excels). Hybrid consistently outperforms either alone. The small dense model keeps latency low.

**Total query latency estimate on CPU:** 50–150ms. Perfectly acceptable for a home system.

---

### Option G: Pre-filtering with Metadata (Orthogonal Speedup)

Before any vector search, filter your index down using metadata. If your documents are tagged by topic (`algebra`, `calculus`, `geometry`, `statistics`), a query about derivatives only needs to search the `calculus` subset.

This is free, requires no model changes, and reduces effective index size — directly reducing search time. Combined with any of the above, it compounds the speedup.

---

### Retrieval Cost/Speed Summary

| Method | Query Latency | GPU Needed | Semantic Understanding | Best For |
|---|---|---|---|---|
| BM25 | <5ms | No | No | Fast baseline, exact terms |
| TF-IDF | <5ms | No | No | Learning/simple setup |
| SPLADE | <10ms | No (inference done at index time) | Yes | Best sparse option |
| Small dense (bge-small) | ~15ms | No (CPU OK) | Yes | Good semantic retrieval |
| Hybrid BM25 + bge-small | ~20ms | No | Yes | **Recommended for your system** |
| Large dense (bge-large) | ~100ms+ | Preferred | Yes | Highest quality, high cost |

---

## Part 2: What Makes a Model Better at Math?

This is a nuanced question. Math performance comes from several compounding factors, not a single thing.

---

### 1. Pre-training Data Composition

The most fundamental factor. Models that are exposed to more math-dense text during pre-training develop stronger internal representations of mathematical reasoning.

**What "math-dense" data looks like:**
- ArXiv papers (mathematics, physics, CS theory)
- Stack Exchange Mathematics (problem + answer threads)
- Project Gutenberg math textbooks
- Competition math (AoPS, Art of Problem Solving forums)
- Code (Python, Sympy, Mathematica) — more on this below

**DeepSeek-Math** was pre-trained on a corpus that included 120B tokens of math-specific data scraped from the web and filtered for quality. This is why it punches well above its weight class — it's not just bigger, it was fed more math.

Compare this to a general model like Llama 3.1 8B, which was trained on a broad internet corpus where math is a small fraction. It knows math but didn't marinate in it.

---

### 2. Code Training Has a Surprising Math Effect

This is one of the most well-documented findings in LLM research: **training on code improves mathematical reasoning**, even for problems that don't involve code.

Why? Code forces the model to learn:
- Precise, unambiguous symbolic manipulation
- Multi-step sequential logic (if A then B, given C...)
- Variable binding and scope (what does x refer to here?)
- Exact syntax where small errors change meaning entirely

These are all core skills in math. Models like Phi-3, Code Llama, and DeepSeek-Coder consistently outperform same-sized non-code models on math benchmarks because of this transfer.

---

### 3. Chain-of-Thought in Fine-Tuning Data

A model trained on `"answer: 42"` learns to output answers. A model trained on `"step 1: ... step 2: ... therefore 42"` learns to **reason**.

Chain-of-thought (CoT) fine-tuning data teaches the model to use its own output as a working scratchpad. This is critical for multi-step math — a model that jumps to an answer is far more likely to be wrong than one that works through it.

The GSM8K dataset (grade school math with full reasoning traces) and MATH dataset (competition math with solutions) are the two most important CoT training sources. DeepSeek-Math, Phi-3, and similar strong math models all trained heavily on these.

---

### 4. Tokenization of Numbers and Symbols

This is an underappreciated technical issue. GPT-2-era tokenizers split numbers in unintuitive ways:
- `1234567` might tokenize as `["123", "4567"]` or `["1", "234", "567"]`
- This makes arithmetic unreliable — the model doesn't "see" the number as a unit

Newer models (LLaMA 3, Mistral, DeepSeek) use tokenizers that handle numbers more carefully, tokenizing digit-by-digit or in consistent groupings. This directly improves arithmetic accuracy.

LaTeX is a similar issue — `\frac{1}{2}` needs to be tokenized in a way that preserves its mathematical structure. Models trained on math-heavy data tend to have better tokenizer coverage of math symbols.

---

### 5. Reinforcement Learning on Math Verifiability

Math is special among reasoning tasks because **answers are verifiable**. You can check if `x = 3` is correct without a human judge — just substitute it back in.

Some models (like DeepSeek-Math and the newer reasoning models like o1/R1) use RL where the reward signal is whether the final answer is mathematically correct. This is called **outcome-based RL** or **process reward modeling**.

This produces models that are much more reliable on complex multi-step problems, because they've been optimized directly for getting right answers, not just producing fluent-sounding math text.

DeepSeek-R1 (the reasoning variant) uses this heavily and is why it's dramatically better at competition-level math than DeepSeek-Math alone.

---

### 6. Model Size vs. Specialization Trade-off

Bigger models are generally better at math, but specialization can overcome the size gap:

| Model | Size | Math Benchmark (MATH) | Notes |
|---|---|---|---|
| GPT-4 | ~1T (est.) | ~87% | Massive, closed, costly |
| DeepSeek-R1 | 671B (MoE) | ~97% | State of the art, open weights |
| DeepSeek-Math-7B | 7B | ~35% | Small, highly specialized |
| Mistral-7B-Instruct | 7B | ~13% | General, not math-focused |
| Phi-3-mini (3.8B) | 3.8B | ~27% | Punches above weight |
| LLaMA 3.1 8B | 8B | ~20% | General purpose |

DeepSeek-Math-7B at ~35% versus Mistral-7B at ~13% — same size, ~3x better at math purely because of training data and fine-tuning choices. This is why it's the right pick for your system.

---

### Summary: Why DeepSeek-Math-7B is the Right Call

1. **120B tokens of math pre-training** — it has genuinely seen far more math than general models
2. **CoT fine-tuning** — trained to show its work step by step
3. **Free and open weights** — runs locally via Ollama, zero API cost
4. **7B is manageable** — runs on 8GB of RAM with 4-bit quantization
5. **Instruction-tuned variant** — the `-instruct` version is already aligned for Q&A use

For your family system, pair it with hybrid BM25 + `bge-small` retrieval and you get a fast, free, math-specialized pipeline.
