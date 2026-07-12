# System Deep Dive: Embeddings, Merging, Costs & Latency

---

## 1. Embedding Model Choice: Why bge-small-en-v1.5

### What an embedding model does
It converts a piece of text into a fixed-length vector of numbers (e.g., 384 numbers for bge-small). Two texts that mean similar things produce vectors that are "close" in space. This is what allows semantic search — you embed the query, then find the stored chunks whose vectors are closest.

### Why bge-small specifically

**The alternatives we considered:**

| Model | Dims | Size on disk | CPU query time | Quality (MTEB) |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | ~90MB | ~8ms | 56.3 |
| `bge-small-en-v1.5` | 384 | ~130MB | ~12ms | 62.2 |
| `bge-base-en-v1.5` | 768 | ~430MB | ~35ms | 63.4 |
| `bge-large-en-v1.5` | 1024 | ~1.2GB | ~120ms | 64.2 |
| `AnReu/math_pretrained_bert` | 768 | ~430MB | ~35ms | Unknown |

You're already using `all-MiniLM-L6-v2` in your baseline experiments and `AnReu/math_pretrained_bert` in the hybrid experiment.

**Why bge-small beats MiniLM for this use case:**
- ~6 points higher on retrieval benchmarks (MTEB) for nearly the same speed
- Better at longer, more complex queries (math questions tend to be specific)
- Trained with more recent techniques (BAAI, 2023 vs. sentence-transformers 2021)
- Same vector dimension so no storage cost increase

**Why not bge-base or bge-large:**
- bge-base is 3x slower and 3x larger for a ~1.2 point quality gain — not worth it on CPU
- bge-large is 10x slower and 9x larger for a ~2 point gain — clearly not worth it

**Why not keep AnReu/math_pretrained_bert:**
- It's a BERT model trained on arXiv math papers — sounds perfect, but BERT models were not designed for retrieval. They produce embeddings that cluster poorly in vector space for similarity search. It was trained on masked language modeling (predicting missing words), not on (query, document) pairs. That's a fundamental mismatch.
- bge-small was trained specifically for retrieval using contrastive learning on (query, passage) pairs, which is exactly what we need.
- You'd need a math fine-tune of bge-small to get the best of both worlds — which is exactly what Phase 5 of the plan covers.

**The honest limitation:** bge-small still doesn't understand LaTeX natively. `\frac{d}{dx}[\sin x]` and "the derivative of sine" might not be close in its vector space. This is why Marker matters — converting equations to faithful LaTeX alongside the surrounding plain text gives the embedding model something it can actually work with.

---

## 2. The Merge Step: Reciprocal Rank Fusion (RRF)

### The problem it solves
After hybrid retrieval you have two ranked lists:
- BM25 returns: `[chunk_A (rank 1), chunk_C (rank 2), chunk_F (rank 3), ...]`
- Dense returns: `[chunk_C (rank 1), chunk_B (rank 2), chunk_A (rank 3), ...]`

You need to combine these into one list. The naive approach — add the scores together — fails because BM25 scores (like 12.4, 8.1, 3.2) and cosine similarity scores (like 0.87, 0.83, 0.71) are on completely different scales. Normalizing them requires knowing the score distribution in advance, which you don't.

### How RRF works
RRF ignores the scores entirely. It only looks at **rank position**. For each chunk, its RRF score is:

```
RRF_score(chunk) = Σ  1 / (k + rank_in_list)
```

Where `k = 60` is a constant (smooths out the impact of rank 1 vs rank 2). Sum over all retrieval lists.

**Example with k=60:**

| Chunk | BM25 rank | Dense rank | RRF score |
|---|---|---|---|
| chunk_A | 1 | 3 | 1/(60+1) + 1/(60+3) = 0.0164 + 0.0157 = **0.0321** |
| chunk_C | 2 | 1 | 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = **0.0325** |
| chunk_B | not found | 2 | 0 + 1/(60+2) = **0.0161** |
| chunk_F | 3 | not found | 1/(60+3) + 0 = **0.0157** |

Final ranking: C → A → B → F

**What this achieves:** chunk_C wins because it appeared highly in both lists — that's a strong signal it's relevant. chunk_A is close behind. chunk_B only appeared in dense search, so it ranks lower. chunk_F only appeared in BM25, also lower.

### Why RRF over alternatives
- **Score normalization** (min-max, z-score): Requires knowing the full score distribution. Fragile — one outlier chunk can distort the scale.
- **Weighted linear combination** (0.7 × dense_score + 0.3 × bm25_score): Requires tuning the weights, which differ per dataset. Also requires normalized scores.
- **RRF**: No tuning needed. k=60 works well across essentially all benchmarks. Pure rank-based, so it's immune to score scale differences. Introduced in a 2009 paper and still the standard a decade and a half later because nothing has reliably beaten it on average.

The only real alternative worth considering is **learned fusion** — training a small model to weight the lists. This outperforms RRF on specific domains but requires labeled training data. For your use case, RRF is the right call.

---

## 3. Costs to Run

### Ingestion (one-time per document)

| Step | Cost | Notes |
|---|---|---|
| Marker PDF extraction | GPU time only | ~5–10 min per 300-page textbook on GPU |
| bge-small embedding | CPU time only | ~1–2 min per 300-page textbook on CPU |
| BM25 index build | CPU time only | <5 seconds |
| ChromaDB write | Disk only | Negligible |

**Total ingestion cost per textbook:** ~$0.05–0.15 of cloud GPU time (if renting). After that, you never touch the PDF again.

### Query time (every question)

| Step | Cost | Notes |
|---|---|---|
| Embed the question (bge-small) | ~$0.000 | CPU, ~12ms |
| BM25 search | ~$0.000 | CPU, ~2ms |
| ChromaDB vector search | ~$0.000 | CPU, ~10ms |
| RRF merge | ~$0.000 | CPU, trivial |
| DeepSeek-Math-7B generation | GPU compute | The only real cost |

**Generation cost options:**

| Option | Cost per query | Latency | Notes |
|---|---|---|---|
| RunPod persistent pod (RTX 4090) | ~$0.001/query (amortized) | ~5–10s | You pay $0.44/hr regardless of usage |
| RunPod serverless GPU | ~$0.002–0.005/query | ~8–15s (cold start first) | Pay only when generating |
| Modal serverless | ~$0.002–0.004/query | ~8–12s | Slightly cleaner API than RunPod |
| Apple Silicon local (M2+) | $0.00 | ~15–25s | Free if you have the hardware |
| Groq (free tier) | $0.00 | ~2–3s | DeepSeek-R1 distill available; math quality slightly lower |

**Monthly cost estimate for family use (~50 questions/day):**
- Modal/RunPod serverless: ~$3–7/month
- Apple Silicon local: $0/month
- Persistent pod left running 24/7: ~$316/month (don't do this)

---

## 4. Bottlenecks

### Bottleneck 1: LLM generation time (primary)
The generation model is always the slowest step. A 300-token math answer at 50 tok/sec = 6 seconds minimum. Nothing in the retrieval pipeline comes close to this. All retrieval optimization only saves milliseconds while generation takes seconds.

**Mitigation:** Streaming (show tokens as they arrive so it feels faster), or use a distilled model like DeepSeek-R1-Distill-Qwen-7B which is slightly faster.

### Bottleneck 2: Cold starts (serverless)
When using Modal or RunPod serverless, the first query after idle spins up a new GPU container. This takes 5–15 seconds on top of normal generation time. Subsequent queries in the same session are fast.

**Mitigation:** Keep-warm pinging (send a dummy request every few minutes to keep the container alive). Fine for a family tool.

### Bottleneck 3: Marker ingestion speed
Marker runs its Surya models roughly 1 page/second on a GPU, far slower on CPU. A 400-page calculus textbook = ~7 minutes on GPU, a couple of hours on CPU.

**Mitigation:** Always run Marker on a GPU. This is the one task that genuinely needs it. Rent for one hour, extract all your textbooks, done.

### Bottleneck 4: ChromaDB at scale
ChromaDB is great up to ~100k chunks (a few hundred textbooks). Beyond that, exact nearest neighbor search slows down noticeably.

**Mitigation:** Not your problem for a family tool. You'd need to ingest thousands of textbooks before this matters.

---

## 5. Latency in Regular Use

Here is what happens, in order, when a family member asks a question:

```
User types: "How do I solve x^2 - 5x + 6 = 0?"
                          ↓
[1] Embed the question       ~12ms    (bge-small on CPU)
[2] BM25 search              ~2ms     (pure Python, top-10 chunks)
[3] ChromaDB search          ~15ms    (cosine similarity, top-10 chunks)
[4] RRF merge                ~1ms     (trivial computation)
[5] Format prompt            ~1ms
[6] DeepSeek-Math generates  ~6–10s   (50 tok/sec × 300–500 tokens)
                          ↓
Answer appears
```

**Total perceived latency: ~6–10 seconds** (essentially all of it is step 6)

With streaming enabled, the user sees the first token in ~1–2 seconds and the answer builds progressively — this feels much faster than waiting 8 seconds for a wall of text.

### What can go wrong in practice

| Failure | Cause | Fix |
|---|---|---|
| Slow first answer | Serverless cold start (~10–15s extra) | Keep-warm ping, or accept it |
| Wrong answer despite correct retrieval | DeepSeek-Math hallucinating on complex proofs | Add `"show your work step by step"` in system prompt |
| Retrieved chunks missing the equation | pymupdf4llm extracted garbage LaTeX | Marker fixes this |
| Retrieval returns irrelevant chunks | Query phrasing doesn't match chunk text | HyDE or query expansion (Phase 5) |
| Gradio UI slow to load | Initial model loading into VRAM | Load once at startup, keep in memory |
| Out of context window | Too many chunks passed to LLM | Keep top-5 after RRF, not top-10 |

# Per-user separation
For each user, we do not need extra containers or DBs
ChromaDB supports collections and we can create one per user for personal document uploads