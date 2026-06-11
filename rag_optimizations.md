# RAG Optimizations: A Thorough Overview

RAG (Retrieval-Augmented Generation) pipelines have several stages where you can optimize: **indexing**, **retrieval**, and **generation**. Here's a breakdown of the most impactful techniques.

---

## 1. Chunking Strategies

The first decision is how to split your documents before embedding them.

**Fixed-size chunking** splits text every N tokens with some overlap. Simple, but ignores semantic boundaries — a chunk might cut a sentence mid-thought.

**Semantic chunking** uses embedding similarity to find natural breakpoints. You split where the meaning "shifts," keeping coherent ideas together. More expensive but produces better retrieval units.

**Hierarchical chunking** (e.g., "parent-child" chunks) stores both a small chunk (precise) and its parent paragraph or section (context-rich). You retrieve the small chunk for relevance scoring, then pass the parent to the LLM for context. This is also called **"small-to-big retrieval."**

**Document-aware chunking** respects structure: splits on headings, paragraphs, or code blocks rather than raw token counts. Critical for PDFs, markdown, and structured docs.

---

## 2. Embedding Model Choice & Fine-Tuning

The embedding model determines how well semantic similarity maps to actual relevance.

**General-purpose models** (e.g., `text-embedding-3-large`, `bge-large`) work well out of the box. Larger models generally produce better embeddings at the cost of speed/cost.

**Domain fine-tuning** trains the embedding model on your specific corpus and query/answer pairs. If your documents are highly technical (legal, medical, code), a fine-tuned model can dramatically outperform a general one because it learns what "similar" means in your domain.

**Matryoshka embeddings** (e.g., `text-embedding-3`) let you truncate the embedding vector to a smaller dimension without retraining. You can store smaller vectors (cheaper) and scale up dimension only when needed.

---

## 3. Hybrid Search

Pure vector (semantic) search can miss exact keyword matches. Pure BM25 keyword search misses synonyms and paraphrases. **Hybrid search** combines both.

- **BM25 + vector search** retrieves candidates from both, then merges results.
- **Reciprocal Rank Fusion (RRF)** is the standard merging strategy: it combines ranked lists from multiple retrievers without needing normalized scores.

This is one of the highest-ROI improvements — practically free to add if your vector database supports it (most do: Elasticsearch, Weaviate, Qdrant, etc.).

---

## 4. Re-Ranking

After retrieval, you have N chunks (e.g., top 20). A **re-ranker** (also called a cross-encoder) scores each `(query, chunk)` pair more accurately than the embedding similarity alone.

- **Cross-encoders** (e.g., Cohere Rerank, `bge-reranker`) take both query and document as input simultaneously — much more accurate than comparing separate embeddings, but too slow to run over the entire corpus.
- The typical pipeline: **bi-encoder** (fast, retrieves top-50) → **cross-encoder** (accurate, re-ranks to top-5).

Re-ranking consistently improves answer quality because embedding similarity is a proxy for relevance, not relevance itself.

---

## 5. Query Transformation

Raw user queries are often too short, ambiguous, or differently phrased than the documents. Several techniques fix this:

**Query expansion** rewrites or augments the query before retrieval. Example: "What's the refund policy?" becomes multiple queries like "refund policy," "return policy," "money back guarantee."

**HyDE (Hypothetical Document Embeddings)** — instead of embedding the query, you ask the LLM to *generate a hypothetical answer*, then embed that. The generated text looks more like a real document chunk, so it matches better in the vector space.

**Step-back prompting** asks the LLM to first abstract the query to a higher-level question, retrieve for both, and combine. Useful when the specific question requires understanding a broader concept.

**Multi-query retrieval** generates several paraphrases of the query, retrieves for each independently, and deduplicates. Catches chunks that match some phrasings but not others.

---

## 6. Contextual Retrieval

Proposed by Anthropic — before indexing, prepend each chunk with a short LLM-generated summary of where that chunk sits in the document (e.g., "This chunk is from Chapter 3 of the Financial Report, discussing Q3 revenue declines in APAC"). This gives the embedding model crucial context it wouldn't have from the raw chunk alone.

Cost: one LLM call per chunk at index time. Benefit: significantly better retrieval, especially for chunks that are unintelligible without surrounding context (e.g., "As noted above, the rate is 5%").

---

## 7. Metadata Filtering

Rather than searching everything, attach structured metadata to chunks (date, author, document type, department, product version) and pre-filter before vector search.

Example: "What were our Q4 2024 sales?" → filter `year=2024, quarter=Q4` first, then run vector search only over those chunks. Dramatically reduces noise in results and speeds up retrieval.

---

## 8. Late Interaction Models (ColBERT)

Standard embeddings compress a document into a single vector. **ColBERT** keeps per-token embeddings and computes relevance as the sum of max-similarities between query tokens and document tokens.

This is more accurate than single-vector retrieval (closer to cross-encoder quality) but faster than cross-encoders. The tradeoff: much higher storage cost (one vector per token vs. one per chunk).

---

## 9. Agentic / Iterative Retrieval

Instead of a single retrieval call, let the LLM decide when and what to retrieve.

**Retrieval-augmented reasoning** loops: the LLM generates a partial answer, decides it needs more info, issues a new query, retrieves, and continues. Used in frameworks like LangGraph, ReAct, and similar agentic setups.

**FLARE (Forward-Looking Active Retrieval)** generates text token-by-token and retrieves new context whenever the model's confidence drops below a threshold.

These are more expensive (multiple LLM + retrieval calls) but handle multi-hop questions that a single retrieval pass can't answer.

---

## 10. Context Window Management

Even after good retrieval, how you arrange chunks in the prompt matters.

**Lost in the middle** is a known LLM failure mode: models attend best to text at the beginning and end of the context window, ignoring the middle. Put your most relevant chunks first and last.

**Context compression** (e.g., LLMLingua) uses a small, fast model to prune irrelevant sentences from retrieved chunks before passing them to the expensive generation model. Lets you fit more information into the context window.

**Maximal Marginal Relevance (MMR)** selects chunks that are relevant *and* diverse, avoiding sending 5 near-identical chunks that waste context window space.

---

## Quick Reference: Where Each Optimization Fits

| Stage | Optimization | Effort | Impact |
|---|---|---|---|
| Indexing | Semantic/hierarchical chunking | Medium | High |
| Indexing | Contextual retrieval | Medium | High |
| Indexing | Metadata tagging | Low | High |
| Retrieval | Hybrid search (BM25 + vector) | Low | High |
| Retrieval | Re-ranking | Low | High |
| Retrieval | Query transformation / HyDE | Medium | Medium-High |
| Retrieval | Metadata filtering | Low | High |
| Generation | Lost-in-middle ordering | Low | Medium |
| Generation | Context compression | Medium | Medium |
| Generation | Agentic retrieval | High | High (complex Q) |

---

The highest-ROI starting points for most projects are **hybrid search**, **re-ranking**, and **better chunking** — they require minimal infrastructure changes but consistently improve answer quality.
