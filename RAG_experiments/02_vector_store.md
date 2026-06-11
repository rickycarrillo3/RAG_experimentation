# Step 2: The Vector Store (ChromaDB)

**Relevant code:** `Chroma(persist_directory="chroma_db", embedding_function=embeddings)`

---

## What is a vector store?

A vector store is a database whose primary job is to answer the question:

> "Given this query vector, which stored vectors are most similar to it?"

This is called **Approximate Nearest Neighbor (ANN)** search, and it's the backbone of semantic search.

---

## What ChromaDB stores

For each chunk, Chroma stores three things together:

| Field | Example |
|---|---|
| `id` | `"abc123"` (auto-generated) |
| `embedding` | `[0.023, -0.14, 0.87, ...]` (384 floats) |
| `document` | `"Employees receive 20 days of paid leave..."` |
| `metadata` | `{"source": "docs/handbook.pdf", "page": 3}` |

When you query, Chroma:
1. Takes your query's embedding vector.
2. Computes **cosine similarity** between your vector and every stored vector.
3. Returns the top-k most similar documents.

---

## Cosine similarity

Cosine similarity measures the angle between two vectors, regardless of their magnitude:

```
similarity = (A · B) / (|A| × |B|)

Range: -1 (opposite) to 1 (identical)
```

Two chunks about the same topic will have vectors pointing in roughly the same direction → high cosine similarity → returned as results.

---

## Why not just use keyword search (like `CTRL+F`)?

| | Keyword search | Vector search |
|---|---|---|
| Matching | Exact word match | Semantic meaning |
| "paid time off" vs "vacation days" | No match | Match (same meaning) |
| Handles synonyms | No | Yes |
| Handles paraphrase | No | Yes |
| Needs index? | Yes (inverted index) | Yes (vector index) |

---

## ChromaDB on disk

After running `ingest.py`, the `chroma_db/` folder contains:

```
chroma_db/
└── chroma.sqlite3   ← all embeddings, documents, metadata stored here
```

It's a standard SQLite file. ChromaDB loads it into memory for fast search when your app starts.

---

## The `k=4` parameter

When you see `search_kwargs={"k": 4}` in the retriever, it means: "return the 4 most similar chunks to my question."

Choosing `k`:
- Too small (k=1): might miss relevant context from different parts of the document.
- Too large (k=10+): fills the prompt with marginally relevant text; increases cost and can confuse the LLM.
- `k=4` is a sensible default for typical use cases.

---

## Read next

- `03_retrieval_and_generation.md` — how the full chain works at query time
