# What is RAG?

## The core problem RAG solves

Large language models like Claude are trained on a huge corpus of text — but that training has a **cutoff date**, and more importantly, it **doesn't include your private documents**.

If you ask Claude "What does our internal policy say about remote work?", it can't answer — it has never seen your policy document.

**RAG (Retrieval-Augmented Generation)** solves this by:

1. Storing your documents in a searchable database.
2. At query time, finding the most relevant passages from those documents.
3. Injecting those passages into the prompt that Claude receives.
4. Claude answers based on what it was just given — not from memory.

---

## The two phases

```
OFFLINE (run once)           ONLINE (run per query)
─────────────────────        ──────────────────────
Load documents               User asks a question
      ↓                             ↓
Chunk text                   Embed the question
      ↓                             ↓
Embed chunks                 Search vector DB → top-k chunks
      ↓                             ↓
Store in vector DB           Build prompt: question + chunks
                                     ↓
                             LLM generates answer
```

---

## Why not just put the whole document in the prompt?

You could — for a small document. But:

- LLMs have a **context window limit** (e.g., 200k tokens for Claude). Large document sets won't fit.
- Flooding the prompt with irrelevant text degrades answer quality.
- Vector search retrieves only the **most relevant** passages, keeping prompts focused.

---

## The three key concepts

### 1. Embeddings
A way to turn text into a list of numbers (a vector) that captures semantic meaning. Similar text produces similar vectors. This is how we do "meaning-based" search rather than keyword matching.

### 2. Vector Store
A database optimized for storing and searching vectors by similarity (using cosine distance or dot product). In this project we use **ChromaDB**.

### 3. Retrieval chain
The orchestration layer that ties everything together: take a question → embed it → search the store → pack results into a prompt → send to the LLM. We use **LangChain** for this.

---

## Read next

- `01_ingestion.md` — how documents are loaded, chunked, and embedded
- `02_vector_store.md` — what ChromaDB does under the hood
- `03_retrieval_and_generation.md` — how the RAG chain works at query time
- `04_the_prompt.md` — why the prompt template matters
