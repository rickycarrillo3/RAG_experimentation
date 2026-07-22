# Step 1: Ingestion

**File:** `knowledge-base/ingest.py`

Ingestion is the offline setup phase. It only needs to run once (or whenever you add new documents). Its job is to transform raw PDFs into a searchable vector index.

---

## Pipeline overview

```
docs/*.pdf
    ↓  PyPDFDirectoryLoader
Raw text (one Document per page, with source metadata)
    ↓  RecursiveCharacterTextSplitter
Smaller chunks (500 chars, 50 char overlap)
    ↓  HuggingFaceEmbeddings (all-MiniLM-L6-v2)
Embedding vectors (384 dimensions each)
    ↓  Chroma.from_documents()
Persisted to chroma_db/
```

---

## Stage 1: Loading

```python
loader = PyPDFDirectoryLoader("docs")
documents = loader.load()
```

`PyPDFDirectoryLoader` scans the `docs/` folder for every `.pdf` file and extracts text page by page using `pypdf`. Each page becomes a LangChain `Document` object:

```python
Document(
    page_content="The actual text on the page...",
    metadata={"source": "docs/my-paper.pdf", "page": 0}
)
```

The `metadata` is important — it's how we later know *which file* an answer came from.

---

## Stage 2: Chunking

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
```

### Why chunk at all?

Embedding models have a **token limit** (all-MiniLM-L6-v2 caps at 256 tokens). More importantly, smaller chunks mean more precise retrieval — you retrieve exactly the paragraph that's relevant, not an entire chapter.

### How RecursiveCharacterTextSplitter works

It tries to split on these separators in order: `\n\n`, `\n`, ` `, `""`. It always tries to split at the largest natural boundary first. For example:
- A long paragraph → split at double newlines between paragraphs
- A very long paragraph → split at single newlines
- Very long line → split at spaces (word boundary)

### What is chunk overlap?

Overlap means the last 50 characters of chunk N also appear at the start of chunk N+1. This prevents an answer from being split exactly at a chunk boundary — context is preserved across the seam.

```
[  chunk 1  ][  overlap  ]
             [  overlap  ][  chunk 2  ]
```

---

## Stage 3: Embedding

```python
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

The embedding model converts each chunk's text into a **384-dimensional vector**. This runs **locally** on your machine — no API call, no cost, no data leaves your computer.

`all-MiniLM-L6-v2` is a popular, small, fast sentence transformer. It's specifically trained to produce vectors where **semantically similar sentences are close together** in vector space.

Example:
```
"What is the company vacation policy?"
→ [0.023, -0.14, 0.87, ...]  (384 numbers)

"Employees receive 20 days of paid leave."
→ [0.031, -0.12, 0.85, ...]  (very similar — high cosine similarity)

"The Python programming language was created by Guido."
→ [-0.45, 0.67, -0.21, ...]  (very different — low cosine similarity)
```

---

## Stage 4: Persisting to ChromaDB

```python
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db",
)
```

This embeds all chunks (calling the model for each one) and saves the resulting vectors to disk in `chroma_db/`. The folder is automatically created.

After this step, `chroma_db/` contains a persistent SQLite database with all embeddings. Future queries just load it — no need to re-embed.

---

## Read next

- `02_vector_store.md` — what ChromaDB stores and how similarity search works
