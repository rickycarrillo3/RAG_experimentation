# Step 3: Retrieval and Generation

**Files:** `knowledge-base/query.py`, `knowledge-base/app.py`

This is the online phase — what happens each time a user asks a question.

---

## The full flow

```
User: "How many vacation days do employees get?"
         ↓
   1. Embed the question
         ↓  (same all-MiniLM-L6-v2 model)
   question_vector = [0.031, -0.12, 0.85, ...]
         ↓
   2. Search ChromaDB for top-4 similar chunks
         ↓
   retrieved_chunks = [
     "Employees receive 20 days of paid leave...",
     "PTO accrues at 1.67 days per month...",
     "Unused vacation may be carried over...",
     "New hires are eligible after 90 days...",
   ]
         ↓
   3. Build the prompt (inject chunks as context)
         ↓
   prompt = """
   Context:
   Employees receive 20 days...
   PTO accrues at 1.67 days...
   ...

   Question: How many vacation days do employees get?
   Answer:
   """
         ↓
   4. Send to Claude
         ↓
   Claude: "Employees receive 20 days of paid leave per year,
            accruing at 1.67 days per month. Source: handbook.pdf"
```

---

## The LangChain components

### `HuggingFaceEmbeddings`
Loads the same embedding model used during ingestion. It must be **identical** — embeddings are only comparable if they come from the same model. Using a different model at query time would be like searching a French dictionary with an English query.

### `Chroma` (load mode)
```python
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
```
This loads the existing database from disk. No re-embedding happens here.

### `ChatAnthropic`
```python
llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0, max_tokens=1024)
```
- `temperature=0`: deterministic, factual answers (no creativity/randomness)
- `max_tokens=1024`: cap on response length

### `RetrievalQA`
```python
chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)
```

This is LangChain's pre-built chain that wires everything together:

```
chain.invoke({"query": "..."})
    → retriever.get_relevant_documents(query)   # searches ChromaDB
    → format prompt with context + question
    → llm.invoke(prompt)                        # calls Claude
    → return {"result": "...", "source_documents": [...]}
```

`return_source_documents=True` means the chain also returns the raw chunks it used, so you can display the source filenames to the user.

---

## chain_type: "stuff"

`chain_type="stuff"` means all retrieved chunks are *stuffed* directly into one prompt. This is the simplest strategy and works well when:
- Your chunks are short (500 chars)
- You're retrieving few chunks (k=4)
- Total prompt fits in the context window

Alternative chain types (not used here):
- `map_reduce`: summarize each chunk individually, then combine
- `refine`: iteratively refine an answer chunk by chunk
- `map_rerank`: score each chunk separately, pick the best

---

## Read next

- `04_the_prompt.md` — why the prompt template shapes everything
