# Step 4: The Prompt Template

**Relevant code:** `query.py` and `app.py` — `PROMPT_TEMPLATE` constant

The prompt template is the instruction layer that tells Claude *how* to use the retrieved context. It's arguably the most important tuning knob in a RAG system.

---

## The template used in this project

```python
PROMPT_TEMPLATE = """You are a helpful assistant that answers questions strictly from the provided context.

Rules:
- Answer only based on the context below.
- If the answer is not present in the context, say exactly: "I don't have that in my knowledge base."
- Always cite the source document filename(s) at the end of your answer.

Context:
{context}

Question: {question}

Answer:"""
```

---

## Breaking it down

### Role instruction
```
You are a helpful assistant that answers questions strictly from the provided context.
```
This sets Claude's behavior mode upfront. "Strictly from the provided context" discourages Claude from drawing on its training knowledge — you want answers grounded in *your* documents, not general knowledge.

### Grounding rules
```
- Answer only based on the context below.
- If the answer is not present in the context, say exactly: "I don't have that in my knowledge base."
```
This is the **hallucination guard**. Without it, Claude might confidently invent an answer. By giving it an explicit escape hatch ("say this phrase if you don't know"), you make it safe to trust the "I don't know" signal.

### Citation rule
```
- Always cite the source document filename(s) at the end of your answer.
```
This increases trust and lets users verify claims by going back to the source PDF.

---

## The template variables

LangChain's `PromptTemplate` fills in two variables at runtime:

| Variable | Filled by | Content |
|---|---|---|
| `{context}` | LangChain | The 4 retrieved chunks concatenated together |
| `{question}` | The user | The user's query |

---

## What the filled prompt looks like

When a user asks "What is the refund policy?", Claude actually receives something like:

```
You are a helpful assistant...

Context:
[chunk 1 text — from policy.pdf, page 2]
[chunk 2 text — from policy.pdf, page 3]
[chunk 3 text — from terms.pdf, page 1]
[chunk 4 text — from faq.pdf, page 5]

Question: What is the refund policy?

Answer:
```

Claude never "remembers" your documents — it just reads them fresh every time, like an open-book exam.

---

## Why the prompt matters so much

The same retrieved chunks can produce very different answers depending on the prompt:

| Prompt instruction | Claude behavior |
|---|---|
| No instruction | May mix retrieved context with general knowledge |
| "Only use the context" | Sticks to your documents |
| "If unsure, say so" | Refuses to hallucinate |
| "Cite sources" | Adds source attribution |
| "Answer in bullet points" | Changes formatting |

Prompt engineering is iterative — if you're getting bad answers, the retrieval and the prompt are the two main places to tune.

---

## Prompt engineering tips for RAG

1. **Be explicit about the grounding constraint.** "Only use the context" > "Use the context."
2. **Give a specific fallback phrase.** Makes it easy to detect "I don't know" answers programmatically.
3. **Ask for citations.** Increases reliability and user trust.
4. **Set the tone.** "Concise and factual" vs. "detailed and thorough" both work — depends on your use case.
5. **Don't over-constrain.** Overly rigid rules can cause Claude to refuse valid questions.

---

## Read next

You now understand the full RAG pipeline end-to-end:

```
00_what_is_rag.md       ← the big picture
01_ingestion.md         ← loading, chunking, embedding
02_vector_store.md      ← ChromaDB and similarity search
03_retrieval_and_generation.md  ← the query-time chain
04_the_prompt.md        ← this file
```

To experiment further:
- Try changing `chunk_size` in `ingest.py` and see how it affects answer quality.
- Try changing `k` in the retriever and observe what changes.
- Try modifying the prompt template to change Claude's behavior.
