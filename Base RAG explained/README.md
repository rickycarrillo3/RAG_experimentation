# Base RAG explained

Background reading on how a Retrieval-Augmented Generation (RAG) pipeline works,
from raw documents to a grounded answer. These are general concept notes — the
math-specific system that grew out of them lives in [`../knowledge-base-math/`](../knowledge-base-math/).

Read them in order:

1. [What is RAG?](00_what_is_rag.md) — the problem RAG solves and the shape of the pipeline.
2. [Step 1: Ingestion](01_ingestion.md) — turning documents into chunks.
3. [Step 2: The Vector Store (ChromaDB)](02_vector_store.md) — embedding chunks and storing them for search.
4. [Step 3: Retrieval and Generation](03_retrieval_and_generation.md) — finding relevant chunks and feeding them to the LLM.
5. [Step 4: The Prompt Template](04_the_prompt.md) — assembling the retrieved context into the model's prompt.
