"""
query.py - CLI chat interface for the hybrid two-collection knowledge base.
Searches text_collection and equation_collection independently, merges results.

Run:
    python query.py
"""

import os
import torch
from typing import List
from transformers import AutoTokenizer, AutoModel
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

CHROMA_DIR = "chroma_db"
TEXT_EMBED_MODEL = "all-MiniLM-L6-v2"
MATH_EMBED_MODEL = "AnReu/math_pretrained_bert"
OLLAMA_MODEL = "qwen2.5:14b"
TOP_K = 4  # results from each collection

SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly from the provided context.

Rules:
- Answer only based on the context below.
- If the answer is not present in the context, say exactly: "I don't have that in my knowledge base."
- Always cite the source document filename(s) at the end of your answer.
- Do not use markdown formatting. Write in plain text only.
- Context may include equation chunks with surrounding context — use it.

Context:
{context}"""


class MathBERTEmbeddings(Embeddings):
    def __init__(self, model_name: str = MATH_EMBED_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _mean_pool(self, texts: List[str]) -> List[List[float]]:
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        token_embs = outputs.last_hidden_state
        summed = torch.sum(token_embs * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return (summed / counts).numpy().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_size = 16
        result = []
        for i in range(0, len(texts), batch_size):
            result.extend(self._mean_pool(texts[i : i + batch_size]))
        return result

    def embed_query(self, text: str) -> List[float]:
        return self._mean_pool([text])[0]


def format_docs(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        content = doc.page_content
        preceding = doc.metadata.get("preceding_context", "")
        following = doc.metadata.get("following_context", "")
        chunk_type = doc.metadata.get("chunk_type", "text")

        if chunk_type == "equation" and (preceding or following):
            parts.append(
                f"[Context before]: {preceding}\n"
                f"[Equation]: {content}\n"
                f"[Context after]: {following}"
            )
        else:
            parts.append(content)
    return "\n\n".join(parts)


def deduplicate(docs: List[Document]) -> List[Document]:
    seen = set()
    unique = []
    for doc in docs:
        key = doc.page_content.strip()[:100]
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def load_components():
    print("Loading text embeddings...")
    text_embeddings = HuggingFaceEmbeddings(model_name=TEXT_EMBED_MODEL)
    text_store = Chroma(
        collection_name="text_collection",
        embedding_function=text_embeddings,
        persist_directory=CHROMA_DIR,
    )

    print("Loading math embeddings...")
    math_embeddings = MathBERTEmbeddings()
    eq_store = Chroma(
        collection_name="equation_collection",
        embedding_function=math_embeddings,
        persist_directory=CHROMA_DIR,
    )

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, num_predict=1024)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    answer_chain = prompt | llm | StrOutputParser()

    return text_store, eq_store, answer_chain


def ask(question: str, text_store: Chroma, eq_store: Chroma, answer_chain) -> tuple:
    text_docs = text_store.similarity_search(question, k=TOP_K)
    eq_docs = eq_store.similarity_search(question, k=TOP_K)

    # Interleave: one text result, one equation result, alternating
    merged = []
    for t, e in zip(text_docs, eq_docs):
        merged.extend([t, e])
    merged += text_docs[len(eq_docs):] + eq_docs[len(text_docs):]
    merged = deduplicate(merged)

    answer = answer_chain.invoke({"context": format_docs(merged), "input": question})
    return answer, merged


def main():
    print("Loading hybrid knowledge base...")
    text_store, eq_store, answer_chain = load_components()
    print("Ready! Type your question (or 'quit' to exit).\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        answer, docs = ask(question, text_store, eq_store, answer_chain)
        print(f"\nQwen: {answer}")
        sources = set(os.path.basename(doc.metadata.get("source", "unknown")) for doc in docs)
        eq_count = sum(1 for d in docs if d.metadata.get("chunk_type") == "equation")
        print(f"Sources: {', '.join(sources)} | Text chunks: {len(docs)-eq_count}, Equation chunks: {eq_count}\n")


if __name__ == "__main__":
    main()
