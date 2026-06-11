"""
query.py - CLI chat interface for the knowledge base.

Run:
    python query.py
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5:14b"

SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly from the provided context.

Rules:
- Answer only based on the context below.
- If the answer is not present in the context, say exactly: "I don't have that in my knowledge base."
- Always cite the source document filename(s) at the end of your answer.
- Do not use markdown formatting. Write in plain text only.

Context:
{context}"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_components():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, num_predict=1024)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    answer_chain = prompt | llm | StrOutputParser()

    return retriever, answer_chain


def ask(question, retriever, answer_chain):
    docs = retriever.invoke(question)
    answer = answer_chain.invoke({"context": format_docs(docs), "input": question})
    return answer, docs


def main():
    print("Loading knowledge base...")
    retriever, answer_chain = load_components()
    print("Ready! Type your question (or 'quit' to exit).\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        answer, docs = ask(question, retriever, answer_chain)

        print(f"\nQwen: {answer}")
        sources = set(os.path.basename(doc.metadata.get("source", "unknown")) for doc in docs)
        print(f"Sources: {', '.join(sources)}\n")


if __name__ == "__main__":
    main()
