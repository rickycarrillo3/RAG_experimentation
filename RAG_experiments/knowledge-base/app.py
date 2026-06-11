"""
app.py - Streamlit chat UI for the knowledge base.

Run:
    streamlit run app.py
"""

import os
import streamlit as st
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

Context:
{context}"""

st.set_page_config(page_title="Knowledge Base", page_icon="📚")
st.title("Knowledge Base")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource
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


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            st.caption(f"Sources: {message['sources']}")

# Chat input
if question := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retriever, answer_chain = load_components()
            docs = retriever.invoke(question)
            answer = answer_chain.invoke({"context": format_docs(docs), "input": question})

        sources = ", ".join(
            set(os.path.basename(doc.metadata.get("source", "unknown")) for doc in docs)
        )
        st.markdown(answer)
        st.caption(f"Sources: {sources}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
