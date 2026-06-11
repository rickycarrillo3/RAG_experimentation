"""
ingest.py - Extract PDFs with pymupdf4llm, detect equation chunks, generate plain-English
descriptions via Qwen, embed the descriptions for semantic search, and store in ChromaDB.

Run:
    python ingest.py
"""

import os
import re
import glob
import requests
import pymupdf4llm
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DOCS_DIR = "../knowledge-base/docs"
TRANSCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../speech_2_text_rag/transcripts")
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5:14b"
OLLAMA_URL = "http://localhost:11434/api/generate"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Detects LaTeX patterns commonly found in math textbooks
EQUATION_RE = re.compile(
    r'(\$\$[\s\S]+?\$\$'          # display math $$...$$
    r'|\$[^$\n]{1,200}?\$'        # inline math $...$
    r'|\\begin\{[^}]+\}'          # \begin{equation}, \begin{align}, etc.
    r'|\\(?:frac|sum|int|prod|lim|infty|alpha|beta|gamma|delta|epsilon'
    r'|theta|lambda|mu|sigma|pi|nabla|partial|sqrt|forall|exists'
    r'|leq|geq|neq|approx|sim|in|subset|cup|cap|rightarrow|Rightarrow))',
)

DESCRIBE_PROMPT = (
    "The following is a passage from a mathematics textbook. "
    "It contains one or more mathematical equations or formulas. "
    "In 1-2 concise plain English sentences, describe what the equation(s) express "
    "and their significance in context. Do not use LaTeX or mathematical notation.\n\n"
    "Passage:\n{chunk}"
)


def has_equation(text):
    return bool(EQUATION_RE.search(text))


def describe_equation(chunk_text):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": DESCRIBE_PROMPT.format(chunk=chunk_text),
        "stream": False,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()["response"].strip()


def load_pdfs(docs_dir):
    documents = []
    for pdf_path in glob.glob(os.path.join(docs_dir, "*.pdf")):
        pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
        for page in pages:
            documents.append(Document(
                page_content=page["text"],
                metadata={"source": pdf_path, "page": page["metadata"]["page_number"]},
            ))
    return documents


def ingest():
    print(f"Loading documents from '{DOCS_DIR}/'...")
    documents = load_pdfs(DOCS_DIR)
    if os.path.isdir(TRANSCRIPTS_DIR):
        txt_loader = DirectoryLoader(TRANSCRIPTS_DIR, glob="*.txt", loader_cls=TextLoader)
        documents += txt_loader.load()

    if not documents:
        print("No documents found. Drop PDFs into docs/ and re-run.")
        return

    print(f"Loaded {len(documents)} page(s).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    eq_count = 0
    print("Detecting equation chunks and generating descriptions...")
    for i, chunk in enumerate(chunks):
        if has_equation(chunk.page_content):
            print(f"  Describing equation chunk {eq_count + 1} (chunk {i+1}/{len(chunks)})...", end="\r", flush=True)
            description = describe_equation(chunk.page_content)
            # Store original LaTeX text; embed the NL description instead
            chunk.metadata["original_text"] = chunk.page_content
            chunk.metadata["equation_description"] = description
            chunk.metadata["has_equation"] = True
            chunk.page_content = description
            eq_count += 1
        else:
            chunk.metadata["has_equation"] = False

    print(f"\nGenerated descriptions for {eq_count} equation chunk(s) out of {len(chunks)} total.")

    print(f"Loading embedding model '{EMBED_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print(f"Embedding and persisting to '{CHROMA_DIR}/'...")
    if os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIR)

    print(f"\nDone! {len(chunks)} chunks stored in '{CHROMA_DIR}/'.")
    print("Equation chunks are embedded by their plain-English description; original LaTeX is in metadata.")


if __name__ == "__main__":
    ingest()
