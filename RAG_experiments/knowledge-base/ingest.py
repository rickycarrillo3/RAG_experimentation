"""
ingest.py - Load PDFs and TXT files from docs/, chunk them, embed, and store in ChromaDB.

Run this once (and re-run whenever you add new documents):
    python ingest.py
"""

import os
import glob
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
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


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
    # 1. Load all PDFs and TXT files from docs/
    print(f"Loading documents from '{DOCS_DIR}/'...")
    documents = load_pdfs(DOCS_DIR)
    if os.path.isdir(TRANSCRIPTS_DIR):
        txt_loader = DirectoryLoader(TRANSCRIPTS_DIR, glob="*.txt", loader_cls=TextLoader)
        documents += txt_loader.load()

    if not documents:
        print("No documents found. Drop PDFs or TXT files into the docs/ folder and re-run.")
        return

    print(f"Loaded {len(documents)} page(s) from {len(set(d.metadata['source'] for d in documents))} file(s).")

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    # 3. Embed using a local HuggingFace model (no API key needed)
    print(f"Loading embedding model '{EMBED_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # 4. Store embeddings in ChromaDB (persisted to chroma_db/)
    print(f"Embedding chunks and persisting to '{CHROMA_DIR}/'...")
    if os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print(f"\nDone! {len(chunks)} chunks stored in '{CHROMA_DIR}/'.")
    print("You can now run query.py or app.py to chat with your documents (PDFs and transcripts).")


if __name__ == "__main__":
    ingest()
