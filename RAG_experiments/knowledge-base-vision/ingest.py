"""
ingest.py - Render each PDF page as an image, extract text+LaTeX via Ollama vision model,
chunk, embed, and store in ChromaDB.

Requires a vision-capable model pulled in Ollama, e.g.:
    ollama pull llava

Run:
    python ingest.py
"""

import os
import glob
import base64
import requests
import pymupdf
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DOCS_DIR = "../knowledge-base/docs"
TRANSCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../speech_2_text_rag/transcripts")
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
VISION_MODEL = "llava"
OLLAMA_URL = "http://localhost:11434/api/generate"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DPI = 150

EXTRACT_PROMPT = (
    "Extract all text from this page exactly as it appears. "
    "Render any mathematical equations or formulas in LaTeX, "
    "using $ for inline math and $$ for display math on its own line. "
    "Return only the extracted text with no commentary or explanation."
)


def page_to_base64(page):
    mat = pymupdf.Matrix(DPI / 72, DPI / 72)
    pix = page.get_pixmap(matrix=mat)
    return base64.b64encode(pix.tobytes("png")).decode()


def extract_page_text(b64_image):
    payload = {
        "model": VISION_MODEL,
        "prompt": EXTRACT_PROMPT,
        "images": [b64_image],
        "stream": False,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()["response"]


def load_pdfs_vision(docs_dir):
    documents = []
    for pdf_path in glob.glob(os.path.join(docs_dir, "*.pdf")):
        doc = pymupdf.open(pdf_path)
        total = doc.page_count
        print(f"  {os.path.basename(pdf_path)} — {total} pages")
        for page_num, page in enumerate(doc, start=1):
            print(f"    Extracting page {page_num}/{total}...", end="\r", flush=True)
            b64 = page_to_base64(page)
            text = extract_page_text(b64)
            documents.append(Document(
                page_content=text,
                metadata={"source": pdf_path, "page": page_num},
            ))
        print()
    return documents


def ingest():
    print(f"Loading PDFs via vision model '{VISION_MODEL}'...")
    documents = load_pdfs_vision(DOCS_DIR)

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

    print(f"Loading embedding model '{EMBED_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print(f"Embedding and persisting to '{CHROMA_DIR}/'...")
    if os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIR)

    print(f"\nDone! {len(chunks)} chunks stored in '{CHROMA_DIR}/'.")


if __name__ == "__main__":
    ingest()
