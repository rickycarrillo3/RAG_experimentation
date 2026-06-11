"""
ingest.py - Dual-collection ChromaDB ingest.
Text chunks -> 'text_collection' embedded with all-MiniLM-L6-v2.
Equation chunks -> 'equation_collection' embedded with AnReu/math_pretrained_bert
                   (BERT pretrained on arXiv math papers).

Run:
    python ingest.py
"""

import os
import re
import glob
import shutil
import pymupdf4llm
import torch
from transformers import AutoTokenizer, AutoModel
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DOCS_DIR = "../knowledge-base/docs"
TRANSCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../speech_2_text_rag/transcripts")
CHROMA_DIR = "chroma_db"
TEXT_EMBED_MODEL = "all-MiniLM-L6-v2"
MATH_EMBED_MODEL = "AnReu/math_pretrained_bert"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CONTEXT_WINDOW = 200  # chars of surrounding context stored in equation metadata

EQUATION_RE = re.compile(
    r'(\$\$[\s\S]+?\$\$'
    r'|\$[^$\n]{1,200}?\$'
    r'|\\begin\{[^}]+\}'
    r'|\\(?:frac|sum|int|prod|lim|infty|alpha|beta|gamma|delta|epsilon'
    r'|theta|lambda|mu|sigma|pi|nabla|partial|sqrt|forall|exists'
    r'|leq|geq|neq|approx|sim|in|subset|cup|cap|rightarrow|Rightarrow))',
)


class MathBERTEmbeddings(Embeddings):
    """Mean-pooled embeddings from AnReu/math_pretrained_bert."""

    def __init__(self, model_name: str = MATH_EMBED_MODEL):
        print(f"Loading math embedding model '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _mean_pool(self, texts: list[str]) -> list[list[float]]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        token_embs = outputs.last_hidden_state
        summed = torch.sum(token_embs * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        embeddings = (summed / counts).numpy()
        return embeddings.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Process in small batches to avoid OOM
        batch_size = 16
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            all_embeddings.extend(self._mean_pool(texts[i : i + batch_size]))
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._mean_pool([text])[0]


def has_equation(text: str) -> bool:
    return bool(EQUATION_RE.search(text))


def load_pdfs(docs_dir: str) -> list[Document]:
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

    text_chunks = []
    eq_chunks = []

    for i, chunk in enumerate(chunks):
        if has_equation(chunk.page_content):
            # Attach surrounding context as metadata for richer retrieval
            preceding = chunks[i - 1].page_content[:CONTEXT_WINDOW] if i > 0 else ""
            following = chunks[i + 1].page_content[:CONTEXT_WINDOW] if i < len(chunks) - 1 else ""
            chunk.metadata["preceding_context"] = preceding
            chunk.metadata["following_context"] = following
            chunk.metadata["chunk_type"] = "equation"
            eq_chunks.append(chunk)
        else:
            chunk.metadata["chunk_type"] = "text"
            text_chunks.append(chunk)

    print(f"Classified: {len(text_chunks)} text chunks, {len(eq_chunks)} equation chunks.")

    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    # Text collection
    print(f"Embedding text collection with '{TEXT_EMBED_MODEL}'...")
    text_embeddings = HuggingFaceEmbeddings(model_name=TEXT_EMBED_MODEL)
    text_store = Chroma(
        collection_name="text_collection",
        embedding_function=text_embeddings,
        persist_directory=CHROMA_DIR,
    )
    if text_chunks:
        text_store.add_documents(text_chunks)

    # Equation collection
    print(f"Embedding equation collection with '{MATH_EMBED_MODEL}'...")
    math_embeddings = MathBERTEmbeddings()
    eq_store = Chroma(
        collection_name="equation_collection",
        embedding_function=math_embeddings,
        persist_directory=CHROMA_DIR,
    )
    if eq_chunks:
        eq_store.add_documents(eq_chunks)

    print(f"\nDone!")
    print(f"  text_collection:     {len(text_chunks)} chunks  ({TEXT_EMBED_MODEL})")
    print(f"  equation_collection: {len(eq_chunks)} chunks  ({MATH_EMBED_MODEL})")


if __name__ == "__main__":
    ingest()
