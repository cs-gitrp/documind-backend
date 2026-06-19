import os, uuid, json
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from config import UPLOAD_DIR, INDEX_DIR

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_document(file_path: str, file_type: str):
    if file_type == "pdf":
        doc = fitz.open(file_path)
        page_texts = []
        for page_num, page in enumerate(doc):
            page_texts.append({"page": page_num + 1, "text": page.get_text()})
        return page_texts, len(doc)
    elif file_type == "docx":
        doc = DocxDocument(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return [{"page": 1, "text": text}], 1
    elif file_type == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return [{"page": 1, "text": text}], 1
    raise ValueError(f"Unsupported file type: {file_type}")

def chunk_and_embed(page_texts: list, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = []
    chunk_pages = []
    
    for page_data in page_texts:
        page_chunks = splitter.split_text(page_data["text"])
        for chunk in page_chunks:
            chunks.append(chunk)
            chunk_pages.append(page_data["page"])
    
    embeddings = model.encode(chunks, show_progress_bar=False)
    return chunks, chunk_pages, embeddings

def save_faiss_index(doc_id: str, chunks: list, chunk_pages: list, embeddings):
    os.makedirs(INDEX_DIR, exist_ok=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))

    index_path = os.path.join(INDEX_DIR, f"{doc_id}.index")
    meta_path = os.path.join(INDEX_DIR, f"{doc_id}_meta.json")

    faiss.write_index(index, index_path)
    with open(meta_path, "w") as f:
        json.dump({"chunks": chunks, "pages": chunk_pages}, f)

    return index_path

def run_ingestion(file_path: str, file_type: str, doc_id: str,
                  chunk_size: int = 1024, chunk_overlap: int = 128):
    page_texts, page_count = load_document(file_path, file_type)
    chunks, chunk_pages, embeddings = chunk_and_embed(page_texts, chunk_size, chunk_overlap)
    index_path = save_faiss_index(doc_id, chunks, chunk_pages, embeddings)
    full_text = " ".join([p["text"] for p in page_texts])
    summary = full_text[:200].replace("\n", " ").strip()
    return {
        "page_count": page_count,
        "chunk_count": len(chunks),
        "index_path": index_path,
        "summary": summary
    }