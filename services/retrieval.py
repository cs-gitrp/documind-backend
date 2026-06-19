import json, os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import INDEX_DIR

model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_chunks(doc_id: str, query: str, top_k: int = 5):
    index_path = os.path.join(INDEX_DIR, f"{doc_id}.index")
    meta_path = os.path.join(INDEX_DIR, f"{doc_id}_meta.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No index found for document {doc_id}")

    index = faiss.read_index(index_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    chunks = meta["chunks"]
    pages = meta.get("pages", [1] * len(chunks))
    query_vec = model.encode([query]).astype(np.float32)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            results.append({
                "chunk": chunks[idx],
                "chunk_id": int(idx),
                "page": pages[idx],
                "score": float(distances[0][i])
            })
    return results