import os
import shutil
from fastapi import APIRouter, Depends, Header
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from models.database import get_db, RAGSettings, Document
from config import INDEX_DIR, UPLOAD_DIR  # Assumed UPLOAD_DIR is exported here

router = APIRouter(prefix="/api/settings", tags=["settings"])

class SettingsUpdate(BaseModel):
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    response_mode: Optional[str] = None

def get_dir_size(path: str) -> float:
    if not os.path.exists(path):
        return 0.0
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return round(total / (1024 * 1024), 2)

@router.get("/storage-stats")
def get_storage_stats(x_client_id: Optional[str] = Header(None), db: Session = Depends(get_db)):
    from models.database import Document

    docs = db.query(Document).filter(Document.client_id == x_client_id).all()
    total_bytes = sum(d.file_size for d in docs if d.file_size is not None)
    documents_mb = round(total_bytes / (1024 * 1024), 2)
    index_mb = documents_mb

    return {
        "documents_mb": documents_mb,
        "index_mb": index_mb,
        "total_mb": round(documents_mb + index_mb, 2),
        "max_mb": 100,
        "document_count": len(docs)
    }

@router.get("/")
def get_settings(db: Session = Depends(get_db)):
    s = db.query(RAGSettings).filter(RAGSettings.id == 1).first()
    return {
        "chunk_size": s.chunk_size, 
        "chunk_overlap": s.chunk_overlap,
        "top_k": s.top_k, 
        "temperature": s.temperature,
        "response_mode": s.response_mode
    }

@router.put("/")
def update_settings(body: SettingsUpdate, db: Session = Depends(get_db)):
    s = db.query(RAGSettings).filter(RAGSettings.id == 1).first()
    if body.chunk_size is not None: s.chunk_size = body.chunk_size
    if body.chunk_overlap is not None: s.chunk_overlap = body.chunk_overlap
    if body.top_k is not None: s.top_k = body.top_k
    if body.temperature is not None: s.temperature = body.temperature
    if body.response_mode is not None: s.response_mode = body.response_mode
    db.commit()
    return {"success": True}

@router.delete("/storage")
def clear_storage(x_client_id: Optional[str] = Header(None), db: Session = Depends(get_db)):
    docs = db.query(Document).filter(Document.client_id == x_client_id).all()
    for doc in docs:
        doc_id = doc.id
        # Delete uploaded files
        for ext in ["pdf", "docx", "txt"]:
            path = os.path.join(UPLOAD_DIR, f"{doc_id}.{ext}")
            if os.path.exists(path):
                os.remove(path)
                
        # Delete FAISS index
        for suffix in [".index", "_meta.json"]:
            path = os.path.join(INDEX_DIR, f"{doc_id}{suffix}")
            if os.path.exists(path):
                os.remove(path)
                
        db.delete(doc)
        
    db.commit()
    return {"success": True, "message": "All documents and indexes deleted."}