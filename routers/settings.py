import os
import shutil
from fastapi import APIRouter, Depends
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
def get_storage_stats(db: Session = Depends(get_db)):
    from models.database import Document
    from config import UPLOAD_DIR, INDEX_DIR

    documents_size = get_dir_size(UPLOAD_DIR)
    index_size = get_dir_size(INDEX_DIR)
    total_docs = db.query(Document).count()

    return {
        "documents_mb": documents_size,
        "index_mb": index_size,
        "total_mb": round(documents_size + index_size, 2),
        "max_mb": 100,
        "document_count": total_docs
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
def clear_storage(db: Session = Depends(get_db)):
    # Delete all uploaded files
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR)
        
    # Delete all FAISS indexes
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
        os.makedirs(INDEX_DIR)
        
    # Delete all document records from PostgreSQL/Supabase
    db.query(Document).delete()
    db.commit()
    
    return {"success": True, "message": "All documents and indexes deleted."}