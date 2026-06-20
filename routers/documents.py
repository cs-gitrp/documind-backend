import os, uuid, shutil
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, Header
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from models.database import get_db, Document
from services.ingestion import run_ingestion
from config import UPLOAD_DIR, INDEX_DIR
from pydantic import BaseModel
from typing import Optional


router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_TYPES = {"pdf": "pdf", "docx": "docx", "txt": "txt"}

class RenameRequest(BaseModel):
    filename: str

@router.patch("/{doc_id}")
def rename_document(
    doc_id: str,
    data: RenameRequest,
    db: Session = Depends(get_db)
):
    doc = db.query(Document).filter(
        Document.id == doc_id
    ).first()

    if not doc:
        raise HTTPException(404, "Document not found")

    doc.filename = data.filename

    db.commit()

    return {
        "success": True
    }
    
@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    x_client_id: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    ext = file.filename.split(".")[-1].lower()
    if ext not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    doc_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}.{ext}")

    with open(save_path, "wb") as f:
        f.write(await file.read())

    doc = Document(
        id=doc_id,
        filename=file.filename,
        file_type=ext,
        file_size=os.path.getsize(save_path),
        status="processing",
        client_id=x_client_id
    )
    db.add(doc)
    db.commit()

    background_tasks.add_task(process_document, doc_id, save_path, ext)
    return {"id": doc_id, "filename": file.filename, "status": "processing"}

def process_document(doc_id: str, file_path: str, file_type: str):
    from models.database import SessionLocal
    db = SessionLocal()
    try:
        settings = db.execute(__import__('sqlalchemy').text(
            "SELECT chunk_size, chunk_overlap FROM rag_settings WHERE id=1"
        )).fetchone()
        chunk_size = settings[0] if settings else 1024
        chunk_overlap = settings[1] if settings else 128

        result = run_ingestion(file_path, file_type, doc_id, chunk_size, chunk_overlap)

        doc = db.query(Document).filter(Document.id == doc_id).first()
        doc.status = "indexed"
        doc.page_count = result["page_count"]
        doc.chunk_count = result["chunk_count"]
        doc.faiss_index_path = result["index_path"]
        doc.summary = result["summary"]
        db.commit()
    except Exception as e:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            doc.status = "failed"
            db.commit()
    finally:
        db.close()

@router.get("/")
def list_documents(x_client_id: Optional[str] = Header(None), db: Session = Depends(get_db)):
    docs = db.query(Document).filter(Document.client_id == x_client_id).order_by(Document.upload_date.desc()).all()
    return [
        {
            "id": d.id, "filename": d.filename, "file_type": d.file_type,
            "file_size": d.file_size, "page_count": d.page_count,
            "chunk_count": d.chunk_count, "status": d.status,
            "summary": d.summary,
            "upload_date": d.upload_date.isoformat() + "Z" if d.upload_date else None
        } for d in docs
    ]

@router.delete("/{doc_id}")
def delete_document(doc_id: str, x_client_id: Optional[str] = Header(None), db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id, Document.client_id == x_client_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")

    # Delete uploaded file
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
    return {"success": True}

@router.get("/{doc_id}/download")
def download_document(doc_id: str, inline: bool = False, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
        
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.{doc.file_type}")
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found on disk")
        
    file_type = doc.file_type.lower()

    # 1. Handle TXT files with clear charset formatting for inline rendering
    if file_type == "txt":
        content_type = "text/plain; charset=utf-8"
    elif file_type == "pdf":
        content_type = "application/pdf"
    else:
        content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    # 2. Block direct DOCX inline tabs to prevent browser-enforced downloads
    if inline and file_type == "docx":
        from fastapi.responses import HTMLResponse
        return HTMLResponse(
            content=f"""
            <html>
                <body style="font-family:sans-serif; display:flex; align-items:center; justify-content:center; height:100vh; background:#0F172A; color:#F8FAFC;">
                    <div style="text-align:center; max-width:400px; padding:2rem; border:1px solid #334155; rounded:12px; background:#1E293B;">
                        <h3 style="margin-bottom:0.5rem;">DOCX Inline Preview</h3>
                        <p style="font-size:0.875rem; color:#94A3B8; margin-bottom:1.5rem;">Browsers cannot view Word files natively. Use the chat panel to query its content, or download it directly.</p>
                        <a href="/api/documents/{doc_id}/download" style="text-decoration:none; background:#2563EB; color:white; padding:0.5rem 1rem; border-radius:6px; font-size:0.875rem;">Download Original</a>
                    </div>
                </body>
            </html>
            """
        )

    if inline:
        return FileResponse(path=file_path, media_type=content_type)
        
    return FileResponse(path=file_path, filename=doc.filename, media_type=content_type)