from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from models.database import (
    get_db,
    Document,
    ChatSession,
    Message
)

router = APIRouter(
    prefix="/api/search",
    tags=["search"]
)

@router.get("/")
def search(
    q: str,
    db: Session = Depends(get_db)
):
    if not q.strip():
        return {
            "documents": [],
            "sessions": [],
            "messages": []
        }

    documents = (
        db.query(Document)
        .filter(Document.filename.ilike(f"%{q}%"))
        .limit(5)
        .all()
    )

    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.title.ilike(f"%{q}%"))
        .order_by(ChatSession.created_at.desc())
        .limit(5)
        .all()
    )

    messages = (
        db.query(Message, ChatSession.document_id)
        .join(ChatSession, Message.session_id == ChatSession.id)
        .filter(Message.content.ilike(f"%{q}%"))
        .limit(5)
        .all()
    )

    return {
        "documents": [
            {
                "id": d.id,
                "filename": d.filename
            }
            for d in documents
        ],

        "sessions": [
            {
                "id": s.id,
                "title": s.title,
                "document_id": s.document_id
            }
            for s in sessions
        ],

        "messages": [
            {
                "id": m.Message.id,
                "session_id": m.Message.session_id,
                "document_id": m.document_id,
                "preview": m.Message.content[:150]
            }
            for m in messages
        ]
    }