import uuid, json
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from models.database import get_db, ChatSession, Message, RAGSettings
from services.retrieval import retrieve_chunks
from services.llm import build_prompt, stream_response

router = APIRouter(prefix="/api/chat", tags=["chat"])

class ChatRequest(BaseModel):
    query: str
    document_id: str
    session_id: Optional[str] = None

@router.post("/")
def chat(req: ChatRequest, x_client_id: Optional[str] = Header(None), db: Session = Depends(get_db)):
    settings = db.query(RAGSettings).filter(RAGSettings.id == 1).first()
    top_k = settings.top_k if settings else 5
    temperature = settings.temperature if settings else 0.7
    response_mode = settings.response_mode if settings else "detailed"

    try:
        chunks = retrieve_chunks(req.document_id, req.query, top_k)
    except FileNotFoundError:
        raise HTTPException(404, "Document index not found. Please re-upload.")

    prompt = build_prompt(req.query, chunks, response_mode)

    # Get or create session
    session_id = req.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        session = ChatSession(
            id=session_id,
            title=req.query[:60],
            document_id=req.document_id,
            client_id=x_client_id
        )
        db.add(session)
        db.commit()

    # Save user message
    db.add(Message(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role="user",
        content=req.query
    ))
    db.commit()

    sources = [
    {"chunk_id": c["chunk_id"], "page": c["page"], "score": c["score"], "text": c["chunk"][:300]}
    for c in chunks
]

    full_response = []

    def generate():
        for token in stream_response(prompt, temperature):
            full_response.append(token)
            yield token
        # Save assistant message after streaming
        final = "".join(full_response)
        from models.database import SessionLocal
        save_db = SessionLocal()
        save_db.add(Message(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content=final,
            sources=json.dumps(sources)
        ))
        session = save_db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            session.message_count += 2
            session.updated_at = datetime.now(timezone.utc)
        save_db.commit()
        save_db.close()
        yield f"\n__SESSION_ID__{session_id}__SOURCES__{json.dumps(sources)}"

    return StreamingResponse(generate(), media_type="text/plain")

@router.get("/sessions")
def get_sessions(x_client_id: Optional[str] = Header(None), db: Session = Depends(get_db)):
    sessions = db.query(ChatSession).filter(ChatSession.client_id == x_client_id).order_by(ChatSession.created_at.desc()).all()
    return [
        {
            "id": s.id, "title": s.title, "document_id": s.document_id,
            "created_at": s.created_at.isoformat() + "Z" if s.created_at else None,
            "updated_at": s.updated_at.isoformat() + "Z" if s.updated_at else None,
            "message_count": s.message_count
        } for s in sessions
    ]

@router.get("/sessions/{session_id}")
def get_session(session_id: str, db: Session = Depends(get_db)):
    messages = db.query(Message).filter(
        Message.session_id == session_id
    ).order_by(Message.created_at).all()
    return [
        {
            "id": m.id, "role": m.role, "content": m.content,
            "sources": json.loads(m.sources) if m.sources else [],
            "created_at": m.created_at.isoformat() + "Z" if m.created_at else None
        } for m in messages
    ]

@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)):
    db.query(Message).filter(Message.session_id == session_id).delete()
    db.query(ChatSession).filter(ChatSession.id == session_id).delete()
    db.commit()
    return {"success": True}