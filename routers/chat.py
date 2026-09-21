"""
routers/chat.py — DocuMind v2

New vs v1:
  - mode="agent"  → runs the agentic tool-calling loop (services/agent.py)
  - mode="rag"    → runs the improved hybrid retrieval + reranking pipeline
  - Query rewriting (HyDE + multi-query) happens transparently in both modes
  - Sources now include rerank_score and rrf_score for transparency
"""

import json
import uuid
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from models.database import ChatSession, Message, RAGSettings, SessionLocal, get_db
from services.agent import run_agent
from services.llm import build_prompt, stream_response
from services.query_rewriter import rewrite_and_expand
from services.retrieval import retrieve_hybrid

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    query:       str
    document_id: str
    session_id:  str | None = None
    mode:        str = "rag"       # "rag" | "agent"
    use_hyde:    bool = True       # enable HyDE query rewriting
    use_multi_query: bool = False  # enable multi-query expansion (slower, higher recall)


@router.post("/")
def chat(
    req: ChatRequest,
    x_client_id: str | None = Header(None),
    db: Session = Depends(get_db),
):
    settings      = db.query(RAGSettings).filter(RAGSettings.id == 1).first()
    top_k         = settings.top_k if settings else 5
    temperature   = settings.temperature if settings else 0.3
    response_mode = settings.response_mode if settings else "balanced"

    # ── Session management ──
    session_id = req.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        session = ChatSession(
            id=session_id,
            title=req.query[:60],
            document_id=req.document_id,
            client_id=x_client_id,
        )
        db.add(session)
        db.commit()

    # Save user message
    db.add(Message(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role="user",
        content=req.query,
    ))
    db.commit()

    # ─────────────────────────────────────────
    #  AGENT MODE
    # ─────────────────────────────────────────
    if req.mode == "agent":
        full_response: list[str] = []

        def agent_generate():
            for token in run_agent(
                query=req.query,
                doc_id=req.document_id,
                temperature=temperature,
            ):
                full_response.append(token)
                yield token

            final = "".join(full_response)
            _save_assistant_message(session_id, final, sources=[], session_db=None)
            yield f"\n__SESSION_ID__{session_id}__SOURCES__[]"

        return StreamingResponse(agent_generate(), media_type="text/plain")

    # ─────────────────────────────────────────
    #  RAG MODE  (hybrid retrieval + reranking)
    # ─────────────────────────────────────────

    # Query rewriting — use HyDE embedding for retrieval, original for prompt
    rewritten = rewrite_and_expand(
        req.query,
        use_hyde=req.use_hyde,
        use_multi=req.use_multi_query,
    )
    retrieval_query = rewritten["hyde_query"]

    # Multi-query: retrieve for each sub-query, deduplicate by chunk_id
    if req.use_multi_query:
        all_chunks: dict[int, dict] = {}
        for sub_q in rewritten["sub_queries"]:
            for chunk in retrieve_hybrid(req.document_id, sub_q, top_k=top_k):
                cid = chunk["chunk_id"]
                if cid not in all_chunks or chunk.get("rerank_score", 0) > all_chunks[cid].get("rerank_score", 0):
                    all_chunks[cid] = chunk
        chunks = sorted(all_chunks.values(), key=lambda x: -x.get("rerank_score", x.get("rrf_score", 0)))[:top_k]
    else:
        try:
            chunks = retrieve_hybrid(req.document_id, retrieval_query, top_k=top_k)
        except FileNotFoundError:
            raise HTTPException(404, "Document index not found. Please re-upload.")

    prompt = build_prompt(req.query, chunks, response_mode)

    sources = [
        {
            "chunk_id":     c["chunk_id"],
            "page":         c["page"],
            "score":        round(c.get("rerank_score", c.get("rrf_score", c.get("score", 0))), 4),
            "rrf_score":    round(c.get("rrf_score", 0), 4),
            "rerank_score": round(c.get("rerank_score", 0), 4),
            "text":         c["chunk"][:300],
        }
        for c in chunks
    ]

    full_response: list[str] = []

    def rag_generate():
        for token in stream_response(prompt, temperature):
            full_response.append(token)
            yield token
        final = "".join(full_response)
        _save_assistant_message(session_id, final, sources, session_db=None)
        yield f"\n__SESSION_ID__{session_id}__SOURCES__{json.dumps(sources, default=lambda value: value.item() if isinstance(value, np.generic) else value)}"

    return StreamingResponse(rag_generate(), media_type="text/plain")


def _save_assistant_message(session_id: str, content: str, sources: list, session_db=None):
    db = SessionLocal()
    try:
        db.add(Message(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content=content,
            sources=json.dumps(sources, default=lambda value: value.item() if isinstance(value, np.generic) else value),
        ))
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            session.message_count += 2
            session.updated_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()


# ── Session endpoints (unchanged from v1) ──

@router.get("/sessions")
def get_sessions(x_client_id: str | None = Header(None), db: Session = Depends(get_db)):
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.client_id == x_client_id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )
    return [
        {
            "id": s.id, "title": s.title, "document_id": s.document_id,
            "created_at": s.created_at.isoformat() + "Z" if s.created_at else None,
            "updated_at": s.updated_at.isoformat() + "Z" if s.updated_at else None,
            "message_count": s.message_count,
        }
        for s in sessions
    ]


@router.get("/sessions/{session_id}")
def get_session(session_id: str, db: Session = Depends(get_db)):
    messages = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at)
        .all()
    )
    return [
        {
            "id": m.id, "role": m.role, "content": m.content,
            "sources": json.loads(m.sources) if m.sources else [],
            "created_at": m.created_at.isoformat() + "Z" if m.created_at else None,
        }
        for m in messages
    ]


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)):
    db.query(Message).filter(Message.session_id == session_id).delete()
    db.query(ChatSession).filter(ChatSession.id == session_id).delete()
    db.commit()
    return {"success": True}
