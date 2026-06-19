import uuid
import datetime
from sqlalchemy import create_engine, Column, String, Integer, Float, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper function to generate timezone-aware UTC timestamps
def get_utc_now():
    return datetime.datetime.now(datetime.timezone.utc)

class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer)
    page_count = Column(Integer)
    chunk_count = Column(Integer)
    status = Column(String, default="processing")
    summary = Column(Text)
    upload_date = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    faiss_index_path = Column(String)

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String)
    document_id = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    message_count = Column(Integer, default=0)

class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))

class RAGSettings(Base):
    __tablename__ = "rag_settings"
    id = Column(Integer, primary_key=True, default=1)
    chunk_size = Column(Integer, default=1024)
    chunk_overlap = Column(Integer, default=128)
    top_k = Column(Integer, default=5)
    temperature = Column(Float, default=0.7)
    response_mode = Column(String, default="detailed")

def init_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    if not db.query(RAGSettings).first():
        db.add(RAGSettings())
        db.commit()
    db.close()