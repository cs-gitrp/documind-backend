# pyrefly: ignore [missing-import]
from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
import os
from models.database import init_db
from routers import documents, chat, settings
from routers import search

app = FastAPI(title="DocuMind AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://documind-ai-ashy-ten.vercel.app",
        "https://documind-ai-csdata294-3679s-projects.vercel.app",
        "https://documind-79yy75tae-csdata294-3679s-projects.vercel.app",
        ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(settings.router)

@app.on_event("startup")
async def startup():
    os.makedirs("./data/uploads", exist_ok=True)
    os.makedirs("./data/indexes", exist_ok=True)
    init_db()

@app.get("/")
def root():
    return {"status": "DocuMind backend running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"healthy": True}