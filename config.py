# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/documind.db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")
INDEX_DIR = os.getenv("INDEX_DIR", "./data/indexes")

DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 128
DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.7
DEFAULT_RESPONSE_MODE = "detailed"