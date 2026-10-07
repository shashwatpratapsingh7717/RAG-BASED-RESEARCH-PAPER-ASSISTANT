import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db" / "faiss"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# Application
APP_TITLE = "AI Research Paper Assistant"
# Embedding model
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)
# Chunking
CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "900")
)
CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "150")
)

# Retrieval
TOP_K_DEFAULT = int(
    os.getenv("TOP_K_DEFAULT", "5")
)