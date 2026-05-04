import os
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
SRC_DIR = PARENT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_vector_db_env = os.environ.get("VECTOR_DB_DIR", "").strip()
VECTOR_DB_DIR = _vector_db_env or str(PARENT_DIR / "vector_db")
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_GITHUB_MODEL = "openai/gpt-4o"
DEFAULT_GITHUB_EMBEDDING_MODEL = "openai/text-embedding-3-small"
GITHUB_EMBEDDING_BASE_URL = "https://models.github.ai/inference"
DEFAULT_CHAT_MODEL = "deepseek-r1:1.5b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"
QUESTION_HISTORY_DIR = Path(VECTOR_DB_DIR) / "question_history"

os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(QUESTION_HISTORY_DIR, exist_ok=True)

__all__ = [
    "BASE_DIR",
    "PARENT_DIR",
    "SRC_DIR",
    "VECTOR_DB_DIR",
    "OLLAMA_BASE_URL",
    "DEFAULT_GITHUB_MODEL",
    "DEFAULT_GITHUB_EMBEDDING_MODEL",
    "GITHUB_EMBEDDING_BASE_URL",
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "QUESTION_HISTORY_DIR",
]