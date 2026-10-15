from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    groq_api_key: str = ""
    generation_model: str = "llama-3.3-70b-versatile"
    # small model for query analysis and grading: cheaper and fast enough
    grader_model: str = "llama-3.1-8b-instant"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    chroma_dir: Path = ROOT_DIR / "chroma_db"
    collection_name: str = "docs"
    corpus_dir: Path = ROOT_DIR / "data" / "corpus"
    feedback_db: Path = ROOT_DIR / "data" / "feedback.sqlite3"

    top_k: int = 5
    max_retries: int = 2
    chunk_size: int = 900
    chunk_overlap: int = 150


@lru_cache
def get_settings() -> Settings:
    return Settings()
