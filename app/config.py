from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    groq_api_key: str = ""
    # Tested with openai/gpt-oss-20b on Groq. Which models an account may use differs,
    # so each role can be overridden on its own (GENERATION_MODEL and so on).
    generation_model: str = "openai/gpt-oss-20b"
    # query analysis, chunk grading and query rewriting
    grader_model: str = "openai/gpt-oss-20b"
    grounding_model: str = "openai/gpt-oss-20b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    chroma_dir: Path = ROOT_DIR / "chroma_db"
    collection_name: str = "docs"
    corpus_dir: Path = ROOT_DIR / "data" / "corpus"
    feedback_db: Path = ROOT_DIR / "data" / "feedback.sqlite3"

    top_k: int = 5
    max_retries: int = 2
    chunk_size: int = 900
    chunk_overlap: int = 150

    enable_hallucination_check: bool = True
    enable_memory: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
