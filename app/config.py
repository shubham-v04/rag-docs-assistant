from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    # optional; only silences the Hugging Face "unauthenticated requests" notice and
    # raises download rate limits. The model is public, so it is not required.
    hf_token: str = ""

    chroma_dir: Path = ROOT_DIR / "chroma_db"
    collection_name: str = "docs"
    corpus_dir: Path = ROOT_DIR / "data" / "corpus"
    feedback_db: Path = ROOT_DIR / "data" / "feedback.sqlite3"

    # json_schema avoids the tool-call errors some Groq models return in function_calling
    # mode. Set STRUCTURED_OUTPUT_METHOD=function_calling to go back if a model refuses it.
    structured_output_method: Literal[
        "json_schema", "function_calling", "json_mode"
    ] = "json_schema"

    top_k: int = 5
    max_retries: int = 2
    chunk_size: int = 900
    chunk_overlap: int = 150

    # free-tier Groq limits are per minute, so grade a few chunks at a time
    grading_concurrency: int = 2
    # characters of each chunk sent to the grader; stored text and citations are untouched
    grading_chunk_chars: int = 1200
    # seconds scripts/evaluate.py waits between questions
    eval_pause: float = 5.0

    enable_hallucination_check: bool = True
    enable_memory: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()