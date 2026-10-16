from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from pydantic import BaseModel

from app.config import get_settings


def get_llm(model_name: str) -> ChatGroq:
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return ChatGroq(
        model=model_name,
        temperature=0,
        api_key=settings.groq_api_key,
        timeout=30,
        max_retries=2,
    )


def get_structured_llm(model_name: str, schema: type[BaseModel]) -> Runnable:
    method = get_settings().structured_output_method
    return get_llm(model_name).with_structured_output(schema, method=method)
