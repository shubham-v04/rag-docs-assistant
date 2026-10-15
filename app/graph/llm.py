from langchain_groq import ChatGroq

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
