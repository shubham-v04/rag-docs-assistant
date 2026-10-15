from typing import Annotated, Literal

from pydantic import BaseModel, Field, HttpUrl, StringConstraints, field_validator

from app.graph.state import QueryType

Question = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=3, max_length=1000)
]
SessionId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=64, pattern=r"^[\w.-]+$"
    ),
]


class QueryRequest(BaseModel):
    question: Question
    # omit it for a one-off question; reuse it to ask follow-ups
    session_id: SessionId | None = None


class SourceInfo(BaseModel):
    source: str
    title: str
    header_path: str


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    sources: list[SourceInfo]
    query_type: QueryType
    retries: int
    no_answer: bool
    session_id: str | None = None


class IngestUrlsRequest(BaseModel):
    urls: list[HttpUrl] = Field(min_length=1, max_length=10)


class IngestedItem(BaseModel):
    source: str
    chunks: int


class IngestErrorItem(BaseModel):
    source: str
    reason: str


class IngestResponse(BaseModel):
    ingested: list[IngestedItem]
    errors: list[IngestErrorItem]


class DocumentInfo(BaseModel):
    source: str
    title: str
    chunk_count: int


class FeedbackRequest(BaseModel):
    query_id: str
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)

    @field_validator("comment")
    @classmethod
    def blank_comment_is_none(cls, value: str | None) -> str | None:
        return value.strip() or None if value else None


class FeedbackResponse(BaseModel):
    feedback_id: int
    query_id: str
    rating: Literal["up", "down"]
    comment: str | None
    created_at: str
