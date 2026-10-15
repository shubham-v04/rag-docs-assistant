import logging
import re
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from pydantic import ValidationError

from app.api.schemas import (
    DocumentInfo,
    FeedbackRequest,
    FeedbackResponse,
    IngestedItem,
    IngestErrorItem,
    IngestResponse,
    IngestUrlsRequest,
    QueryRequest,
    QueryResponse,
)
from app.ingestion.indexer import list_documents
from app.services.feedback import FeedbackStore, UnknownQueryError
from app.services.ingestion import ingest_sources
from app.services.memory import ConversationMemory
from app.services.query import IndexEmptyError, QueryFailedError, answer_question

logger = logging.getLogger(__name__)

router = APIRouter()

# Endpoints are plain `def` on purpose: FastAPI runs them in a thread pool, so the
# blocking graph, embedding and sqlite calls do not stall the event loop.


def get_graph(request: Request):
    return request.app.state.graph


def get_feedback_store(request: Request) -> FeedbackStore:
    return request.app.state.feedback


def get_memory(request: Request) -> ConversationMemory | None:
    return request.app.state.memory


GraphDep = Annotated[Any, Depends(get_graph)]
StoreDep = Annotated[FeedbackStore, Depends(get_feedback_store)]
MemoryDep = Annotated[ConversationMemory | None, Depends(get_memory)]


@router.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "graph_ready": request.app.state.graph is not None}


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, graph: GraphDep, store: StoreDep, memory: MemoryDep):
    # checked here, not in the dependency, so a malformed body still gets a 422 first
    if graph is None:
        raise HTTPException(503, "The language model is not configured on the server.")
    try:
        result = answer_question(body.question, graph, store, memory, body.session_id)
    except IndexEmptyError:
        raise HTTPException(
            503, "The index is empty. Ingest documents first (POST /ingest)."
        )
    except QueryFailedError:
        raise HTTPException(
            502, "The answer could not be generated. Please try again later."
        )
    return {**result, "retries": result["retry_count"]}


@router.post(
    "/ingest",
    status_code=201,
    response_model=IngestResponse,
    responses={400: {"model": IngestResponse}, 413: {"model": IngestResponse}},
)
def ingest(
    response: Response,
    files: Annotated[list[UploadFile] | None, File()] = None,
    urls: Annotated[
        str | None, Form(description="URLs separated by spaces, commas or newlines")
    ] = None,
):
    uploads = [(f.filename, f.file) for f in files or [] if f.filename]
    url_list = parse_urls(urls)
    if not uploads and not url_list:
        raise HTTPException(400, "Provide at least one file or URL.")

    result = ingest_sources(uploads, url_list)
    if not result.ingested:
        response.status_code = result.errors[0].status_code
    return IngestResponse(
        ingested=[IngestedItem(source=s, chunks=n) for s, n in result.ingested.items()],
        errors=[
            IngestErrorItem(source=e.source, reason=e.reason) for e in result.errors
        ],
    )


def parse_urls(raw: str | None) -> list[str]:
    candidates = [part for part in re.split(r"[\s,]+", raw or "") if part]
    if not candidates:
        return []
    try:
        return [str(url) for url in IngestUrlsRequest(urls=candidates).urls]
    except ValidationError as exc:
        problems = "; ".join(f"{e['msg']}" for e in exc.errors())
        raise HTTPException(422, f"Invalid urls: {problems}")


@router.get("/documents", response_model=list[DocumentInfo])
def documents():
    return list_documents()


@router.post("/feedback", status_code=201, response_model=FeedbackResponse)
def feedback(body: FeedbackRequest, store: StoreDep):
    try:
        record = store.add_feedback(body.query_id, body.rating, body.comment)
    except UnknownQueryError:
        raise HTTPException(404, f"Unknown query_id '{body.query_id}'.")
    return FeedbackResponse(
        feedback_id=record.id,
        query_id=record.query_id,
        rating=record.rating,
        comment=record.comment,
        created_at=record.created_at,
    )
