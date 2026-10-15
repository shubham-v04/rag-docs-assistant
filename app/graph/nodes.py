import logging
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from app.graph.state import GraphState, QueryType, Source, Turn

logger = logging.getLogger(__name__)

DEFAULT_QUERY_TYPE: QueryType = "conceptual"
GRADER_CONCURRENCY = 5
MAX_REGENERATIONS = 1
HISTORY_ANSWER_CHARS = 600

NO_ANSWER_MESSAGE = (
    "I could not find this in the indexed documentation. "
    "Try rephrasing the question, or ingest documents that cover the topic."
)
UNVERIFIED_MESSAGE = (
    "I found documentation related to your question, but I could not verify an answer "
    "against it, so I am not going to guess. Try rephrasing or narrowing the question."
)
GENERATION_ERROR_MESSAGE = "I found relevant documentation but could not generate an answer right now. Please try again."


class QueryAnalysis(BaseModel):
    search_query: str = Field(
        description="Standalone search query for the documentation"
    )
    query_type: QueryType


class RewrittenQuery(BaseModel):
    search_query: str = Field(
        description="A new search query, different from the failed one"
    )


class ChunkGrade(BaseModel):
    relevant: bool = Field(description="True if the excerpt helps answer the question")


class GroundingResult(BaseModel):
    grounded: bool = Field(
        description="True if every claim is supported by the excerpts"
    )
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Claims the excerpts do not support"
    )


# Takes a query and k, returns (chunk, similarity) pairs with higher meaning closer.
Retriever = Callable[[str, int], list[tuple[Document, float]]]


@dataclass
class GraphDeps:
    """Everything the nodes need from outside, so tests can swap in fakes.

    Each chain takes a dict of prompt variables: the analyzer, rewriter and grader
    return their pydantic models, the generator returns the answer text.
    """

    analyzer: Runnable
    grader: Runnable
    rewriter: Runnable
    generator: Runnable
    retriever: Retriever
    top_k: int
    # only needed when the hallucination check is switched on
    grounding_checker: Runnable | None = None


def analyze_query(state: GraphState, deps: GraphDeps) -> dict:
    question = state["question"]
    try:
        result = deps.analyzer.invoke(
            {"question": question, "history": format_history(state["history"])}
        )
        search_query = result.search_query.strip() or question
        query_type = result.query_type
    except Exception:
        # search with the raw question rather than failing the whole request
        logger.warning(
            "query analysis failed, using the original question", exc_info=True
        )
        search_query, query_type = question, DEFAULT_QUERY_TYPE
    return {
        "search_query": search_query,
        "query_type": query_type,
        "trace": ["analyze_query"],
    }


def retrieve(state: GraphState, deps: GraphDeps) -> dict:
    results = deps.retriever(state["search_query"], deps.top_k)
    documents = [
        Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "score": round(score, 4)},
        )
        for doc, score in results
    ]
    return {"documents": documents, "trace": ["retrieve"]}


def grade_documents(state: GraphState, deps: GraphDeps) -> dict:
    documents = state["documents"]
    inputs = [
        {"question": state["question"], "context": format_chunk(doc)}
        for doc in documents
    ]
    grades = deps.grader.batch(
        inputs, config={"max_concurrency": GRADER_CONCURRENCY}, return_exceptions=True
    )

    relevant: list[Document] = []
    for doc, grade in zip(documents, grades):
        if not isinstance(grade, ChunkGrade):
            # an unparseable or failed grade must not let a chunk through
            logger.warning(
                "could not grade %s, treating as irrelevant: %r",
                chunk_label(doc),
                grade,
            )
        elif grade.relevant:
            relevant.append(doc)
    return {"relevant_documents": relevant, "trace": ["grade_documents"]}


def rewrite_query(state: GraphState, deps: GraphDeps) -> dict:
    failed = state["search_query"]
    try:
        result = deps.rewriter.invoke(
            {"question": state["question"], "failed_query": failed}
        )
        new_query = result.search_query.strip() or failed
    except Exception:
        logger.warning(
            "query rewrite failed, keeping the previous query", exc_info=True
        )
        new_query = failed
    if new_query.lower() == failed.lower():
        logger.warning("rewrite did not change the query: %r", failed)
    # the count goes up even if the query did not change, so the retry loop always ends
    return {
        "search_query": new_query,
        "retry_count": state["retry_count"] + 1,
        "trace": ["rewrite_query"],
    }


def generate(state: GraphState, deps: GraphDeps) -> dict:
    documents = state["relevant_documents"]
    context = "\n\n---\n\n".join(format_chunk(doc) for doc in documents)
    # a failed grounding check sends us back here with the claims to avoid
    regenerating = state["grounded"] is False
    extra = stricter_instructions(state["unsupported_claims"]) if regenerating else ""
    try:
        answer = deps.generator.invoke(
            {
                "question": state["question"],
                "context": context,
                "extra_instructions": extra,
            }
        )
    except Exception:
        logger.exception("answer generation failed")
        return {
            "answer": GENERATION_ERROR_MESSAGE,
            "sources": [],
            "no_answer": True,
            "trace": ["generate"],
        }
    return {
        "answer": answer.strip(),
        "sources": unique_sources(documents),
        "no_answer": False,
        "regenerate_count": state["regenerate_count"] + int(regenerating),
        "trace": ["generate"],
    }


def check_grounding(state: GraphState, deps: GraphDeps) -> dict:
    if state["no_answer"]:
        # generation already failed, there is no answer to verify
        return {"grounded": None, "trace": ["check_grounding"]}

    context = "\n\n---\n\n".join(
        format_chunk(doc) for doc in state["relevant_documents"]
    )
    try:
        result = deps.grounding_checker.invoke(
            {"context": context, "answer": state["answer"]}
        )
    except Exception:
        # a broken checker should not block an answer the user is waiting for
        logger.warning(
            "grounding check failed, passing the answer through", exc_info=True
        )
        return {"grounded": None, "trace": ["check_grounding"]}
    return {
        "grounded": result.grounded,
        "unsupported_claims": result.unsupported_claims,
        "trace": ["check_grounding"],
    }


def unverified(state: GraphState) -> dict:
    return {
        "answer": UNVERIFIED_MESSAGE,
        "sources": [],
        "no_answer": True,
        "trace": ["unverified"],
    }


def no_answer(state: GraphState) -> dict:
    return {
        "answer": NO_ANSWER_MESSAGE,
        "sources": [],
        "no_answer": True,
        "trace": ["no_answer"],
    }


def stricter_instructions(claims: list[str]) -> str:
    listed = "\n".join(f"- {claim}" for claim in claims) or "- (not specified)"
    return (
        "Your previous answer contained claims the excerpts do not support:\n"
        f"{listed}\n"
        "Write the answer again. Remove or correct those claims and state only what "
        "the excerpts say."
    )


def format_history(history: list[Turn]) -> str:
    if not history:
        return "(none)"
    return "\n".join(
        f"User: {turn['question']}\nAssistant: {turn['answer'][:HISTORY_ANSWER_CHARS]}"
        for turn in history
    )


def format_chunk(doc: Document) -> str:
    header = f"{doc.metadata['source']} | {doc.metadata['header_path']}".rstrip(" |")
    return f"[source: {header}]\n{doc.page_content}"


def chunk_label(doc: Document) -> str:
    return f"{doc.metadata['source']}#{doc.metadata['chunk_index']}"


def unique_sources(documents: list[Document]) -> list[Source]:
    seen: set[tuple[str, str]] = set()
    sources: list[Source] = []
    for doc in documents:
        meta = doc.metadata
        key = (meta["source"], meta["header_path"])
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "source": meta["source"],
                    "title": meta["title"],
                    "header_path": meta["header_path"],
                }
            )
    return sources
