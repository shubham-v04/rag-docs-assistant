from functools import lru_cache, partial
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.graph.llm import get_llm, get_structured_llm
from app.graph.nodes import (
    MAX_REGENERATIONS,
    ChunkGrade,
    GraphDeps,
    GroundingResult,
    ProviderError,
    QueryAnalysis,
    RewrittenQuery,
    analyze_query,
    check_grounding,
    generate,
    grade_documents,
    no_answer,
    retrieve,
    rewrite_query,
    unverified,
)
from app.graph.prompts import (
    ANALYZE_PROMPT,
    GENERATE_PROMPT,
    GRADE_PROMPT,
    GROUNDING_PROMPT,
    REWRITE_PROMPT,
)
from app.graph.retry import retrying
from app.graph.state import GraphState, Turn, initial_state


def decide_after_grading(
    state: GraphState,
) -> Literal["generate", "rewrite_query", "no_answer", "provider_error"]:
    if state["provider_error"]:
        return "provider_error"
    if state["relevant_documents"]:
        return "generate"
    if state["retry_count"] < state["max_retries"]:
        return "rewrite_query"
    return "no_answer"


def decide_after_grounding(
    state: GraphState,
) -> Literal["end", "regenerate", "unverified"]:
    # None means the check did not run or failed: let the answer through
    if state["grounded"] is not False:
        return "end"
    if state["regenerate_count"] < MAX_REGENERATIONS:
        return "regenerate"
    return "unverified"


def build_graph(deps: GraphDeps, enable_grounding_check: bool | None = None):
    if enable_grounding_check is None:
        enable_grounding_check = get_settings().enable_hallucination_check
    if enable_grounding_check and deps.grounding_checker is None:
        raise ValueError(
            "the grounding check is enabled but deps has no grounding_checker"
        )

    graph = StateGraph(GraphState)
    graph.add_node("analyze_query", partial(analyze_query, deps=deps))
    graph.add_node("retrieve", partial(retrieve, deps=deps))
    graph.add_node("grade_documents", partial(grade_documents, deps=deps))
    graph.add_node("rewrite_query", partial(rewrite_query, deps=deps))
    graph.add_node("generate", partial(generate, deps=deps))
    graph.add_node("no_answer", no_answer)

    graph.add_edge(START, "analyze_query")
    graph.add_edge("analyze_query", "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "no_answer": "no_answer",
            "provider_error": END,
        },
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("no_answer", END)

    if enable_grounding_check:
        graph.add_node("check_grounding", partial(check_grounding, deps=deps))
        graph.add_node("unverified", unverified)
        graph.add_edge("generate", "check_grounding")
        graph.add_conditional_edges(
            "check_grounding",
            decide_after_grounding,
            {"end": END, "regenerate": "generate", "unverified": "unverified"},
        )
        graph.add_edge("unverified", END)
    else:
        graph.add_edge("generate", END)
    return graph.compile()


def default_deps() -> GraphDeps:
    # imported here so building a graph with fake deps does not load the embedding model
    from app.ingestion.indexer import get_vectorstore

    settings = get_settings()

    def retriever(query: str, k: int):
        results = get_vectorstore().similarity_search_with_score(query, k=k)
        # chroma returns cosine distance; flip it so a higher score means closer
        return [(doc, 1 - distance) for doc, distance in results]

    # every LLM call goes through retrying(), so a 429 is waited out, not treated as a result
    return GraphDeps(
        analyzer=retrying(
            ANALYZE_PROMPT | get_structured_llm(settings.grader_model, QueryAnalysis)
        ),
        grader=retrying(
            GRADE_PROMPT | get_structured_llm(settings.grader_model, ChunkGrade)
        ),
        rewriter=retrying(
            REWRITE_PROMPT | get_structured_llm(settings.grader_model, RewrittenQuery)
        ),
        generator=retrying(
            GENERATE_PROMPT | get_llm(settings.generation_model) | StrOutputParser()
        ),
        retriever=retriever,
        top_k=settings.top_k,
        grounding_checker=retrying(
            GROUNDING_PROMPT
            | get_structured_llm(settings.grounding_model, GroundingResult)
        ),
        grading_concurrency=settings.grading_concurrency,
        grading_chunk_chars=settings.grading_chunk_chars,
    )


@lru_cache
def get_graph():
    return build_graph(default_deps())


def run_graph(
    question: str,
    graph=None,
    max_retries: int | None = None,
    history: list[Turn] | None = None,
) -> dict:
    graph = graph or get_graph()
    if max_retries is None:
        max_retries = get_settings().max_retries

    # each retry adds rewrite_query, retrieve and grade_documents to the step count,
    # and the grounding check can add a second generate and check on top
    config = {"recursion_limit": 3 * max_retries + 15}
    state = initial_state(question, max_retries, history)
    final = graph.invoke(state, config=config)
    if final["provider_error"]:
        raise ProviderError(final["provider_error"])
    return {
        "answer": final["answer"],
        "sources": final["sources"],
        "query_type": final["query_type"],
        "retry_count": final["retry_count"],
        "trace": final["trace"],
        "standalone_question": final["standalone_question"],
        "search_query": final["search_query"],
        "no_answer": final["no_answer"],
        "grounded": final["grounded"],
    }
