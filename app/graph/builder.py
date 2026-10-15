from functools import lru_cache, partial
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.graph.llm import get_llm
from app.graph.nodes import (
    ChunkGrade,
    GraphDeps,
    QueryAnalysis,
    RewrittenQuery,
    analyze_query,
    generate,
    grade_documents,
    no_answer,
    retrieve,
    rewrite_query,
)
from app.graph.prompts import (
    ANALYZE_PROMPT,
    GENERATE_PROMPT,
    GRADE_PROMPT,
    REWRITE_PROMPT,
)
from app.graph.state import GraphState, initial_state


def decide_after_grading(
    state: GraphState,
) -> Literal["generate", "rewrite_query", "no_answer"]:
    if state["relevant_documents"]:
        return "generate"
    if state["retry_count"] < state["max_retries"]:
        return "rewrite_query"
    return "no_answer"


def build_graph(deps: GraphDeps):
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
        },
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("no_answer", END)
    return graph.compile()


def default_deps() -> GraphDeps:
    # imported here so building a graph with fake deps does not load the embedding model
    from app.ingestion.indexer import get_vectorstore

    settings = get_settings()
    fast_llm = get_llm(settings.grader_model)
    answer_llm = get_llm(settings.generation_model)

    def retriever(query: str, k: int):
        results = get_vectorstore().similarity_search_with_score(query, k=k)
        # chroma returns cosine distance; flip it so a higher score means closer
        return [(doc, 1 - distance) for doc, distance in results]

    return GraphDeps(
        analyzer=ANALYZE_PROMPT | fast_llm.with_structured_output(QueryAnalysis),
        grader=GRADE_PROMPT | fast_llm.with_structured_output(ChunkGrade),
        rewriter=REWRITE_PROMPT | fast_llm.with_structured_output(RewrittenQuery),
        generator=GENERATE_PROMPT | answer_llm | StrOutputParser(),
        retriever=retriever,
        top_k=settings.top_k,
    )


@lru_cache
def get_graph():
    return build_graph(default_deps())


def run_graph(question: str, graph=None, max_retries: int | None = None) -> dict:
    graph = graph or get_graph()
    if max_retries is None:
        max_retries = get_settings().max_retries

    # each retry adds rewrite_query, retrieve and grade_documents to the step count
    config = {"recursion_limit": 3 * max_retries + 10}
    final = graph.invoke(initial_state(question, max_retries), config=config)
    return {
        "answer": final["answer"],
        "sources": final["sources"],
        "query_type": final["query_type"],
        "retry_count": final["retry_count"],
        "trace": final["trace"],
        "search_query": final["search_query"],
        "no_answer": final["no_answer"],
    }
