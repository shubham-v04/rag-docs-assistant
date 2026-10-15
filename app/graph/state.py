import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document

QueryType = Literal["conceptual", "how-to", "troubleshooting", "api-reference"]


class Source(TypedDict):
    source: str
    title: str
    header_path: str


class GraphState(TypedDict):
    """State passed between the graph nodes.

    Who writes what:
      question            - set once by run_graph, never overwritten (grading and
                            generation always judge against the user's own words)
      search_query        - analyze_query sets it, rewrite_query replaces it on a retry
      query_type          - analyze_query
      documents           - retrieve (replaced on every retrieval, scores in metadata)
      relevant_documents  - grade_documents (the subset of documents that passed)
      retry_count         - starts at 0, only rewrite_query increments it, so it equals
                            the number of rewrites done; the router compares it with
                            max_retries, which means a question gets 1 + max_retries
                            retrieval attempts at most
      max_retries         - set once by run_graph from config
      answer, sources     - generate, or no_answer when nothing relevant was found
      no_answer           - False by default; the no_answer node (or a failed
                            generation) sets it to True
      trace               - every node appends its own name; the reducer concatenates
                            the lists so nodes only return the entry they add
    """

    question: str
    search_query: str
    query_type: QueryType | None
    documents: list[Document]
    relevant_documents: list[Document]
    retry_count: int
    max_retries: int
    answer: str
    sources: list[Source]
    no_answer: bool
    trace: Annotated[list[str], operator.add]


def initial_state(question: str, max_retries: int) -> GraphState:
    return GraphState(
        question=question,
        search_query=question,
        query_type=None,
        documents=[],
        relevant_documents=[],
        retry_count=0,
        max_retries=max_retries,
        answer="",
        sources=[],
        no_answer=False,
        trace=[],
    )
