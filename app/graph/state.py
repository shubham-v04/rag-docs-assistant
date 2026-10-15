import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document

QueryType = Literal["conceptual", "how-to", "troubleshooting", "api-reference"]


class Source(TypedDict):
    source: str
    title: str
    header_path: str


class Turn(TypedDict):
    question: str
    answer: str


class GraphState(TypedDict):
    """State passed between the graph nodes.

    Who writes what:
      question            - set once by run_graph, never overwritten (grading and
                            generation always judge against the user's own words)
      standalone_question - analyze_query: the question rewritten so it makes sense without
                            the conversation. Equal to question when there is no history.
                            grade_documents, generate and rewrite_query use it, so they
                            never need the history itself
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
      no_answer           - False by default; the no_answer and unverified nodes set it
      provider_error      - set by grade_documents when every grading call in a round
                            fails, and by generate when the answer call fails. The
                            graph ends there and run_graph raises ProviderError, so a
                            broken provider is never reported as "not found"
      history             - earlier turns of the conversation, set once by run_graph and
                            read only by analyze_query. Grading never sees it.
      grounded            - check_grounding: True/False, or None if the check did not run
                            or its LLM call failed (the answer then passes through)
      unsupported_claims  - check_grounding; generate lists them in its stricter retry
      regenerate_count    - generate increments it each time it redoes an answer after a
                            failed grounding check; the router allows one such redo
      trace               - every node appends its own name; the reducer concatenates
                            the lists so nodes only return the entry they add
    """

    question: str
    standalone_question: str
    search_query: str
    query_type: QueryType | None
    documents: list[Document]
    relevant_documents: list[Document]
    retry_count: int
    max_retries: int
    answer: str
    sources: list[Source]
    no_answer: bool
    provider_error: str | None
    history: list[Turn]
    grounded: bool | None
    unsupported_claims: list[str]
    regenerate_count: int
    trace: Annotated[list[str], operator.add]


def initial_state(
    question: str, max_retries: int, history: list[Turn] | None = None
) -> GraphState:
    return GraphState(
        question=question,
        standalone_question=question,
        search_query=question,
        query_type=None,
        documents=[],
        relevant_documents=[],
        retry_count=0,
        max_retries=max_retries,
        answer="",
        sources=[],
        no_answer=False,
        provider_error=None,
        history=history or [],
        grounded=None,
        unsupported_claims=[],
        regenerate_count=0,
        trace=[],
    )
