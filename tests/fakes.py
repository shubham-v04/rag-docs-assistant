from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app.graph.nodes import (
    ChunkGrade,
    GraphDeps,
    GroundingResult,
    QueryAnalysis,
    RewrittenQuery,
)


def make_doc(source: str, header_path: str, text: str, index: int = 0) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "title": source.removesuffix(".md").replace("-", " ").title(),
            "header_path": header_path,
            "chunk_index": index,
        },
    )


class FakeRetriever:
    """Returns docs chosen by the query and remembers every query it saw."""

    def __init__(self, docs_for_query: Callable[[str], list[Document]]):
        self.docs_for_query = docs_for_query
        self.calls: list[tuple[str, int]] = []

    def __call__(self, query: str, k: int) -> list[tuple[Document, float]]:
        self.calls.append((query, k))
        return [(doc, 0.9) for doc in self.docs_for_query(query)]

    @property
    def queries(self) -> list[str]:
        return [query for query, _ in self.calls]


def make_deps(
    retriever: FakeRetriever,
    grade: Callable[[dict], bool],
    analyze: Callable[[dict], QueryAnalysis] | None = None,
    rewrite: Callable[[dict], RewrittenQuery] | None = None,
    generate: Callable[[dict], str] | None = None,
    grounding: Callable[[dict], GroundingResult] | None = None,
    top_k: int = 3,
) -> GraphDeps:
    return GraphDeps(
        analyzer=RunnableLambda(
            analyze
            or (
                lambda v: QueryAnalysis(
                    standalone_question=v["question"],
                    search_query="expanded query",
                    query_type="how-to",
                )
            )
        ),
        grader=RunnableLambda(lambda v: ChunkGrade(relevant=grade(v))),
        rewriter=RunnableLambda(
            rewrite
            or (lambda v: RewrittenQuery(search_query=v["failed_query"] + " again"))
        ),
        generator=RunnableLambda(generate or (lambda v: "an answer [source: a.md]")),
        retriever=retriever,
        top_k=top_k,
        grounding_checker=RunnableLambda(
            grounding or (lambda v: GroundingResult(grounded=True))
        ),
    )
