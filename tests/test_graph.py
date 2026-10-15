import logging

import pytest

from app.graph.builder import build_graph, run_graph
from app.graph.nodes import (
    NO_ANSWER_MESSAGE,
    ProviderError,
    QueryAnalysis,
    RewrittenQuery,
)
from tests.fakes import FakeRetriever, make_deps, make_doc

QUESTION = "How do I declare an optional query parameter?"

DEFAULTS_A = make_doc(
    "query-params.md", "Query Parameters > Defaults", "defaults: skip=0", 0
)
DEFAULTS_B = make_doc(
    "query-params.md", "Query Parameters > Defaults", "defaults: limit=10", 1
)
BODY = make_doc("request-body.md", "Request Body", "declare a pydantic model", 2)

ANALYZE_EXPANDED = QueryAnalysis(
    standalone_question=QUESTION,
    search_query="optional query parameter default None",
    query_type="how-to",
)


def mentions_defaults(variables: dict) -> bool:
    return "defaults" in variables["context"]


def test_happy_path_generates_with_deduplicated_sources():
    retriever = FakeRetriever(lambda query: [DEFAULTS_A, DEFAULTS_B, BODY])
    generated: list[dict] = []
    graded: list[dict] = []

    def grade(variables: dict) -> bool:
        graded.append(variables)
        return mentions_defaults(variables)

    def generate(variables: dict) -> str:
        generated.append(variables)
        return "  Set a default of None. [source: query-params.md]  "

    deps = make_deps(
        retriever, grade, analyze=lambda v: ANALYZE_EXPANDED, generate=generate, top_k=3
    )
    result = run_graph(
        QUESTION, graph=build_graph(deps, enable_grounding_check=False), max_retries=2
    )

    assert result["trace"] == [
        "analyze_query",
        "retrieve",
        "grade_documents",
        "generate",
    ]
    assert result["query_type"] == "how-to"
    assert result["retry_count"] == 0
    assert result["no_answer"] is False
    assert result["answer"] == "Set a default of None. [source: query-params.md]"
    # two chunks from the same section collapse into one source
    assert result["sources"] == [
        {
            "source": "query-params.md",
            "title": "Query Params",
            "header_path": "Query Parameters > Defaults",
        }
    ]
    # retrieval uses the rewritten query, grading and generation use the user's question
    assert retriever.calls == [("optional query parameter default None", 3)]
    assert all(v["question"] == QUESTION for v in graded)
    assert generated[0]["question"] == QUESTION
    assert "request-body.md" not in generated[0]["context"]


def test_retries_until_limit_then_gives_up():
    retriever = FakeRetriever(lambda query: [BODY])
    generated: list[dict] = []
    rewrites = iter(["first rewrite", "second rewrite"])

    deps = make_deps(
        retriever,
        grade=lambda v: False,
        analyze=lambda v: QueryAnalysis(
            standalone_question=v["question"],
            search_query="first try",
            query_type="conceptual",
        ),
        rewrite=lambda v: RewrittenQuery(search_query=next(rewrites)),
        generate=lambda v: generated.append(v) or "should not run",
    )
    result = run_graph(
        "How do I train a CNN in PyTorch?",
        graph=build_graph(deps, enable_grounding_check=False),
        max_retries=2,
    )

    assert result["trace"] == [
        "analyze_query",
        "retrieve",
        "grade_documents",
        "rewrite_query",
        "retrieve",
        "grade_documents",
        "rewrite_query",
        "retrieve",
        "grade_documents",
        "no_answer",
    ]
    assert retriever.queries == ["first try", "first rewrite", "second rewrite"]
    assert result["retry_count"] == 2
    assert result["no_answer"] is True
    assert result["answer"] == NO_ANSWER_MESSAGE
    assert result["sources"] == []
    assert generated == []


def test_zero_retries_gives_up_after_first_attempt():
    retriever = FakeRetriever(lambda query: [BODY])
    deps = make_deps(retriever, grade=lambda v: False)
    result = run_graph(
        QUESTION, graph=build_graph(deps, enable_grounding_check=False), max_retries=0
    )

    assert len(retriever.calls) == 1
    assert result["trace"][-1] == "no_answer"
    assert result["retry_count"] == 0


def test_rewritten_query_can_recover_and_reach_generate():
    def docs_for_query(query: str):
        return [DEFAULTS_A] if query == "better query" else [BODY]

    retriever = FakeRetriever(docs_for_query)
    deps = make_deps(
        retriever,
        grade=mentions_defaults,
        rewrite=lambda v: RewrittenQuery(search_query="better query"),
    )
    result = run_graph(
        QUESTION, graph=build_graph(deps, enable_grounding_check=False), max_retries=2
    )

    assert result["trace"] == [
        "analyze_query",
        "retrieve",
        "grade_documents",
        "rewrite_query",
        "retrieve",
        "grade_documents",
        "generate",
    ]
    assert result["retry_count"] == 1
    assert result["search_query"] == "better query"
    assert result["no_answer"] is False
    assert [s["source"] for s in result["sources"]] == ["query-params.md"]


def test_chunk_that_fails_grading_is_dropped_not_fatal(caplog):
    retriever = FakeRetriever(lambda query: [DEFAULTS_A, BODY])

    def grade(variables: dict) -> bool:
        if "request-body.md" in variables["context"]:
            raise ValueError("could not parse model output")
        return True

    deps = make_deps(retriever, grade=grade)
    with caplog.at_level(logging.WARNING):
        result = run_graph(
            QUESTION,
            graph=build_graph(deps, enable_grounding_check=False),
            max_retries=0,
        )

    assert result["trace"][-1] == "generate"
    assert [s["source"] for s in result["sources"]] == ["query-params.md"]
    assert "could not grade request-body.md#2" in caplog.text


def test_failed_query_analysis_falls_back_to_the_question():
    def broken_analyzer(variables: dict):
        raise RuntimeError("llm unavailable")

    retriever = FakeRetriever(lambda query: [DEFAULTS_A])
    deps = make_deps(retriever, grade=lambda v: True, analyze=broken_analyzer)
    result = run_graph(
        QUESTION, graph=build_graph(deps, enable_grounding_check=False), max_retries=0
    )

    assert retriever.queries == [QUESTION]
    assert result["query_type"] == "conceptual"
    assert result["trace"][-1] == "generate"


# --- provider failures are not "not found" ----------------------------------


def failing_grader(variables: dict) -> bool:
    raise RuntimeError("Error code: 404 - model_not_found")


def test_every_grading_call_failing_is_a_provider_error_not_a_no_answer():
    retriever = FakeRetriever(lambda query: [DEFAULTS_A, BODY])
    generated: list[dict] = []
    deps = make_deps(
        retriever, grade=failing_grader, generate=lambda v: generated.append(v) or "x"
    )

    with pytest.raises(ProviderError) as exc:
        run_graph(
            QUESTION,
            graph=build_graph(deps, enable_grounding_check=False),
            max_retries=2,
        )

    assert "every grading call failed" in str(exc.value)
    assert "model_not_found" in str(exc.value)
    # no point retrying a broken provider
    assert len(retriever.calls) == 1
    assert generated == []


def test_unparseable_grades_for_every_chunk_are_also_a_provider_error():
    from langchain_core.runnables import RunnableLambda

    deps = make_deps(
        FakeRetriever(lambda query: [DEFAULTS_A, BODY]), grade=lambda v: True
    )
    deps.grader = RunnableLambda(lambda v: None)

    with pytest.raises(ProviderError, match="could not be parsed"):
        run_graph(
            QUESTION,
            graph=build_graph(deps, enable_grounding_check=False),
            max_retries=2,
        )


def test_one_failed_chunk_among_graded_ones_is_just_irrelevant():
    def grade(variables: dict) -> bool:
        if "request-body.md" in variables["context"]:
            raise ValueError("bad output")
        return False

    deps = make_deps(FakeRetriever(lambda query: [DEFAULTS_A, BODY]), grade=grade)
    result = run_graph(
        QUESTION, graph=build_graph(deps, enable_grounding_check=False), max_retries=0
    )

    assert result["no_answer"] is True
    assert result["answer"] == NO_ANSWER_MESSAGE


def test_provider_failure_in_a_later_round_is_still_reported():
    rounds = {"count": 0}

    def grade(variables: dict) -> bool:
        if rounds["count"] >= 2:
            raise RuntimeError("provider went down")
        rounds["count"] += 1
        return False

    retriever = FakeRetriever(lambda query: [DEFAULTS_A, BODY])
    deps = make_deps(retriever, grade=grade)

    with pytest.raises(ProviderError, match="provider went down"):
        run_graph(
            QUESTION,
            graph=build_graph(deps, enable_grounding_check=False),
            max_retries=2,
        )
    assert len(retriever.calls) == 2


def test_an_empty_retrieval_is_not_a_provider_error():
    deps = make_deps(FakeRetriever(lambda query: []), grade=failing_grader)
    result = run_graph(
        QUESTION, graph=build_graph(deps, enable_grounding_check=False), max_retries=1
    )

    assert result["no_answer"] is True
    assert result["retry_count"] == 1
