import threading
import time

import pytest
from langchain_core.runnables import RunnableLambda

from app.graph.builder import build_graph, run_graph
from app.graph.nodes import ChunkGrade, ProviderError
from app.graph.retry import retrying
from app.graph.state import initial_state
from tests.fakes import FakeRetriever, make_deps, make_doc
from tests.test_retry import rate_limit_error

QUESTION = "How do I declare an optional query parameter?"

GOOD = make_doc("query-params.md", "Query Parameters > Defaults", "defaults: skip=0", 0)
OTHER = make_doc("request-body.md", "Request Body", "defaults: a pydantic model", 1)


def run(deps, max_retries=0):
    graph = build_graph(deps, enable_grounding_check=False)
    return run_graph(QUESTION, graph=graph, max_retries=max_retries)


def flaky_grader(fail_for, sleeps, failures_per_chunk=1):
    """A retrying grader. Chunks whose text contains `fail_for` get a 429 first."""
    attempts: dict[str, int] = {}
    lock = threading.Lock()

    def grade(variables):
        context = variables["context"]
        with lock:
            attempts[context] = attempts.get(context, 0) + 1
            count = attempts[context]
        if fail_for in context and count <= failures_per_chunk:
            raise rate_limit_error("Please try again in 1s")
        return ChunkGrade(relevant=True)

    chain = retrying(RunnableLambda(grade), sleep=sleeps.append)
    chain.attempts = attempts
    return chain


# --- rate limits during grading -----------------------------------------------


def test_a_rate_limited_grading_call_is_retried_and_the_chunk_is_graded():
    sleeps = []
    deps = make_deps(FakeRetriever(lambda q: [GOOD, OTHER]), grade=lambda v: True)
    deps.grader = flaky_grader("query-params.md", sleeps)

    result = run(deps)

    # the 429 chunk was graded relevant on the second attempt, not dropped as irrelevant
    assert result["no_answer"] is False
    assert [s["source"] for s in result["sources"]] == [
        "query-params.md",
        "request-body.md",
    ]
    assert len(sleeps) == 1
    assert (
        deps.grader.attempts[
            next(c for c in deps.grader.attempts if "query-params.md" in c)
        ]
        == 2
    )


def test_a_chunk_that_stays_rate_limited_counts_as_failed_but_the_run_goes_on():
    sleeps = []
    deps = make_deps(FakeRetriever(lambda q: [GOOD, OTHER]), grade=lambda v: True)
    deps.grader = flaky_grader("query-params.md", sleeps, failures_per_chunk=100)

    result = run(deps)

    assert result["no_answer"] is False
    assert [s["source"] for s in result["sources"]] == ["request-body.md"]
    assert len(sleeps) == 4  # five attempts, four waits


def test_every_chunk_staying_rate_limited_is_a_provider_error():
    sleeps = []
    deps = make_deps(FakeRetriever(lambda q: [GOOD, OTHER]), grade=lambda v: True)
    deps.grader = flaky_grader("defaults", sleeps, failures_per_chunk=100)

    with pytest.raises(
        ProviderError, match="every grading call failed: RateLimitError"
    ):
        run(deps, max_retries=2)

    assert len(sleeps) == 8  # four waits for each of the two chunks


def test_every_llm_step_survives_a_single_rate_limit():
    """analysis, grading, rewrite, generation and the grounding check each get one 429."""
    sleeps = []
    seen = set()
    lock = threading.Lock()

    def flaky_once(chain):
        def call(value):
            key = (id(chain), repr(value))
            with lock:
                first_time = key not in seen
                seen.add(key)
            if first_time:
                raise rate_limit_error("Please try again in 1s")
            return chain.invoke(value)

        return retrying(RunnableLambda(call), sleep=sleeps.append)

    retriever = FakeRetriever(lambda q: [GOOD] if q.endswith("again") else [OTHER])
    deps = make_deps(retriever, grade=lambda v: "query-params.md" in v["context"])
    deps.analyzer = flaky_once(deps.analyzer)
    deps.grader = flaky_once(deps.grader)
    deps.rewriter = flaky_once(deps.rewriter)
    deps.generator = flaky_once(deps.generator)
    deps.grounding_checker = flaky_once(deps.grounding_checker)

    graph = build_graph(deps, enable_grounding_check=True)
    result = run_graph(QUESTION, graph=graph, max_retries=1)

    assert result["trace"] == [
        "analyze_query",
        "retrieve",
        "grade_documents",
        "rewrite_query",
        "retrieve",
        "grade_documents",
        "generate",
        "check_grounding",
    ]
    assert result["no_answer"] is False
    # analysis, two grading rounds, rewrite, generation and grounding: one 429 each
    assert len(sleeps) == 6


# --- concurrency --------------------------------------------------------------


def peak_concurrency(grading_concurrency: int, chunks: int = 6) -> int:
    in_flight = 0
    peak = 0
    lock = threading.Lock()

    def grade(variables):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.05)
        with lock:
            in_flight -= 1
        return ChunkGrade(relevant=True)

    docs = [make_doc("a.md", "A", f"chunk {i}", i) for i in range(chunks)]
    deps = make_deps(FakeRetriever(lambda q: docs), grade=lambda v: True, top_k=chunks)
    deps.grader = retrying(RunnableLambda(grade))
    deps.grading_concurrency = grading_concurrency
    run(deps)
    return peak


def test_grading_runs_at_most_the_configured_number_of_calls_at_once():
    assert peak_concurrency(1) == 1
    assert peak_concurrency(2) == 2
    assert peak_concurrency(3) == 3


def test_default_grading_concurrency_is_two():
    deps = make_deps(FakeRetriever(lambda q: []), grade=lambda v: True)
    assert deps.grading_concurrency == 2


# --- truncation ---------------------------------------------------------------


LONG_TEXT = "".join(
    f"line {i:04d} of a long chunk. " for i in range(100)
)  # ~3400 chars


def test_only_the_grading_prompt_sees_a_shortened_chunk():
    long_doc = make_doc("query-params.md", "Query Parameters", LONG_TEXT, 0)
    short_doc = make_doc("request-body.md", "Request Body", "a short chunk", 1)
    graded, generated = [], []

    deps = make_deps(
        FakeRetriever(lambda q: [long_doc, short_doc]),
        grade=lambda v: graded.append(v) or True,
        generate=lambda v: generated.append(v) or "an answer",
    )
    graph = build_graph(deps, enable_grounding_check=False)
    final = graph.invoke(initial_state(QUESTION, 0))

    long_context = next(
        v["context"] for v in graded if "query-params.md" in v["context"]
    )
    short_context = next(
        v["context"] for v in graded if "request-body.md" in v["context"]
    )
    assert LONG_TEXT not in long_context
    assert LONG_TEXT[:1200].rstrip() in long_context
    assert long_context.endswith("[truncated]")
    assert "[truncated]" not in short_context

    # generation, the stored chunks and the citations keep the full text
    assert LONG_TEXT in generated[0]["context"]
    assert final["relevant_documents"][0].page_content == LONG_TEXT
    assert final["sources"][0] == {
        "source": "query-params.md",
        "title": long_doc.metadata["title"],
        "header_path": "Query Parameters",
    }


def test_the_truncation_length_is_configurable():
    doc = make_doc("a.md", "A", "x" * 500, 0)
    graded = []
    deps = make_deps(
        FakeRetriever(lambda q: [doc]), grade=lambda v: graded.append(v) or True
    )
    deps.grading_chunk_chars = 100
    run(deps)

    assert "x" * 100 in graded[0]["context"]
    assert "x" * 101 not in graded[0]["context"]
