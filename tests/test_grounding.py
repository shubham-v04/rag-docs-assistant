import logging
from types import SimpleNamespace

import pytest

from app.graph.builder import build_graph, decide_after_grounding, run_graph
from app.graph.nodes import UNVERIFIED_MESSAGE, GroundingResult, ProviderError
from app.graph.state import initial_state
from tests.fakes import FakeRetriever, make_deps, make_doc

QUESTION = "How do I declare an optional query parameter?"
DOC = make_doc(
    "query-params.md", "Query Parameters > Optional parameters", "q: str | None = None"
)

BASE_TRACE = [
    "analyze_query",
    "retrieve",
    "grade_documents",
    "generate",
    "check_grounding",
]


def run(deps, enable=True):
    return run_graph(
        QUESTION, graph=build_graph(deps, enable_grounding_check=enable), max_retries=0
    )


def deps_with(grounding, generate=None):
    retriever = FakeRetriever(lambda query: [DOC])
    return make_deps(
        retriever, grade=lambda v: True, grounding=grounding, generate=generate
    )


# --- routing ----------------------------------------------------------------


def state_with(grounded, regenerate_count=0):
    state = initial_state("q", 0)
    state["grounded"] = grounded
    state["regenerate_count"] = regenerate_count
    return state


@pytest.mark.parametrize(
    ("grounded", "regenerate_count", "expected"),
    [
        (True, 0, "end"),
        (None, 0, "end"),
        (False, 0, "regenerate"),
        (False, 1, "unverified"),
    ],
)
def test_decide_after_grounding(grounded, regenerate_count, expected):
    assert decide_after_grounding(state_with(grounded, regenerate_count)) == expected


# --- the loop ---------------------------------------------------------------


def test_grounded_answer_goes_straight_to_end():
    generated = []

    def generate(variables):
        generated.append(variables)
        return "Use None as the default. [source: query-params.md]"

    result = run(deps_with(lambda v: GroundingResult(grounded=True), generate))

    assert result["trace"] == BASE_TRACE
    assert result["grounded"] is True
    assert result["no_answer"] is False
    assert result["answer"].startswith("Use None")
    assert len(generated) == 1
    assert generated[0]["extra_instructions"] == ""


def test_ungrounded_answer_is_regenerated_with_the_unsupported_claims():
    verdicts = iter(
        [
            GroundingResult(
                grounded=False, unsupported_claims=["It is deprecated in v2"]
            ),
            GroundingResult(grounded=True),
        ]
    )
    generated = []

    def generate(variables):
        generated.append(variables)
        return f"answer {len(generated)}"

    result = run(deps_with(lambda v: next(verdicts), generate))

    assert result["trace"] == BASE_TRACE + ["generate", "check_grounding"]
    assert result["answer"] == "answer 2"
    assert result["grounded"] is True
    assert result["no_answer"] is False
    assert [s["source"] for s in result["sources"]] == ["query-params.md"]
    assert generated[0]["extra_instructions"] == ""
    assert "It is deprecated in v2" in generated[1]["extra_instructions"]


def test_still_ungrounded_after_one_retry_gives_an_unverified_response():
    generated = []

    def generate(variables):
        generated.append(variables)
        return "an invented answer"

    deps = deps_with(
        lambda v: GroundingResult(grounded=False, unsupported_claims=["invented"]),
        generate,
    )
    result = run(deps)

    assert result["trace"] == BASE_TRACE + ["generate", "check_grounding", "unverified"]
    assert len(generated) == 2
    assert result["no_answer"] is True
    assert result["answer"] == UNVERIFIED_MESSAGE
    assert "could not verify" in result["answer"]
    assert result["sources"] == []


def test_grounding_call_failure_passes_the_answer_through(caplog):
    def broken_checker(variables):
        raise RuntimeError("groq unavailable")

    with caplog.at_level(logging.WARNING):
        result = run(deps_with(broken_checker, lambda v: "a fine answer"))

    assert result["trace"] == BASE_TRACE
    assert result["answer"] == "a fine answer"
    assert result["no_answer"] is False
    assert result["grounded"] is None
    assert "grounding check failed" in caplog.text


def test_failed_generation_is_a_provider_error_and_skips_the_checker():
    checked = []

    def generate(variables):
        raise RuntimeError("llm down")

    deps = deps_with(
        lambda v: checked.append(v) or GroundingResult(grounded=True), generate
    )

    with pytest.raises(
        ProviderError, match="answer generation failed: RuntimeError: llm down"
    ):
        run(deps)
    assert checked == []


# --- flag -------------------------------------------------------------------


def test_flag_off_restores_the_original_flow():
    checked = []
    deps = deps_with(lambda v: checked.append(v) or GroundingResult(grounded=False))
    result = run(deps, enable=False)

    assert result["trace"] == [
        "analyze_query",
        "retrieve",
        "grade_documents",
        "generate",
    ]
    assert result["grounded"] is None
    assert result["no_answer"] is False
    assert checked == []


def test_flag_is_read_from_settings_by_default(monkeypatch):
    monkeypatch.setattr(
        "app.graph.builder.get_settings",
        lambda: SimpleNamespace(enable_hallucination_check=False, max_retries=0),
    )
    deps = deps_with(lambda v: GroundingResult(grounded=False))
    result = run_graph(QUESTION, graph=build_graph(deps), max_retries=0)

    assert result["trace"][-1] == "generate"


def test_enabling_the_check_without_a_checker_is_an_error():
    deps = deps_with(lambda v: GroundingResult(grounded=True))
    deps.grounding_checker = None

    with pytest.raises(ValueError, match="grounding_checker"):
        build_graph(deps, enable_grounding_check=True)
