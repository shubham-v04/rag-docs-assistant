import logging

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.config import Settings
from app.graph import retry
from app.graph.builder import build_graph, default_deps, run_graph
from app.graph.llm import get_structured_llm
from app.graph.nodes import (
    ChunkGrade,
    GroundingResult,
    QueryAnalysis,
    RewrittenQuery,
)
from tests.fakes import FakeRetriever, make_deps, make_doc
from tests.test_retry import rate_limit_error

STRUCTURED_SCHEMAS = {
    "QueryAnalysis",
    "ChunkGrade",
    "RewrittenQuery",
    "GroundingResult",
}


def settings_with(**overrides) -> Settings:
    return Settings(_env_file=None, groq_api_key="test-key", **overrides)


# --- the real ChatGroq wrapper, no network ------------------------------------


def test_json_schema_is_the_default_mode_for_structured_output(monkeypatch):
    monkeypatch.setattr("app.graph.llm.get_settings", lambda: settings_with())
    chain = get_structured_llm("openai/gpt-oss-20b", GroundingResult)

    response_format = chain.first.kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "GroundingResult"
    assert "tools" not in chain.first.kwargs


def test_function_calling_can_be_restored_from_config(monkeypatch):
    monkeypatch.setattr(
        "app.graph.llm.get_settings",
        lambda: settings_with(structured_output_method="function_calling"),
    )
    chain = get_structured_llm("openai/gpt-oss-20b", GroundingResult)

    assert "tools" in chain.first.kwargs
    assert "response_format" not in chain.first.kwargs


def test_unknown_structured_output_method_is_rejected():
    with pytest.raises(ValueError):
        Settings(_env_file=None, structured_output_method="xml")


# --- default_deps wiring, with a fake model -----------------------------------


class FakeLLM(RunnableLambda):
    """Stands in for ChatGroq: records how structured output was requested, and
    rate-limits the first call of every chain built from it."""

    def __init__(self, model_name: str, record: list):
        super().__init__(self.respond)
        self.model_name = model_name
        self.record = record
        self.calls = 0

    def respond(self, value):
        self.calls += 1
        if self.calls == 1:
            raise rate_limit_error("Please try again in 1s")
        return AIMessage(content="an answer")

    def with_structured_output(self, schema, method=None, **kwargs):
        self.record.append((self.model_name, schema.__name__, method))
        samples = {
            "QueryAnalysis": QueryAnalysis(
                standalone_question="q", search_query="q", query_type="how-to"
            ),
            "ChunkGrade": ChunkGrade(relevant=True),
            "RewrittenQuery": RewrittenQuery(search_query="q2"),
            "GroundingResult": GroundingResult(grounded=True),
        }
        calls = {"n": 0}

        def respond(value):
            calls["n"] += 1
            if calls["n"] == 1:
                raise rate_limit_error("Please try again in 1s")
            return samples[schema.__name__]

        return RunnableLambda(respond)


@pytest.fixture
def fake_models(monkeypatch):
    record: list = []
    settings = settings_with(
        grader_model="grader-model",
        generation_model="generation-model",
        grounding_model="grounding-model",
    )
    for module in ("app.graph.llm", "app.graph.builder"):
        monkeypatch.setattr(f"{module}.get_settings", lambda s=settings: s)
    monkeypatch.setattr("app.graph.llm.get_llm", lambda name: FakeLLM(name, record))
    monkeypatch.setattr("app.graph.builder.get_llm", lambda name: FakeLLM(name, record))
    # the first retry waits zero seconds
    monkeypatch.setattr(retry, "BASE_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(retry.random, "uniform", lambda low, high: 0.0)
    return record


def test_every_structured_chain_is_built_in_json_schema_mode(fake_models):
    default_deps()

    assert {schema for _, schema, _ in fake_models} == STRUCTURED_SCHEMAS
    assert {method for _, _, method in fake_models} == {"json_schema"}


def test_each_chain_uses_the_model_configured_for_its_role(fake_models):
    default_deps()

    by_schema = {schema: model for model, schema, _ in fake_models}
    assert by_schema["GroundingResult"] == "grounding-model"
    assert by_schema["QueryAnalysis"] == "grader-model"
    assert by_schema["ChunkGrade"] == "grader-model"
    assert by_schema["RewrittenQuery"] == "grader-model"


def test_the_old_mode_is_used_when_configured(fake_models, monkeypatch):
    settings = settings_with(structured_output_method="function_calling")
    for module in ("app.graph.llm", "app.graph.builder"):
        monkeypatch.setattr(f"{module}.get_settings", lambda s=settings: s)
    default_deps()

    assert {method for _, _, method in fake_models} == {"function_calling"}


def test_default_deps_wraps_every_llm_chain_in_the_retry_layer(fake_models):
    """Each fake model rate-limits its first call. If any chain were not wrapped in
    retrying(), that call would raise here."""
    deps = default_deps()

    assert (
        deps.analyzer.invoke({"question": "q", "history": "(none)"}).query_type
        == "how-to"
    )
    assert deps.grader.invoke({"question": "q", "context": "c"}).relevant is True
    assert (
        deps.rewriter.invoke({"question": "q", "failed_query": "x"}).search_query
        == "q2"
    )
    generated = deps.generator.invoke(
        {"question": "q", "context": "c", "extra_instructions": ""}
    )
    assert generated == "an answer"
    assert (
        deps.grounding_checker.invoke({"context": "c", "answer": "a"}).grounded is True
    )


def test_default_deps_passes_the_load_settings_through(monkeypatch, fake_models):
    settings = settings_with(grading_concurrency=3, grading_chunk_chars=500, top_k=7)
    for module in ("app.graph.llm", "app.graph.builder"):
        monkeypatch.setattr(f"{module}.get_settings", lambda s=settings: s)
    deps = default_deps()

    assert (deps.grading_concurrency, deps.grading_chunk_chars, deps.top_k) == (
        3,
        500,
        7,
    )


# --- the grounding check still fails open -------------------------------------

DOC = make_doc("query-params.md", "Query Parameters", "defaults: skip=0")


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError(
            "400 tool_use_failed: attempted to call tool 'groundingResult' which was not in request.tools"
        ),
        ValueError("model output did not match the schema"),
        rate_limit_error("still rate limited after the retries"),
    ],
)
def test_any_grounding_error_passes_the_answer_through_with_a_warning(error, caplog):
    def broken_checker(variables):
        raise error

    deps = make_deps(
        FakeRetriever(lambda q: [DOC]),
        grade=lambda v: True,
        generate=lambda v: "a fine answer",
        grounding=broken_checker,
    )
    with caplog.at_level(logging.WARNING):
        result = run_graph(
            "q?", graph=build_graph(deps, enable_grounding_check=True), max_retries=0
        )

    assert result["answer"] == "a fine answer"
    assert result["no_answer"] is False
    assert result["grounded"] is None
    assert "grounding check failed" in caplog.text
