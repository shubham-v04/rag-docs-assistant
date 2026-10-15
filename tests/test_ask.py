import sys

import pytest

from app.graph.nodes import ProviderError
from scripts import ask


def run_ask(monkeypatch, run_graph):
    monkeypatch.setattr(sys, "argv", ["ask.py", "How do I declare a query parameter?"])
    monkeypatch.setattr(ask, "run_graph", run_graph)
    ask.main()


def test_provider_errors_are_printed_as_errors(monkeypatch):
    def failing(question):
        raise ProviderError("every grading call failed: NotFoundError: model_not_found")

    with pytest.raises(SystemExit) as exc:
        run_ask(monkeypatch, failing)

    assert (
        str(exc.value)
        == "error: every grading call failed: NotFoundError: model_not_found"
    )


def test_a_normal_result_is_printed(monkeypatch, capsys):
    result = {
        "answer": "Use a default of None.",
        "sources": [
            {"source": "query-params.md", "title": "T", "header_path": "A > B"}
        ],
        "query_type": "how-to",
        "retry_count": 1,
        "trace": ["analyze_query", "generate"],
        "standalone_question": "How do I declare a query parameter?",
        "search_query": "declare query parameter",
        "no_answer": False,
        "grounded": True,
    }
    run_ask(monkeypatch, lambda question: result)

    out = capsys.readouterr().out
    assert "analyze_query -> generate" in out
    assert "retries:      1" in out
    assert "Use a default of None." in out
    assert "query-params.md | A > B" in out
