import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import evaluate

ROOT = Path(__file__).resolve().parent.parent

ANSWERABLE = {
    "question": "How do I declare an optional query parameter?",
    "expect": "answer",
    "expected_source": "query-params.md",
}
OUT_OF_SCOPE = {"question": "How do I train a CNN in PyTorch?", "expect": "no_answer"}


def graph_output(sources=(), no_answer=False, retries=0) -> dict:
    """Shaped like what run_graph returns."""
    return {
        "answer": "an answer" if not no_answer else "I could not find this",
        "sources": [{"source": s, "title": "T", "header_path": "H"} for s in sources],
        "query_type": "how-to",
        "retry_count": retries,
        "trace": ["analyze_query", "retrieve"],
        "search_query": "q",
        "standalone_question": "q",
        "no_answer": no_answer,
        "grounded": None,
    }


# --- the shipped question set -----------------------------------------------


def test_shipped_questions_are_valid_and_point_at_real_corpus_files():
    items = evaluate.load_questions(ROOT / "data" / "eval_questions.json")

    assert len(items) == 10
    assert sum(i["expect"] == "answer" for i in items) == 7
    assert sum(i["expect"] == "no_answer" for i in items) == 3
    assert len({i["question"] for i in items}) == 10
    for item in items:
        if item["expect"] == "answer":
            assert (ROOT / "data" / "corpus" / item["expected_source"]).exists()


@pytest.mark.parametrize(
    ("item", "message"),
    [
        ({"question": "q", "expect": "maybe"}, "expect must be"),
        ({"question": "q", "expect": "answer"}, "expected_source"),
        ({"question": "", "expect": "no_answer"}, "no text"),
    ],
)
def test_load_questions_rejects_malformed_entries(tmp_path, item, message):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([item]))

    with pytest.raises(ValueError, match=message):
        evaluate.load_questions(path)


# --- scoring ----------------------------------------------------------------


def test_correct_answer_with_expected_source():
    result = evaluate.score_result(
        ANSWERABLE, graph_output(["query-params.md", "x.md"]), 1.234
    )

    assert result["actual"] == "answer"
    assert result["correct"] is True
    assert result["source_found"] is True
    assert result["sources"] == ["query-params.md", "x.md"]
    assert result["latency_seconds"] == 1.23
    assert result["error"] is None


def test_sources_are_deduplicated_but_keep_their_order():
    output = graph_output(["b.md", "a.md", "b.md"])
    assert evaluate.score_result(ANSWERABLE, output, 0)["sources"] == ["b.md", "a.md"]


def test_answer_without_the_expected_source_is_correct_routing_but_source_not_found():
    result = evaluate.score_result(ANSWERABLE, graph_output(["path-params.md"]), 1)

    assert result["correct"] is True
    assert result["source_found"] is False


def test_no_answer_when_an_answer_was_expected_is_wrong():
    result = evaluate.score_result(
        ANSWERABLE, graph_output(no_answer=True, retries=2), 1
    )

    assert result["actual"] == "no_answer"
    assert result["correct"] is False
    assert result["source_found"] is False
    assert result["retries"] == 2


def test_out_of_scope_question_that_gets_no_answer_is_correct():
    result = evaluate.score_result(
        OUT_OF_SCOPE, graph_output(no_answer=True, retries=2), 1
    )

    assert result["correct"] is True
    assert result["source_found"] is None


def test_out_of_scope_question_that_gets_an_answer_is_wrong():
    result = evaluate.score_result(OUT_OF_SCOPE, graph_output(["query-params.md"]), 1)

    assert result["correct"] is False


def test_error_is_recorded_and_counts_as_wrong():
    result = evaluate.score_result(ANSWERABLE, None, 0.5, "RuntimeError: boom")

    assert result["actual"] == "error"
    assert result["correct"] is False
    assert result["source_found"] is False
    assert result["retries"] is None
    assert result["error"] == "RuntimeError: boom"


def test_run_question_catches_exceptions():
    def failing(question):
        raise ConnectionError("groq unreachable")

    result = evaluate.run_question(ANSWERABLE, failing)

    assert result["actual"] == "error"
    assert result["error"] == "ConnectionError: groq unreachable"


# --- summary and table ------------------------------------------------------


def make_results() -> list[dict]:
    return [
        evaluate.score_result(ANSWERABLE, graph_output(["query-params.md"]), 2.0),
        evaluate.score_result(ANSWERABLE, graph_output(["path-params.md"]), 4.0),
        evaluate.score_result(
            OUT_OF_SCOPE, graph_output(no_answer=True, retries=2), 6.0
        ),
        evaluate.score_result(OUT_OF_SCOPE, None, 0.0, "TimeoutError: slow"),
    ]


def test_summarize_counts_routing_sources_and_errors():
    summary = evaluate.summarize(make_results())

    assert summary == {
        "total": 4,
        "routing_correct": 3,
        "answerable": 2,
        "source_found": 1,
        "errors": 1,
        "average_latency_seconds": 3.0,
    }


def test_summary_line_format():
    clean = {
        "total": 10,
        "routing_correct": 9,
        "answerable": 7,
        "source_found": 6,
        "errors": 0,
    }
    assert (
        evaluate.summary_line(clean)
        == "correct routing 9/10, expected source found 6/7"
    )
    assert evaluate.summary_line({**clean, "errors": 2}).endswith(
        ", 2 failed with an error"
    )


def test_summarize_handles_an_empty_run():
    assert evaluate.summarize([])["average_latency_seconds"] == 0.0


def test_format_table_has_a_row_per_result_and_all_columns():
    table = evaluate.format_table(make_results())
    lines = table.splitlines()

    assert len(lines) == 2 + 4
    for column in (
        "question",
        "expected",
        "actual",
        "retries",
        "sources",
        "source found",
        "latency",
    ):
        assert column in lines[0]
    assert "query-params.md" in lines[2]
    assert " yes " in lines[2] and " no " in lines[3]
    assert "no_answer" in lines[4]
    assert "error" in lines[5]


def test_format_table_truncates_long_questions():
    long_item = {"question": "word " * 40, "expect": "no_answer"}
    table = evaluate.format_table(
        [evaluate.score_result(long_item, graph_output(no_answer=True), 1)]
    )

    assert "…" in table
    assert max(len(line) for line in table.splitlines()) < 140


# --- the whole script -------------------------------------------------------


def test_main_runs_every_question_and_writes_the_results_file(tmp_path, capsys):
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps([ANSWERABLE, OUT_OF_SCOPE, {**ANSWERABLE, "question": "boom"}])
    )
    output = tmp_path / "results.json"

    def fake_run(question):
        if question == "boom":
            raise RuntimeError("provider exploded")
        if question == OUT_OF_SCOPE["question"]:
            return graph_output(no_answer=True, retries=2)
        return graph_output(["query-params.md"])

    evaluate.main(
        ["--questions", str(questions), "--output", str(output), "--pause", "0"],
        run=fake_run,
    )

    saved = json.loads(output.read_text())
    assert set(saved) == {"run_at", "complete", "summary", "results"}
    assert saved["complete"] is True
    assert saved["summary"]["total"] == 3
    assert saved["summary"]["routing_correct"] == 2
    assert saved["summary"]["errors"] == 1
    assert [r["actual"] for r in saved["results"]] == ["answer", "no_answer", "error"]
    expected_keys = {
        "question", "expect", "actual", "correct", "retries", "sources", "expected_source",
        "source_found", "latency_seconds", "query_type", "grounded", "trace", "answer", "error",
    }  # fmt: skip
    for record in saved["results"]:
        assert set(record) == expected_keys
    assert saved["results"][2]["error"] == "RuntimeError: provider exploded"

    out = capsys.readouterr().out
    assert "source found" in out
    assert (
        "correct routing 2/3, expected source found 1/2, 1 failed with an error" in out
    )
    assert f"results saved to {output}" in out


def test_main_exits_with_a_clear_message_without_a_groq_key(monkeypatch, tmp_path):
    monkeypatch.setattr(
        evaluate, "get_settings", lambda: SimpleNamespace(groq_api_key="", eval_pause=0)
    )
    called = []
    monkeypatch.setattr(evaluate, "run_graph", lambda q: called.append(q))

    with pytest.raises(SystemExit) as exc:
        evaluate.main(["--output", str(tmp_path / "out.json")])

    assert "GROQ_API_KEY is not set" in str(exc.value)
    assert called == []
    assert not (tmp_path / "out.json").exists()


def test_main_exits_when_the_index_is_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(
        evaluate,
        "get_settings",
        lambda: SimpleNamespace(groq_api_key="k", eval_pause=0),
    )
    monkeypatch.setattr("app.ingestion.indexer.chunk_count", lambda: 0)

    with pytest.raises(SystemExit) as exc:
        evaluate.main(["--output", str(tmp_path / "out.json")])

    assert "index is empty" in str(exc.value)


def test_a_provider_error_is_recorded_as_an_error_not_as_no_answer():
    from app.graph.nodes import ProviderError

    def run(question):
        raise ProviderError("every grading call failed: NotFoundError: model_not_found")

    result = evaluate.run_question(OUT_OF_SCOPE, run)

    assert result["actual"] == "error"
    assert result["correct"] is False
    assert result["error"].startswith("ProviderError: every grading call failed")


# --- pause, saving as it goes, progress --------------------------------------


def two_questions(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps([ANSWERABLE, OUT_OF_SCOPE, {**ANSWERABLE, "question": "third"}])
    )
    return path


def test_pause_is_applied_between_questions_but_not_before_the_first_or_after_the_last(
    tmp_path,
):
    sleeps = []
    evaluate.main(
        [
            "--questions",
            str(two_questions(tmp_path)),
            "--output",
            str(tmp_path / "r.json"),
            "--pause",
            "7",
        ],
        run=lambda q: graph_output(["query-params.md"]),
        sleep=sleeps.append,
    )

    assert sleeps == [7.0, 7.0]


def test_pause_defaults_to_the_eval_pause_setting(monkeypatch, tmp_path):
    monkeypatch.setattr(
        evaluate, "get_settings", lambda: SimpleNamespace(eval_pause=5.0)
    )
    sleeps = []
    evaluate.main(
        [
            "--questions",
            str(two_questions(tmp_path)),
            "--output",
            str(tmp_path / "r.json"),
        ],
        run=lambda q: graph_output(["query-params.md"]),
        sleep=sleeps.append,
    )

    assert sleeps == [5.0, 5.0]


def test_zero_pause_never_sleeps(tmp_path):
    sleeps = []
    evaluate.main(
        [
            "--questions",
            str(two_questions(tmp_path)),
            "--output",
            str(tmp_path / "r.json"),
            "--pause",
            "0",
        ],
        run=lambda q: graph_output(["query-params.md"]),
        sleep=sleeps.append,
    )

    assert sleeps == []


def test_results_file_is_rewritten_after_every_question_and_survives_an_error(tmp_path):
    output = tmp_path / "results.json"
    seen_on_disk = []

    def run(question):
        # what is on disk at the moment this question starts
        seen_on_disk.append(json.loads(output.read_text()) if output.exists() else None)
        if question == OUT_OF_SCOPE["question"]:
            raise RuntimeError("rate limit survived the retries")
        return graph_output(["query-params.md"])

    evaluate.main(
        [
            "--questions",
            str(two_questions(tmp_path)),
            "--output",
            str(output),
            "--pause",
            "0",
        ],
        run=run,
    )

    assert seen_on_disk[0] is None
    assert [len(d["results"]) for d in seen_on_disk[1:]] == [1, 2]
    assert all(d["complete"] is False for d in seen_on_disk[1:])
    final = json.loads(output.read_text())
    assert final["complete"] is True
    assert [r["actual"] for r in final["results"]] == ["answer", "error", "answer"]
    assert "rate limit survived" in final["results"][1]["error"]
    assert not list(tmp_path.glob("*.tmp"))


def test_an_interrupted_run_keeps_the_results_so_far(tmp_path, capsys):
    output = tmp_path / "results.json"

    def run(question):
        if question == OUT_OF_SCOPE["question"]:
            raise KeyboardInterrupt
        return graph_output(["query-params.md"])

    evaluate.main(
        [
            "--questions",
            str(two_questions(tmp_path)),
            "--output",
            str(output),
            "--pause",
            "0",
        ],
        run=run,
    )

    saved = json.loads(output.read_text())
    assert saved["complete"] is False
    assert len(saved["results"]) == 1
    assert "interrupted after 1 of 3 questions" in capsys.readouterr().err


def test_progress_lines_name_the_question_outcome_and_errors(tmp_path, capsys):
    def run(question):
        if question == "third":
            raise RuntimeError("provider down")
        return graph_output(["query-params.md"])

    evaluate.main(
        [
            "--questions",
            str(two_questions(tmp_path)),
            "--output",
            str(tmp_path / "r.json"),
            "--pause",
            "0",
        ],
        run=run,
    )

    err = capsys.readouterr().err
    assert "[1/3] running: How do I declare an optional query parameter?" in err
    assert "[1/3] answer (expected answer), retries 0, source found yes" in err
    assert "[2/3] answer (expected no_answer)" in err
    assert "[3/3] ERROR after" in err and "RuntimeError: provider down" in err
