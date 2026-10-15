from app.graph.builder import decide_after_grading
from app.graph.state import initial_state
from tests.fakes import make_doc


def state_with(
    relevant: bool, retry_count: int, max_retries: int = 2, error: str | None = None
):
    state = initial_state("q", max_retries)
    state["provider_error"] = error
    state["retry_count"] = retry_count
    if relevant:
        state["relevant_documents"] = [make_doc("a.md", "A", "text")]
    return state


def test_relevant_docs_go_to_generate():
    assert decide_after_grading(state_with(relevant=True, retry_count=0)) == "generate"


def test_relevant_docs_go_to_generate_even_after_retries():
    assert decide_after_grading(state_with(relevant=True, retry_count=2)) == "generate"


def test_no_docs_with_retries_left_rewrites():
    assert (
        decide_after_grading(state_with(relevant=False, retry_count=1))
        == "rewrite_query"
    )


def test_no_docs_with_retries_exhausted_gives_up():
    assert (
        decide_after_grading(state_with(relevant=False, retry_count=2)) == "no_answer"
    )


def test_provider_error_ends_the_run_even_with_retries_left():
    state = state_with(relevant=False, retry_count=0, error="every grading call failed")
    assert decide_after_grading(state) == "provider_error"
