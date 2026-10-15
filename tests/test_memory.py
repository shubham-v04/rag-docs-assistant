import pytest

from app.graph.builder import build_graph
from app.graph.nodes import QueryAnalysis
from app.services.feedback import FeedbackStore
from app.services.memory import ConversationMemory
from app.services.query import answer_question
from tests.fakes import FakeRetriever, make_deps, make_doc

FIRST = "How do I declare a query parameter?"
FOLLOW_UP = "What about optional ones?"
STANDALONE = "how to declare an optional query parameter"
FIRST_ANSWER = "Declare it as a function argument. [source: query-params.md]"

DOC = make_doc(
    "query-params.md", "Query Parameters", "query parameters are function arguments"
)


# --- the store --------------------------------------------------------------


def test_unknown_session_has_no_history():
    assert ConversationMemory().get("nope") == []


def test_keeps_only_the_last_six_turns():
    memory = ConversationMemory()
    for i in range(8):
        memory.add("s", f"q{i}", f"a{i}")

    assert [t["question"] for t in memory.get("s")] == [f"q{i}" for i in range(2, 8)]


def test_evicts_the_least_recently_used_session_when_full():
    memory = ConversationMemory(max_sessions=2)
    memory.add("a", "q", "a")
    memory.add("b", "q", "a")
    memory.get("a")  # a is now more recent than b
    memory.add("c", "q", "a")

    assert memory.get("b") == []
    assert memory.get("a") != []
    assert memory.get("c") != []


def test_sessions_are_separate_and_get_returns_a_copy():
    memory = ConversationMemory()
    memory.add("a", "question a", "answer a")
    memory.add("b", "question b", "answer b")

    history = memory.get("a")
    history[0]["question"] = "tampered"
    history.append({"question": "extra", "answer": "extra"})

    assert memory.get("a") == [{"question": "question a", "answer": "answer a"}]
    assert memory.get("b") == [{"question": "question b", "answer": "answer b"}]


# --- through the graph ------------------------------------------------------


class Conversation:
    """A graph with fake chains that records what the analyzer and grader receive."""

    def __init__(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.services.query.chunk_count", lambda: 1)
        self.store = FeedbackStore(tmp_path / "fb.sqlite3")
        self.analyzer_inputs: list[dict] = []
        self.grader_inputs: list[dict] = []
        self.retriever = FakeRetriever(lambda query: [DOC])

        def analyze(variables):
            self.analyzer_inputs.append(variables)
            # a real model would resolve "optional ones" from the history
            has_history = variables["history"] != "(none)"
            query = STANDALONE if has_history else variables["question"]
            return QueryAnalysis(search_query=query, query_type="how-to")

        def grade(variables):
            self.grader_inputs.append(variables)
            return True

        deps = make_deps(
            self.retriever,
            grade=grade,
            analyze=analyze,
            generate=lambda v: FIRST_ANSWER,
        )
        self.graph = build_graph(deps, enable_grounding_check=False)

    def ask(self, question, memory, session_id):
        return answer_question(question, self.graph, self.store, memory, session_id)


@pytest.fixture
def conversation(tmp_path, monkeypatch):
    return Conversation(tmp_path, monkeypatch)


def test_follow_up_is_rewritten_using_the_first_turn(conversation):
    memory = ConversationMemory()
    conversation.ask(FIRST, memory, "s1")
    conversation.ask(FOLLOW_UP, memory, "s1")

    first_call, second_call = conversation.analyzer_inputs
    assert first_call["history"] == "(none)"
    assert FIRST in second_call["history"]
    assert FIRST_ANSWER in second_call["history"]
    assert second_call["question"] == FOLLOW_UP
    # retrieval used the standalone rewrite, not the vague follow-up
    assert conversation.retriever.queries == [FIRST, STANDALONE]


def test_history_never_reaches_grading(conversation):
    memory = ConversationMemory()
    conversation.ask(FIRST, memory, "s1")
    conversation.ask(FOLLOW_UP, memory, "s1")

    second_turn_grades = conversation.grader_inputs[1:]
    assert second_turn_grades
    for variables in second_turn_grades:
        assert set(variables) == {"question", "context"}
        assert variables["question"] == FOLLOW_UP
        assert FIRST not in variables["context"]


def test_two_sessions_never_see_each_others_history(conversation):
    memory = ConversationMemory()
    conversation.ask("session A first question", memory, "a")
    conversation.ask("session B first question", memory, "b")
    conversation.ask("session A second question", memory, "a")

    _, b_first, a_second = conversation.analyzer_inputs
    assert b_first["history"] == "(none)"
    assert "session A first question" in a_second["history"]
    assert "session B" not in a_second["history"]


def test_without_a_session_id_every_question_is_standalone(conversation):
    memory = ConversationMemory()
    conversation.ask(FIRST, memory, None)
    result = conversation.ask(FOLLOW_UP, memory, None)

    assert conversation.analyzer_inputs[1]["history"] == "(none)"
    assert result["session_id"] is None


def test_memory_off_ignores_the_session_id(conversation):
    conversation.ask(FIRST, None, "s1")
    result = conversation.ask(FOLLOW_UP, None, "s1")

    assert conversation.analyzer_inputs[1]["history"] == "(none)"
    assert result["session_id"] is None
