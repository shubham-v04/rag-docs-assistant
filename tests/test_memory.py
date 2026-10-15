import pytest

from app.graph.builder import build_graph
from app.graph.nodes import QueryAnalysis
from app.graph.state import initial_state
from app.services.feedback import FeedbackStore
from app.services.memory import ConversationMemory
from app.services.query import answer_question
from tests.fakes import FakeRetriever, make_deps, make_doc

FIRST = "How do I declare a query parameter?"
FOLLOW_UP = "What about optional ones?"
STANDALONE_QUESTION = "How do I declare an optional query parameter?"
SEARCH_QUERY = "declare optional query parameter default None"
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
    """A graph with fake chains that records what each LLM step receives."""

    def __init__(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.services.query.chunk_count", lambda: 1)
        self.store = FeedbackStore(tmp_path / "fb.sqlite3")
        self.analyzer_inputs: list[dict] = []
        self.grader_inputs: list[dict] = []
        self.generator_inputs: list[dict] = []
        self.retriever = FakeRetriever(lambda query: [DOC])

        def analyze(variables):
            self.analyzer_inputs.append(variables)
            # a real model resolves "optional ones" from the history. It also rewords
            # the question on the first turn, which the graph must ignore.
            has_history = variables["history"] != "(none)"
            query = SEARCH_QUERY if has_history else variables["question"]
            return QueryAnalysis(
                standalone_question=STANDALONE_QUESTION,
                search_query=query,
                query_type="how-to",
            )

        def grade(variables):
            self.grader_inputs.append(variables)
            return True

        def generate(variables):
            self.generator_inputs.append(variables)
            return FIRST_ANSWER

        deps = make_deps(
            self.retriever, grade=grade, analyze=analyze, generate=generate
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
    # retrieval used the rewrite, not the vague follow-up
    assert conversation.retriever.queries == [FIRST, SEARCH_QUERY]


def test_grading_and_generation_get_the_standalone_question_not_the_history(
    conversation,
):
    memory = ConversationMemory()
    conversation.ask(FIRST, memory, "s1")
    conversation.ask(FOLLOW_UP, memory, "s1")

    # the second turn is the follow-up
    grading = conversation.grader_inputs[1:]
    generation = conversation.generator_inputs[1:]
    assert grading
    assert generation

    for variables in grading:
        assert set(variables) == {"question", "context"}
        assert variables["question"] == STANDALONE_QUESTION
    for variables in generation:
        assert set(variables) == {"question", "context", "extra_instructions"}
        assert variables["question"] == STANDALONE_QUESTION
    # nothing from the first turn leaks into either step
    for variables in grading + generation:
        assert FIRST not in "".join(variables.values())
        assert FIRST_ANSWER not in "".join(variables.values())


def test_without_history_grading_and_generation_use_the_users_own_wording(conversation):
    conversation.ask(FIRST, ConversationMemory(), "s1")

    assert conversation.grader_inputs[0]["question"] == FIRST
    assert conversation.generator_inputs[0]["question"] == FIRST


def test_the_original_question_stays_in_state(conversation):
    history = [{"question": FIRST, "answer": FIRST_ANSWER}]
    final = conversation.graph.invoke(initial_state(FOLLOW_UP, 2, history))

    assert final["question"] == FOLLOW_UP
    assert final["standalone_question"] == STANDALONE_QUESTION


def test_failed_analysis_falls_back_to_the_original_question():
    grader_inputs = []

    def broken_analyzer(variables):
        raise RuntimeError("llm unavailable")

    retriever = FakeRetriever(lambda query: [DOC])
    deps = make_deps(
        retriever,
        grade=lambda v: grader_inputs.append(v) or True,
        analyze=broken_analyzer,
    )
    history = [{"question": FIRST, "answer": FIRST_ANSWER}]
    graph = build_graph(deps, enable_grounding_check=False)
    graph.invoke(initial_state(FOLLOW_UP, 0, history))

    assert grader_inputs[0]["question"] == FOLLOW_UP


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
