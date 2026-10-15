import io

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.services.feedback import FeedbackStore
from app.services.ingestion import MAX_FILE_BYTES, IngestFailure, check_public_host
from app.services.memory import ConversationMemory

SOURCES = [
    {
        "source": "query-params.md",
        "title": "Query Parameters",
        "header_path": "Query Parameters > Optional parameters",
    }
]


class FakeGraph:
    """Stands in for the compiled graph: returns a final state, or raises."""

    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[dict] = []

    def invoke(self, state: dict, config: dict | None = None) -> dict:
        self.calls.append(state)
        if self.error:
            raise self.error
        return {
            **state,
            "answer": "Give it a default of None. [source: query-params.md]",
            "sources": SOURCES,
            "query_type": "how-to",
            "retry_count": 1,
            "trace": ["analyze_query", "retrieve", "grade_documents", "generate"],
            "search_query": "optional query parameter default None",
            "no_answer": False,
        }


@pytest.fixture
def fake_graph():
    return FakeGraph()


@pytest.fixture
def store(tmp_path):
    return FeedbackStore(tmp_path / "feedback.sqlite3")


@pytest.fixture
def client(fake_graph, store, monkeypatch):
    app = create_app()
    # no `with` block, so the lifespan (embedding model, real graph) is not started
    app.state.graph = fake_graph
    app.state.feedback = store
    app.state.memory = ConversationMemory()
    monkeypatch.setattr("app.services.query.chunk_count", lambda: 84)
    monkeypatch.setattr(
        "app.services.ingestion.ingest_documents",
        lambda docs: {d.source: 3 for d in docs},
    )
    return TestClient(app)


def upload(name: str, content: bytes, content_type: str = "text/markdown"):
    return ("files", (name, io.BytesIO(content), content_type))


# --- health -----------------------------------------------------------------


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "graph_ready": True}


# --- /query -----------------------------------------------------------------


def test_query_happy_path(client, fake_graph, store):
    resp = client.post(
        "/query", json={"question": "  How do I declare an optional parameter?  "}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {
        "query_id",
        "answer",
        "sources",
        "query_type",
        "retries",
        "no_answer",
        "session_id",
    }
    assert body["answer"].startswith("Give it a default of None.")
    assert body["sources"] == SOURCES
    assert body["query_type"] == "how-to"
    assert body["retries"] == 1
    assert body["no_answer"] is False
    assert fake_graph.calls[0]["question"] == "How do I declare an optional parameter?"
    # the returned id can be used for feedback straight away
    fb = client.post("/feedback", json={"query_id": body["query_id"], "rating": "up"})
    assert fb.status_code == 201


@pytest.mark.parametrize("question", ["hi", "   ", "x" * 1001])
def test_query_rejects_bad_question_length(client, question):
    assert client.post("/query", json={"question": question}).status_code == 422


def test_query_requires_question_field(client):
    assert client.post("/query", json={}).status_code == 422


def test_query_with_empty_index_returns_503(client, fake_graph, monkeypatch):
    monkeypatch.setattr("app.services.query.chunk_count", lambda: 0)
    resp = client.post("/query", json={"question": "What is a path parameter?"})

    assert resp.status_code == 503
    assert "Ingest documents first" in resp.json()["detail"]
    assert fake_graph.calls == []


def test_query_graph_failure_returns_502_without_leaking_details(client, fake_graph):
    fake_graph.error = RuntimeError("groq key sk-secret rejected")
    resp = client.post("/query", json={"question": "What is a path parameter?"})

    assert resp.status_code == 502
    assert "sk-secret" not in resp.text
    assert (
        resp.json()["detail"]
        == "The answer could not be generated. Please try again later."
    )


def test_query_without_configured_llm_returns_503(client):
    client.app.state.graph = None
    resp = client.post("/query", json={"question": "What is a path parameter?"})

    assert resp.status_code == 503
    assert client.get("/health").json()["graph_ready"] is False


def test_validation_error_wins_over_missing_llm(client):
    client.app.state.graph = None
    assert client.post("/query", json={"question": "hi"}).status_code == 422


# --- /feedback --------------------------------------------------------------


def test_feedback_is_stored_and_returned(client, store):
    store.save_query("q1", "a question", "an answer", False)
    resp = client.post(
        "/feedback",
        json={"query_id": "q1", "rating": "down", "comment": "  missing an example "},
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["query_id"] == "q1"
    assert body["rating"] == "down"
    assert body["comment"] == "missing an example"
    assert body["feedback_id"] == 1


def test_feedback_comment_is_optional(client, store):
    store.save_query("q1", "a question", "an answer", False)
    resp = client.post(
        "/feedback", json={"query_id": "q1", "rating": "up", "comment": "   "}
    )

    assert resp.status_code == 201
    assert resp.json()["comment"] is None


def test_feedback_for_unknown_query_returns_404(client):
    resp = client.post("/feedback", json={"query_id": "nope", "rating": "up"})
    assert resp.status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {"query_id": "q1", "rating": "meh"},
        {"query_id": "q1", "rating": "up", "comment": "x" * 1001},
        {"rating": "up"},
    ],
)
def test_feedback_validation(client, store, payload):
    store.save_query("q1", "a question", "an answer", False)
    assert client.post("/feedback", json=payload).status_code == 422


def test_feedback_survives_a_restart(tmp_path):
    path = tmp_path / "feedback.sqlite3"
    FeedbackStore(path).save_query("q1", "a question", "an answer", False)

    reopened = FeedbackStore(path)
    record = reopened.add_feedback("q1", "up", None)
    assert record.id == 1


# --- /ingest ----------------------------------------------------------------


def test_ingest_markdown_upload(client):
    resp = client.post("/ingest", files=[upload("notes.md", b"# Notes\n\nSome text.")])

    assert resp.status_code == 201
    assert resp.json() == {
        "ingested": [{"source": "notes.md", "chunks": 3}],
        "errors": [],
    }


def test_ingest_rejects_unsupported_file_type(client):
    resp = client.post(
        "/ingest", files=[upload("setup.exe", b"MZ...", "application/octet-stream")]
    )

    assert resp.status_code == 400
    body = resp.json()
    assert body["ingested"] == []
    assert body["errors"][0]["source"] == "setup.exe"
    assert "unsupported file type '.exe'" in body["errors"][0]["reason"]


def test_ingest_rejects_empty_file(client):
    resp = client.post("/ingest", files=[upload("empty.md", b"")])

    assert resp.status_code == 400
    assert resp.json()["errors"] == [{"source": "empty.md", "reason": "file is empty"}]


def test_ingest_rejects_file_over_2mb(client):
    resp = client.post(
        "/ingest", files=[upload("big.txt", b"a" * (MAX_FILE_BYTES + 1), "text/plain")]
    )

    assert resp.status_code == 413
    assert "larger than 2 MB" in resp.json()["errors"][0]["reason"]


def test_ingest_reports_each_file_when_some_fail(client):
    resp = client.post(
        "/ingest",
        files=[
            upload("good.md", b"# Good\n\ntext"),
            upload("bad.exe", b"MZ", "application/x-msdownload"),
        ],
    )

    assert resp.status_code == 201
    body = resp.json()
    assert [i["source"] for i in body["ingested"]] == ["good.md"]
    assert [e["source"] for e in body["errors"]] == ["bad.exe"]


def test_ingest_without_input_returns_400(client):
    assert client.post("/ingest").status_code == 400


def test_ingest_urls_field(client, monkeypatch):
    from app.ingestion.loader import LoadedDocument

    fetched = []

    def fake_load_url(url, client=None):
        fetched.append(url)
        return LoadedDocument(text="page text", source=url, title="Page")

    monkeypatch.setattr("app.services.ingestion.load_url", fake_load_url)
    resp = client.post(
        "/ingest",
        data={"urls": "https://example.com/a.html,\nhttps://example.com/b.html"},
    )

    assert resp.status_code == 201
    assert fetched == ["https://example.com/a.html", "https://example.com/b.html"]
    assert len(resp.json()["ingested"]) == 2


def test_ingest_rejects_malformed_url(client):
    resp = client.post("/ingest", data={"urls": "not-a-url"})
    assert resp.status_code == 422


def test_ingest_refuses_private_addresses(client):
    # fails in the request hook, before any connection is made
    resp = client.post("/ingest", data={"urls": "http://127.0.0.1:9/secret"})

    assert resp.status_code == 400
    assert "private or local" in resp.json()["errors"][0]["reason"]


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://10.0.0.5/",
        "http://169.254.169.254/latest/",
        "http://[::1]/",
    ],
)
def test_check_public_host_blocks_internal_ips(url):
    with pytest.raises(IngestFailure):
        check_public_host(url)


def test_check_public_host_allows_public_ip():
    check_public_host("http://93.184.216.34/")


# --- /documents -------------------------------------------------------------


def test_documents_lists_indexed_files(client, monkeypatch):
    docs = [
        {"source": "query-params.md", "title": "Query Parameters", "chunk_count": 9}
    ]
    monkeypatch.setattr("app.api.routes.list_documents", lambda: docs)

    resp = client.get("/documents")
    assert resp.status_code == 200
    assert resp.json() == docs


# --- /query with sessions ----------------------------------------------------


def ask(client, question: str, session_id: str | None = None):
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    return client.post("/query", json=payload)


def test_query_echoes_session_id_and_passes_history_to_the_graph(client, fake_graph):
    first = ask(client, "How do I declare a query parameter?", "session-a")
    second = ask(client, "What about optional ones?", "session-a")

    assert first.json()["session_id"] == "session-a"
    assert second.json()["session_id"] == "session-a"
    assert fake_graph.calls[0]["history"] == []
    assert fake_graph.calls[1]["history"] == [
        {
            "question": "How do I declare a query parameter?",
            "answer": "Give it a default of None. [source: query-params.md]",
        }
    ]


def test_sessions_do_not_share_history(client, fake_graph):
    ask(client, "Question from session A", "session-a")
    ask(client, "Question from session B", "session-b")

    assert fake_graph.calls[1]["history"] == []


def test_query_without_session_id_is_stateless(client, fake_graph):
    first = ask(client, "How do I declare a query parameter?")
    ask(client, "What about optional ones?")

    assert first.json()["session_id"] is None
    assert fake_graph.calls[1]["history"] == []


def test_session_id_is_ignored_when_memory_is_off(client, fake_graph):
    client.app.state.memory = None
    first = ask(client, "How do I declare a query parameter?", "session-a")
    ask(client, "What about optional ones?", "session-a")

    assert first.status_code == 200
    assert first.json()["session_id"] is None
    assert fake_graph.calls[1]["history"] == []


@pytest.mark.parametrize("session_id", ["has space", "x" * 65, "semi;colon"])
def test_query_rejects_bad_session_id(client, session_id):
    assert (
        ask(client, "How do I declare a query parameter?", session_id).status_code
        == 422
    )


def test_provider_failure_inside_the_graph_returns_502_not_a_no_answer(client):
    from app.graph.builder import build_graph
    from tests.fakes import FakeRetriever, make_deps, make_doc

    def broken_grader(variables):
        raise RuntimeError("Error code: 404 - model_not_found (secret-model-name)")

    retriever = FakeRetriever(lambda query: [make_doc("a.md", "A", "text")])
    deps = make_deps(retriever, grade=broken_grader)
    client.app.state.graph = build_graph(deps, enable_grounding_check=False)

    resp = ask(client, "How do I declare a query parameter?", "session-a")

    assert resp.status_code == 502
    assert (
        resp.json()["detail"]
        == "The answer could not be generated. Please try again later."
    )
    assert "secret-model-name" not in resp.text
    # a failed question must not leave a half-recorded turn behind
    ok_graph = FakeGraph()
    client.app.state.graph = ok_graph
    ask(client, "What about optional ones?", "session-a")
    assert ok_graph.calls[0]["history"] == []
