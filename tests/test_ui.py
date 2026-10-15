import httpx
import pytest

import ui


def client_with(handler) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://api.test"
    )


def test_ask_returns_the_payload():
    seen = {}

    def handler(request):
        seen["body"] = request.read()
        return httpx.Response(
            200, json={"answer": "hi", "sources": [], "query_id": "q1"}
        )

    payload = ui.ask(client_with(handler), "What is a path parameter?", "sess1")

    assert payload["query_id"] == "q1"
    assert b'"session_id":"sess1"' in seen["body"].replace(b" ", b"")


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (
            503,
            {"detail": "The index is empty."},
            "The service is not ready: The index is empty.",
        ),
        (502, {"detail": "generic"}, "could not produce an answer"),
        (
            422,
            {"detail": [{"msg": "String should have at least 3 characters"}]},
            "at least 3 characters",
        ),
        (500, {}, "HTTP 500"),
    ],
)
def test_http_errors_become_readable_messages(status, body, expected):
    client = client_with(lambda request: httpx.Response(status, json=body))

    with pytest.raises(ui.ApiError, match=expected):
        ui.ask(client, "What is a path parameter?", "sess1")


def test_connection_refused_is_explained():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ui.ApiError, match="Cannot reach the API"):
        ui.ask(client_with(handler), "What is a path parameter?", "sess1")


def test_timeout_is_explained():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ui.ApiError, match="took too long"):
        ui.ask(client_with(handler), "What is a path parameter?", "sess1")


def test_respond_shows_errors_in_the_chat(monkeypatch):
    def failing_ask(client, question, session_id):
        raise ui.ApiError(
            "Cannot reach the API at http://localhost:8000. Is the server running?"
        )

    monkeypatch.setattr(ui, "ask", failing_ask)
    chat, textbox, query_id = ui.respond(
        "What is a path parameter?", [], "sess1", "old-id"
    )

    assert chat[0] == {"role": "user", "content": "What is a path parameter?"}
    assert chat[1]["role"] == "assistant"
    assert "Cannot reach the API" in chat[1]["content"]
    assert textbox == ""
    assert query_id is None


def test_respond_formats_answer_with_sources(monkeypatch):
    payload = {
        "query_id": "q9",
        "answer": "Use a default of None.",
        "sources": [
            {
                "source": "query-params.md",
                "title": "Query Parameters",
                "header_path": "A > B",
            }
        ],
    }
    monkeypatch.setattr(ui, "ask", lambda client, question, session_id: payload)
    chat, _, query_id = ui.respond("optional parameter?", [], "sess1", None)

    assert query_id == "q9"
    assert "Use a default of None." in chat[1]["content"]
    assert "`query-params.md`: A > B" in chat[1]["content"]


def test_feedback_is_posted_for_the_last_query():
    seen = {}

    def handler(request):
        seen["body"] = request.read()
        return httpx.Response(201, json={"feedback_id": 1})

    ui.send_feedback(client_with(handler), "q9", "down")
    assert b'"query_id":"q9"' in seen["body"].replace(b" ", b"")
    assert b'"rating":"down"' in seen["body"].replace(b" ", b"")
