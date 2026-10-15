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


# --- settings from .env -----------------------------------------------------

UI_VARS = ["API_BASE_URL", "UI_HOST", "UI_PORT"]


@pytest.fixture
def clean_ui_env(monkeypatch):
    for name in UI_VARS:
        # set then delete, so monkeypatch removes whatever load_env adds afterwards
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)


def test_defaults_without_env_or_file(clean_ui_env, tmp_path):
    ui.load_env(tmp_path / "missing.env")

    assert ui.api_base_url() == "http://localhost:8000"
    assert ui.launch_settings() == ("127.0.0.1", 7860)


def test_settings_in_the_env_file_are_used(clean_ui_env, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "API_BASE_URL=http://api.internal:9000/\nUI_HOST=0.0.0.0\nUI_PORT=8123\n"
    )
    ui.load_env(env_file)

    assert ui.api_base_url() == "http://api.internal:9000"
    assert ui.launch_settings() == ("0.0.0.0", 8123)
    assert str(ui.make_client().base_url) == "http://api.internal:9000"


def test_real_environment_variables_win_over_the_env_file(
    clean_ui_env, monkeypatch, tmp_path
):
    env_file = tmp_path / ".env"
    env_file.write_text("API_BASE_URL=http://from-file:1\nUI_PORT=1111\n")
    monkeypatch.setenv("API_BASE_URL", "http://from-shell:2")
    ui.load_env(env_file)

    assert ui.api_base_url() == "http://from-shell:2"
    assert ui.launch_settings()[1] == 1111


def test_importing_the_module_does_not_read_a_env_file():
    import importlib
    from unittest import mock

    with mock.patch("dotenv.load_dotenv") as loader:
        importlib.reload(ui)
    importlib.reload(ui)  # drop the mock the first reload bound into the module
    assert loader.call_count == 0
