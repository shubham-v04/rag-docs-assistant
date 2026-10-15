"""Minimal chat UI that talks to the running API.

Start the API first (uvicorn app.api.main:app), then run: python ui.py
"""

import os
import uuid
from pathlib import Path

import gradio as gr
import httpx
from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parent / ".env"
DEFAULT_API_BASE_URL = "http://localhost:8000"
# a full run does several LLM calls, so allow for a slow answer
REQUEST_TIMEOUT_SECONDS = 120


class ApiError(Exception):
    """An API problem, worded so it can be shown to the user as is."""


def load_env(path: Path = ENV_FILE) -> None:
    # variables already set in the shell win over the file
    load_dotenv(path)


def api_base_url() -> str:
    return os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def launch_settings() -> tuple[str, int]:
    return os.getenv("UI_HOST", "127.0.0.1"), int(os.getenv("UI_PORT", "7860"))


def make_client() -> httpx.Client:
    return httpx.Client(base_url=api_base_url(), timeout=REQUEST_TIMEOUT_SECONDS)


def describe_http_error(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None

    if response.status_code == 422:
        # validation errors come back as a list of {"msg": ...}
        msg = (
            detail[0]["msg"] if isinstance(detail, list) and detail else "invalid input"
        )
        return f"The question was not accepted: {msg}."
    if response.status_code == 503:
        return f"The service is not ready: {detail or 'try again later'}"
    if response.status_code == 502:
        return (
            "The service could not produce an answer right now. Please try again later."
        )
    if response.status_code == 404 and isinstance(detail, str):
        return detail
    return f"Unexpected error from the API (HTTP {response.status_code})."


def post_json(client: httpx.Client, path: str, payload: dict) -> dict:
    try:
        response = client.post(path, json=payload)
    except httpx.ConnectError as exc:
        raise ApiError(
            f"Cannot reach the API at {client.base_url}. Is the server running?"
        ) from exc
    except httpx.TimeoutException as exc:
        raise ApiError("The API took too long to respond. Please try again.") from exc
    except httpx.HTTPError as exc:
        raise ApiError("The request to the API failed.") from exc

    if response.is_error:
        raise ApiError(describe_http_error(response))
    return response.json()


def format_answer(payload: dict) -> str:
    text = payload["answer"]
    if payload.get("sources"):
        lines = [
            f"- `{s['source']}`: {s['header_path'] or s['title']}"
            for s in payload["sources"]
        ]
        text += "\n\n**Sources**\n" + "\n".join(lines)
    return text


def ask(client: httpx.Client, question: str, session_id: str) -> dict:
    return post_json(client, "/query", {"question": question, "session_id": session_id})


def send_feedback(client: httpx.Client, query_id: str, rating: str) -> None:
    post_json(client, "/feedback", {"query_id": query_id, "rating": rating})


def respond(message: str, chat: list, session_id: str, last_query_id: str | None):
    message = message.strip()
    if not message:
        return chat, "", last_query_id

    chat = [*chat, {"role": "user", "content": message}]
    try:
        with make_client() as client:
            payload = ask(client, message, session_id)
    except ApiError as exc:
        chat.append({"role": "assistant", "content": f"⚠️ {exc}"})
        return chat, "", None

    chat.append({"role": "assistant", "content": format_answer(payload)})
    return chat, "", payload["query_id"]


def rate(rating: str, last_query_id: str | None) -> None:
    if not last_query_id:
        gr.Warning("Ask a question first, then rate its answer.")
        return
    try:
        with make_client() as client:
            send_feedback(client, last_query_id, rating)
    except ApiError as exc:
        gr.Warning(str(exc))
        return
    gr.Info("Thanks for the feedback.")


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Docs Assistant") as demo:
        gr.Markdown(
            "# Docs Assistant\nAsk questions about the indexed FastAPI documentation."
        )
        # one id per browser session, so follow-up questions share history
        session_id = gr.State(lambda: uuid.uuid4().hex)
        last_query_id = gr.State(None)

        chat = gr.Chatbot(height=460)
        box = gr.Textbox(placeholder="Ask a question and press Enter", show_label=False)
        with gr.Row():
            up = gr.Button("👍 Helpful")
            down = gr.Button("👎 Not helpful")

        box.submit(
            respond,
            inputs=[box, chat, session_id, last_query_id],
            outputs=[chat, box, last_query_id],
        )
        up.click(lambda qid: rate("up", qid), inputs=[last_query_id])
        down.click(lambda qid: rate("down", qid), inputs=[last_query_id])
    return demo


if __name__ == "__main__":
    # only when run as a script, so importing this module never reads a .env file
    load_env()
    host, port = launch_settings()
    build_ui().launch(server_name=host, server_port=port)
