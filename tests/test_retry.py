import groq
import httpx
import pytest
from langchain_core.runnables import RunnableLambda

from app.graph import retry
from app.graph.retry import (
    BASE_DELAY_SECONDS,
    MAX_ATTEMPTS,
    MAX_DELAY_SECONDS,
    backoff_seconds,
    call_with_retry,
    is_rate_limited,
    retry_after_seconds,
    retrying,
)


def rate_limit_error(
    message: str = "Rate limit reached", retry_after: str | None = None
):
    """A real groq.RateLimitError, built the way the SDK builds it from a 429 response."""
    headers = {"retry-after": retry_after} if retry_after else {}
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, headers=headers, request=request)
    return groq.RateLimitError(message, response=response, body=None)


def not_found_error():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return groq.NotFoundError(
        "model_not_found", response=httpx.Response(404, request=request), body=None
    )


@pytest.fixture
def no_jitter(monkeypatch):
    monkeypatch.setattr(retry.random, "uniform", lambda low, high: 0.0)


# --- recognising and reading rate-limit errors --------------------------------


def test_only_429_counts_as_rate_limited():
    assert is_rate_limited(rate_limit_error())
    assert not is_rate_limited(not_found_error())
    assert not is_rate_limited(ValueError("boom"))


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Rate limit reached. Please try again in 7.66s. Need more tokens?", 7.66),
        ("Please try again in 1m3.5s.", 63.5),
        ("Please try again in 250ms.", 0.25),
        ("Please try again in 2h", 7200.0),
        ("Rate limit reached with no hint", None),
    ],
)
def test_retry_after_is_read_from_the_message(message, expected):
    result = retry_after_seconds(rate_limit_error(message))

    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_retry_after_header_wins_over_the_message():
    exc = rate_limit_error("Please try again in 30s", retry_after="4")
    assert retry_after_seconds(exc) == 4.0


# --- backoff ----------------------------------------------------------------


def test_backoff_doubles_each_attempt_without_a_hint(no_jitter):
    delays = [backoff_seconds(n, rate_limit_error()) for n in (1, 2, 3, 4)]
    assert delays == [BASE_DELAY_SECONDS * 2**i for i in range(4)]


def test_backoff_adds_jitter_within_bounds():
    for attempt in (1, 2, 3):
        base = BASE_DELAY_SECONDS * 2 ** (attempt - 1)
        samples = {backoff_seconds(attempt, rate_limit_error()) for _ in range(30)}
        assert all(base <= d <= base * 1.25 for d in samples)
        assert len(samples) > 1


def test_backoff_honours_the_providers_hint(no_jitter):
    assert backoff_seconds(1, rate_limit_error(retry_after="7")) == 7.0
    assert backoff_seconds(3, rate_limit_error("Please try again in 1.5s")) == 1.5


def test_backoff_is_capped(no_jitter):
    assert backoff_seconds(1, rate_limit_error(retry_after="900")) == MAX_DELAY_SECONDS
    assert backoff_seconds(10, rate_limit_error()) == MAX_DELAY_SECONDS


# --- call_with_retry ----------------------------------------------------------


def flaky(failures: int, error=rate_limit_error):
    calls = []

    def fn(value):
        calls.append(value)
        if len(calls) <= failures:
            raise error()
        return f"ok:{value}"

    fn.calls = calls
    return fn


def test_a_429_is_retried_until_it_succeeds(no_jitter):
    sleeps = []
    fn = flaky(2)

    assert call_with_retry(fn, "x", sleep=sleeps.append) == "ok:x"
    assert len(fn.calls) == 3
    assert sleeps == [2.0, 4.0]


def test_a_persistent_429_is_raised_after_the_last_attempt(no_jitter):
    sleeps = []
    fn = flaky(100)

    with pytest.raises(groq.RateLimitError):
        call_with_retry(fn, "x", sleep=sleeps.append)

    assert len(fn.calls) == MAX_ATTEMPTS == 5
    assert len(sleeps) == MAX_ATTEMPTS - 1


def test_other_errors_are_not_retried():
    sleeps = []
    fn = flaky(100, error=not_found_error)

    with pytest.raises(groq.NotFoundError):
        call_with_retry(fn, "x", sleep=sleeps.append)

    assert len(fn.calls) == 1
    assert sleeps == []


def test_the_providers_wait_is_what_gets_slept(no_jitter):
    sleeps = []
    fn = flaky(1, error=lambda: rate_limit_error(retry_after="9"))

    call_with_retry(fn, "x", sleep=sleeps.append)
    assert sleeps == [9.0]


# --- the runnable wrapper -----------------------------------------------------


def test_retrying_wraps_invoke_and_retries_each_batch_item_on_its_own(no_jitter):
    seen = {}
    sleeps = []

    def fn(value):
        seen[value] = seen.get(value, 0) + 1
        if value == "b" and seen[value] == 1:
            raise rate_limit_error()
        return value.upper()

    chain = retrying(RunnableLambda(fn), sleep=sleeps.append)

    assert chain.invoke("a") == "A"
    assert chain.batch(["a", "b", "c"], return_exceptions=True) == ["A", "B", "C"]
    assert seen == {"a": 2, "b": 2, "c": 1}
    assert sleeps == [2.0]
