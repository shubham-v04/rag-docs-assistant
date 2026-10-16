import logging
import random
import re
import time
from collections.abc import Callable
from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 5
BASE_DELAY_SECONDS = 2.0
MAX_DELAY_SECONDS = 60.0

# Groq words the wait as "Please try again in 7.66s", "1m3.5s" or "250ms"
_HINT_RE = re.compile(r"try again in ((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)", re.IGNORECASE)
_PART_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)", re.IGNORECASE)
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def is_rate_limited(exc: BaseException) -> bool:
    return getattr(exc, "status_code", None) == 429


def retry_after_seconds(exc: BaseException) -> float | None:
    """The wait the provider asked for, from the retry-after header or the message."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    value = headers.get("retry-after") if headers else None
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    match = _HINT_RE.search(str(exc))
    if match:
        parts = _PART_RE.findall(match.group(1))
        return sum(
            float(number) * _UNIT_SECONDS[unit.lower()] for number, unit in parts
        )
    return None


def backoff_seconds(attempt: int, exc: BaseException) -> float:
    """Wait before retry number `attempt` (1-based): the provider's hint if it gave one,
    otherwise exponential, plus jitter so parallel calls do not retry in lockstep."""
    hint = retry_after_seconds(exc)
    if hint is not None:
        delay = hint + random.uniform(0, 1)
    else:
        delay = BASE_DELAY_SECONDS * 2 ** (attempt - 1)
        delay += random.uniform(0, delay * 0.25)
    return min(delay, MAX_DELAY_SECONDS)


def call_with_retry(
    fn: Callable[..., Any],
    *args: Any,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int = MAX_ATTEMPTS,
) -> Any:
    """Call fn, retrying only on HTTP 429. Anything else, and the last 429, is raised."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args)
        except Exception as exc:
            if not is_rate_limited(exc) or attempt == max_attempts:
                raise
            delay = backoff_seconds(attempt, exc)
            logger.warning(
                "rate limited (attempt %d of %d), waiting %.1fs",
                attempt,
                max_attempts,
                delay,
            )
            sleep(delay)
    raise AssertionError("unreachable")


def retrying(chain: Runnable, sleep: Callable[[float], None] = time.sleep) -> Runnable:
    """Wrap a chain so each invoke, and each item of a batch, retries on rate limits."""
    return RunnableLambda(
        lambda value: call_with_retry(chain.invoke, value, sleep=sleep)
    )
