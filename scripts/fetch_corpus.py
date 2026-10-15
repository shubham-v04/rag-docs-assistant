"""Download a few pages of the FastAPI tutorial into data/corpus/."""

import logging
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings

logger = logging.getLogger("fetch_corpus")

BASE = "https://raw.githubusercontent.com/fastapi/fastapi/master"
DOCS = f"{BASE}/docs/en/docs/tutorial"

# local filename -> path under docs/en/docs/tutorial
PAGES = {
    "first-steps.md": "first-steps.md",
    "path-params.md": "path-params.md",
    "query-params.md": "query-params.md",
    "request-body.md": "body.md",
    "dependencies.md": "dependencies/index.md",
}

# The upstream markdown pulls example code in with {* ../../docs_src/x.py hl[1] *}
INCLUDE_RE = re.compile(r"^\{\*\s*(\S+)(?:\s+[^*]*)?\*\}\s*$", re.MULTILINE)
HEADING_ID_RE = re.compile(r"[ \t]*\{\s*#[\w-]+\s*\}[ \t]*$", re.MULTILINE)
ADMONITION_RE = re.compile(r"^///\s*(\w+)?.*$", re.MULTILINE)


def inline_code(client: httpx.Client, text: str) -> str:
    """Replace code include markers with the actual source as fenced blocks."""

    def repl(match: re.Match) -> str:
        # docs_src lives at the repo root; the marker path is relative to the docs build
        url = f"{BASE}/{match.group(1).replace('../../', '', 1)}"
        resp = client.get(url)
        if resp.status_code != 200:
            logger.warning(
                "could not fetch code example %s (%s)", url, resp.status_code
            )
            return ""
        return f"\n```python\n{resp.text.rstrip()}\n```\n"

    return INCLUDE_RE.sub(repl, text)


def clean_admonitions(text: str) -> str:
    def repl(match: re.Match) -> str:
        kind = match.group(1)
        return f"**{kind.capitalize()}:**" if kind else ""

    return ADMONITION_RE.sub(repl, text)


def prepare(client: httpx.Client, text: str) -> str:
    text = inline_code(client, text)
    text = HEADING_ID_RE.sub("", text)
    return clean_admonitions(text)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    out_dir = get_settings().corpus_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for name, remote in PAGES.items():
            target = out_dir / name
            if target.exists():
                logger.info("skip %s (already exists)", name)
                continue
            resp = client.get(f"{DOCS}/{remote}")
            resp.raise_for_status()
            target.write_text(prepare(client, resp.text), encoding="utf-8")
            logger.info("saved %s", name)


if __name__ == "__main__":
    main()
