import logging
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".html", ".htm"}
_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
# MkDocs writes explicit anchors as "## Title { #title }"; they are noise once indexed
_HEADING_ANCHOR_RE = re.compile(r"^(#{1,6}\s+.*?)[ \t]*\{\s*#[\w-]+\s*\}[ \t]*$")


@dataclass
class LoadedDocument:
    text: str
    source: str
    title: str


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def strip_heading_anchors(text: str) -> str:
    """Remove "{ #id }" from heading lines, leaving code fences alone."""
    lines = []
    in_fence = False
    for line in text.split("\n"):
        if line.startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            line = _HEADING_ANCHOR_RE.sub(r"\1", line)
        lines.append(line)
    return "\n".join(lines)


def _markdown_title(text: str, fallback: str) -> str:
    match = _H1_RE.search(text)
    return match.group(1) if match else fallback


def _parse_html(html: str, fallback_title: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(
        ["script", "style", "nav", "footer", "header", "aside", "noscript"]
    ):
        tag.decompose()

    title = fallback_title
    if soup.title and soup.title.string and soup.title.string.strip():
        title = soup.title.string.strip()
    elif (h1 := soup.find("h1")) and h1.get_text(strip=True):
        title = h1.get_text(strip=True)

    text = soup.get_text("\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, title


def load_bytes(data: bytes, filename: str, source: str | None = None) -> LoadedDocument:
    """Build a document from raw bytes; the file extension decides how it is parsed."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported file type '{suffix}' (use .md, .txt or .html)")

    name = source or filename
    fallback_title = Path(filename).stem
    raw = _decode(data)

    if suffix in {".html", ".htm"}:
        text, title = _parse_html(raw, fallback_title)
        return LoadedDocument(text=text, source=name, title=title)

    text = raw.strip()
    if suffix in {".md", ".markdown"}:
        text = strip_heading_anchors(text)
        return LoadedDocument(text, name, _markdown_title(text, fallback_title))
    return LoadedDocument(text, name, fallback_title)


def load_file(path: str | Path) -> LoadedDocument:
    path = Path(path)
    return load_bytes(path.read_bytes(), path.name)


def load_url(url: str, client: httpx.Client | None = None) -> LoadedDocument:
    """Fetch a page and parse it. The source name is the URL itself."""
    owns_client = client is None
    client = client or httpx.Client(timeout=30, follow_redirects=True)
    try:
        resp = client.get(url)
        resp.raise_for_status()
    finally:
        if owns_client:
            client.close()

    url_path = urlparse(url).path
    suffix = Path(url_path).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        content_type = resp.headers.get("content-type", "")
        suffix = ".html" if "html" in content_type else ".txt"

    filename = Path(url_path).name or urlparse(url).netloc
    if not Path(filename).suffix:
        filename += suffix
    logger.info("fetched %s (%d bytes)", url, len(resp.content))
    return load_bytes(resp.content, filename, source=url)