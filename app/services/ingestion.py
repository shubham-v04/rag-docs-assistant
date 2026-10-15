import ipaddress
import logging
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlparse

import httpx

from app.ingestion.indexer import ingest_documents
from app.ingestion.loader import (
    SUPPORTED_SUFFIXES,
    LoadedDocument,
    load_bytes,
    load_url,
)

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 2 * 1024 * 1024
URL_TIMEOUT_SECONDS = 15


class IngestFailure(Exception):
    def __init__(self, source: str, reason: str, status_code: int = 400):
        super().__init__(f"{source}: {reason}")
        self.source = source
        self.reason = reason
        self.status_code = status_code


@dataclass
class IngestResult:
    ingested: dict[str, int] = field(default_factory=dict)
    errors: list[IngestFailure] = field(default_factory=list)


def load_upload(filename: str, fileobj: BinaryIO) -> LoadedDocument:
    """Validate an uploaded file and turn it into a document."""
    name = Path(filename).name
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise IngestFailure(
            name, f"unsupported file type '{suffix}' (use .md, .txt or .html)"
        )

    data = fileobj.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise IngestFailure(
            name, f"file is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB", 413
        )
    if not data.strip():
        raise IngestFailure(name, "file is empty")
    return load_bytes(data, name)


def _strip_scope(address: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    return ipaddress.ip_address(address.split("%")[0])


def check_public_host(url: str) -> None:
    """Refuse URLs that point at this machine or a private network.

    Without this, /ingest would let a caller make the server fetch internal addresses.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise IngestFailure(url, "only http and https URLs are supported")
    try:
        addresses = {
            info[4][0] for info in socket.getaddrinfo(parsed.hostname, parsed.port)
        }
    except socket.gaierror as exc:
        raise IngestFailure(url, "could not resolve the host") from exc
    if any(not ip.is_global for ip in map(_strip_scope, addresses)):
        raise IngestFailure(
            url, "URLs that resolve to private or local addresses are not allowed"
        )


def _check_request(request: httpx.Request) -> None:
    # runs on every redirect hop too, not only the first request
    check_public_host(str(request.url))


def _check_response_size(response: httpx.Response) -> None:
    length = response.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_FILE_BYTES:
        raise IngestFailure(str(response.url), "page is larger than 2 MB", 413)


def fetch_url(url: str) -> LoadedDocument:
    client = httpx.Client(
        timeout=URL_TIMEOUT_SECONDS,
        follow_redirects=True,
        event_hooks={"request": [_check_request], "response": [_check_response_size]},
    )
    try:
        with client:
            doc = load_url(url, client=client)
    except IngestFailure as failure:
        raise IngestFailure(url, failure.reason, failure.status_code) from failure
    except httpx.HTTPError as exc:
        logger.warning("could not fetch %s: %s", url, exc)
        raise IngestFailure(url, "could not fetch the URL") from exc
    if len(doc.text.encode("utf-8")) > MAX_FILE_BYTES:
        raise IngestFailure(url, "page is larger than 2 MB", 413)
    return doc


def ingest_sources(
    uploads: list[tuple[str, BinaryIO]], urls: list[str]
) -> IngestResult:
    """Validate every input, then index the ones that passed. One bad input does not
    stop the others."""
    result = IngestResult()
    docs: list[LoadedDocument] = []

    loaders = [(name, lambda n=name, f=f: load_upload(n, f)) for name, f in uploads]
    loaders += [(url, lambda u=url: fetch_url(u)) for url in urls]
    for source, load in loaders:
        try:
            docs.append(load())
        except IngestFailure as failure:
            result.errors.append(failure)
        except ValueError as exc:
            result.errors.append(IngestFailure(source, str(exc)))

    if docs:
        result.ingested = ingest_documents(docs)
    for doc in docs:
        if doc.source not in result.ingested:
            result.errors.append(
                IngestFailure(doc.source, "no text could be extracted")
            )
    return result
