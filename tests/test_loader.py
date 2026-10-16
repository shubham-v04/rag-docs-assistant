import httpx
import pytest

from app.ingestion.loader import load_bytes, load_file, load_url

HTML = b"""<html><head><title>Page Title</title><style>p {color: red}</style>
<script>var secret = 1;</script></head>
<body><nav>Site menu</nav><h1>Heading</h1><p>Visible text.</p><footer>Footer</footer></body></html>"""


def test_markdown_title_comes_from_the_first_h1():
    doc = load_bytes(b"intro\n\n# The Title\n\n## Section\n", "page.md")

    assert doc.title == "The Title"
    assert doc.source == "page.md"


def test_title_falls_back_to_the_file_name():
    assert load_bytes(b"just text", "my-notes.txt").title == "my-notes"
    assert load_bytes(b"no heading here", "plain.md").title == "plain"


def test_html_is_reduced_to_visible_text():
    doc = load_bytes(HTML, "page.html")

    assert doc.title == "Page Title"
    assert "Visible text." in doc.text
    for hidden in ("secret", "color: red", "Site menu", "Footer"):
        assert hidden not in doc.text


def test_unsupported_extension_is_rejected():
    with pytest.raises(ValueError, match="unsupported file type"):
        load_bytes(b"MZ", "program.exe")


def test_load_file_reads_from_disk(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text("# Guide\n\nbody", encoding="utf-8")

    doc = load_file(path)
    assert (doc.source, doc.title) == ("guide.md", "Guide")


def test_load_url_uses_the_url_as_source_and_content_type_for_parsing():
    def handler(request):
        return httpx.Response(200, content=HTML, headers={"content-type": "text/html"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    doc = load_url("https://example.com/docs/page", client=client)

    assert doc.source == "https://example.com/docs/page"
    assert doc.title == "Page Title"
    assert "secret" not in doc.text


def test_load_url_raises_on_http_errors():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))

    with pytest.raises(httpx.HTTPStatusError):
        load_url("https://example.com/missing.md", client=client)


MKDOCS_PAGE = b"""# CORS (Cross-Origin Resource Sharing) { #cors-cross-origin-resource-sharing }

Intro text with {braces} and a literal { #not-a-heading } in prose.

## Use CORSMiddleware { #use-corsmiddleware }

```python
# a comment { #kept }
origins = {"a": 1}
```
"""


def test_mkdocs_anchors_are_removed_from_titles_and_headings():
    doc = load_bytes(MKDOCS_PAGE, "cors.md")

    assert doc.title == "CORS (Cross-Origin Resource Sharing)"
    assert "# CORS (Cross-Origin Resource Sharing)\n" in doc.text
    assert "## Use CORSMiddleware\n" in doc.text
    assert "cors-cross-origin" not in doc.text
    # only heading lines outside code fences are touched
    assert "# a comment { #kept }" in doc.text
    assert "{braces}" in doc.text
    assert "literal { #not-a-heading } in prose" in doc.text


def test_url_ingested_markdown_gets_clean_titles_and_header_paths():
    from app.ingestion.chunker import chunk_document

    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, content=MKDOCS_PAGE)
        )
    )
    doc = load_url("https://example.com/tutorial/cors.md", client=client)
    paths = {c.metadata["header_path"] for c in chunk_document(doc)}

    assert doc.title == "CORS (Cross-Origin Resource Sharing)"
    assert "CORS (Cross-Origin Resource Sharing) > Use CORSMiddleware" in paths
    assert not any("{" in path for path in paths)