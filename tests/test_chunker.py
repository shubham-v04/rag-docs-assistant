from types import SimpleNamespace

import pytest

from app.ingestion.chunker import chunk_document
from app.ingestion.loader import LoadedDocument

CODE = """```python
from fastapi import FastAPI

# a comment that looks like a heading marker
app = FastAPI()


@app.get("/items/")
async def read_items(skip: int = 0):
    if skip:
        return []
    return [{"id": 1}]
```"""

SAMPLE = f"""# Query Parameters

Intro paragraph about query parameters.

## Defaults

{CODE}

Another sentence after the example.

## Long section

{" ".join(f"Sentence number {i} explains something about parameters." for i in range(40))}

### Nested heading

Nested text.
"""


def use_settings(monkeypatch, chunk_size: int, chunk_overlap: int = 40):
    monkeypatch.setattr(
        "app.ingestion.chunker.get_settings",
        lambda: SimpleNamespace(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
    )


def doc(text: str, source: str = "sample.md") -> LoadedDocument:
    return LoadedDocument(text=text, source=source, title="Query Parameters")


@pytest.mark.parametrize("chunk_size", [300, 500, 900])
def test_chunks_respect_the_configured_size(monkeypatch, chunk_size):
    use_settings(monkeypatch, chunk_size)
    chunks = chunk_document(doc(SAMPLE))

    assert len(chunks) > 1
    # the sample's code block is smaller than the smallest size, so nothing may exceed it
    assert max(len(c.page_content) for c in chunks) <= chunk_size


def test_smaller_chunk_size_gives_more_chunks(monkeypatch):
    use_settings(monkeypatch, 900)
    large = chunk_document(doc(SAMPLE))
    use_settings(monkeypatch, 300)
    small = chunk_document(doc(SAMPLE))

    assert len(small) > len(large)


def test_every_chunk_has_the_expected_metadata(monkeypatch):
    use_settings(monkeypatch, 300)
    chunks = chunk_document(doc(SAMPLE, source="query-params.md"))

    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.metadata["source"] == "query-params.md"
        assert chunk.metadata["title"] == "Query Parameters"
        assert isinstance(chunk.metadata["header_path"], str)
        assert chunk.page_content.strip()


def test_header_path_follows_the_heading_hierarchy(monkeypatch):
    use_settings(monkeypatch, 300)
    paths = {c.metadata["header_path"] for c in chunk_document(doc(SAMPLE))}

    assert "Query Parameters" in paths
    assert "Query Parameters > Defaults" in paths
    assert "Query Parameters > Long section > Nested heading" in paths


def test_code_blocks_are_not_cut_in_half(monkeypatch):
    use_settings(monkeypatch, 300)
    chunks = chunk_document(doc(SAMPLE))

    for chunk in chunks:
        assert chunk.page_content.count("```") % 2 == 0
    code_chunks = [c for c in chunks if "def read_items" in c.page_content]
    assert len(code_chunks) == 1
    # the whole block survives with its indentation
    assert CODE in code_chunks[0].page_content


def test_hash_lines_inside_code_do_not_start_a_section(monkeypatch):
    use_settings(monkeypatch, 900)
    paths = {c.metadata["header_path"] for c in chunk_document(doc(SAMPLE))}

    assert not any("a comment that looks like" in path for path in paths)


def test_oversized_code_block_is_split_into_balanced_fenced_parts(monkeypatch):
    use_settings(monkeypatch, 200)
    long_code = (
        "```python\n" + "\n".join(f"value_{i} = {i}" for i in range(80)) + "\n```"
    )
    chunks = chunk_document(doc(f"# Big\n\n{long_code}\n"))

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.page_content.count("```") % 2 == 0
        assert chunk.page_content.startswith("# Big") or chunk.page_content.startswith(
            "```python"
        )


def test_text_without_headings_is_chunked_with_an_empty_header_path(monkeypatch):
    use_settings(monkeypatch, 200)
    text = " ".join(f"Plain sentence {i}." for i in range(60))
    chunks = chunk_document(
        LoadedDocument(text=text, source="notes.txt", title="notes")
    )

    assert len(chunks) > 1
    assert {c.metadata["header_path"] for c in chunks} == {""}
    assert max(len(c.page_content) for c in chunks) <= 200


def test_blank_document_gives_no_chunks(monkeypatch):
    use_settings(monkeypatch, 300)
    assert chunk_document(doc("   \n\n  ")) == []