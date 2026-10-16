import hashlib
from types import SimpleNamespace

import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.ingestion import indexer
from app.ingestion.loader import LoadedDocument


class FakeEmbeddings(Embeddings):
    """Hash-based vectors, so no model is downloaded. Remembers what it embedded."""

    def __init__(self):
        self.embedded_documents: list[str] = []

    @staticmethod
    def vector(text: str) -> list[float]:
        return [b / 255 + 0.01 for b in hashlib.sha256(text.encode()).digest()[:8]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded_documents.extend(texts)
        return [self.vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.vector(text)


@pytest.fixture
def embeddings(tmp_path, monkeypatch):
    fake = FakeEmbeddings()
    store = Chroma(
        collection_name="test_docs",
        embedding_function=fake,
        persist_directory=str(tmp_path / "chroma"),
        collection_metadata={"hnsw:space": "cosine"},
    )
    monkeypatch.setattr(indexer, "get_embeddings", lambda: fake)
    monkeypatch.setattr(indexer, "get_vectorstore", lambda: store)
    # small chunks so a short test document produces several of them
    monkeypatch.setattr(
        "app.ingestion.chunker.get_settings",
        lambda: SimpleNamespace(chunk_size=300, chunk_overlap=30),
    )
    return fake


def make_document(source: str, sections: int, title: str = "Guide") -> LoadedDocument:
    body = "\n\n".join(
        f"## Section {i}\n\n" + " ".join(f"{source} fact {i}.{j}." for j in range(12))
        for i in range(sections)
    )
    return LoadedDocument(text=f"# {title}\n\n{body}", source=source, title=title)


def stored_ids() -> list[str]:
    return indexer.get_vectorstore().get()["ids"]


def test_ingest_reports_chunk_counts_and_lists_documents(embeddings):
    counts = indexer.ingest_documents(
        [make_document("a.md", 3), make_document("b.md", 2)]
    )

    assert set(counts) == {"a.md", "b.md"}
    assert counts["a.md"] > counts["b.md"] > 0
    listed = {d["source"]: d for d in indexer.list_documents()}
    assert listed["a.md"] == {
        "source": "a.md",
        "title": "Guide",
        "chunk_count": counts["a.md"],
    }
    assert listed["b.md"]["chunk_count"] == counts["b.md"]
    assert indexer.chunk_count() == sum(counts.values())


def test_chunk_ids_are_source_and_index(embeddings):
    counts = indexer.ingest_documents([make_document("a.md", 3)])

    assert sorted(stored_ids()) == sorted(f"a.md::{i}" for i in range(counts["a.md"]))


def test_reingesting_the_same_document_does_not_duplicate_chunks(embeddings):
    first = indexer.ingest_documents([make_document("a.md", 3)])
    second = indexer.ingest_documents([make_document("a.md", 3)])

    assert first == second
    assert indexer.chunk_count() == first["a.md"]


def test_reingesting_a_shorter_version_removes_the_stale_chunks(embeddings):
    long_count = indexer.ingest_documents([make_document("a.md", 5)])["a.md"]
    short_count = indexer.ingest_documents([make_document("a.md", 1)])["a.md"]

    assert short_count < long_count
    assert indexer.chunk_count() == short_count
    assert sorted(stored_ids()) == sorted(f"a.md::{i}" for i in range(short_count))


def test_replacing_one_source_leaves_the_others_alone(embeddings):
    indexer.ingest_documents([make_document("a.md", 3), make_document("b.md", 3)])
    before = {d["source"]: d["chunk_count"] for d in indexer.list_documents()}

    indexer.ingest_documents([make_document("a.md", 1, title="Rewritten")])
    after = {d["source"]: d for d in indexer.list_documents()}

    assert after["b.md"]["chunk_count"] == before["b.md"]
    assert after["a.md"]["title"] == "Rewritten"
    assert after["a.md"]["chunk_count"] < before["a.md"]


def test_stored_text_is_the_chunk_but_the_embedded_text_has_the_header_path(embeddings):
    indexer.ingest_documents([make_document("a.md", 2)])

    stored = indexer.get_vectorstore().get(include=["documents", "metadatas"])
    for text, meta in zip(stored["documents"], stored["metadatas"]):
        assert not text.startswith("Guide >")
        assert {"source", "title", "header_path", "chunk_index"} <= set(meta)
    # every embedded string is a stored chunk with its section name in front
    assert any(
        e.startswith("Guide > Section 0\n") for e in embeddings.embedded_documents
    )
    assert sorted(e.split("\n", 1)[1] for e in embeddings.embedded_documents) == sorted(
        stored["documents"]
    )


def test_a_document_with_no_text_is_skipped(embeddings):
    blank = LoadedDocument(text="  \n ", source="blank.md", title="blank")

    assert indexer.ingest_documents([blank]) == {}
    assert indexer.chunk_count() == 0


def test_similarity_search_returns_the_closest_chunk(embeddings):
    indexer.ingest_documents([make_document("a.md", 2)])
    store = indexer.get_vectorstore()
    chunk = store.get(include=["documents"])["documents"][0]

    # fake vectors come from the embedded text, so search with the same header + chunk
    header = store.get(include=["metadatas"])["metadatas"][0]["header_path"]
    best = store.similarity_search(f"{header}\n{chunk}", k=1)[0]
    assert best.page_content == chunk