import logging
import os
from collections import defaultdict
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings
from app.ingestion.chunker import chunk_document
from app.ingestion.loader import LoadedDocument

logger = logging.getLogger(__name__)


class _DropHubNotice(logging.Filter):
    """The embedding model is public, so the "unauthenticated requests" advice is noise."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "unauthenticated requests" not in record.getMessage()


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    settings = get_settings()
    if settings.hf_token:
        # huggingface_hub reads the token from the environment, not from our .env
        os.environ.setdefault("HF_TOKEN", settings.hf_token)
    else:
        logging.getLogger("huggingface_hub.utils._http").addFilter(_DropHubNotice())
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache
def get_vectorstore() -> Chroma:
    settings = get_settings()
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def _chunk_id(source: str, index: int) -> str:
    return f"{source}::{index}"


def _embedding_text(chunk: Document) -> str:
    """Prefix the header path so a chunk's section name helps it match queries."""
    header_path = chunk.metadata.get("header_path", "")
    return f"{header_path}\n{chunk.page_content}" if header_path else chunk.page_content


def _delete_source(source: str) -> None:
    store = get_vectorstore()
    old_ids = store.get(where={"source": source})["ids"]
    if old_ids:
        store.delete(ids=old_ids)


def ingest_documents(docs: list[LoadedDocument]) -> dict[str, int]:
    """Chunk and index documents. Returns chunk counts keyed by source."""
    store = get_vectorstore()
    counts: dict[str, int] = {}
    for doc in docs:
        chunks = chunk_document(doc)
        if not chunks:
            logger.warning("no text found in %s, skipping", doc.source)
            continue
        # drop old chunks first: a shorter re-upload would otherwise leave stale ones
        _delete_source(doc.source)
        ids = [_chunk_id(doc.source, c.metadata["chunk_index"]) for c in chunks]
        # the wrapper embeds page_content, but we want to embed a different text than we
        # store, so go through the collection directly
        store._collection.upsert(
            ids=ids,
            documents=[c.page_content for c in chunks],
            embeddings=get_embeddings().embed_documents(
                [_embedding_text(c) for c in chunks]
            ),
            metadatas=[c.metadata for c in chunks],
        )
        counts[doc.source] = len(chunks)
        logger.info("indexed %s (%d chunks)", doc.source, len(chunks))
    return counts


def list_documents() -> list[dict]:
    metadatas = get_vectorstore().get(include=["metadatas"])["metadatas"]
    chunk_counts: dict[str, int] = defaultdict(int)
    titles: dict[str, str] = {}
    for meta in metadatas:
        source = meta["source"]
        chunk_counts[source] += 1
        titles[source] = meta.get("title", source)
    return [
        {"source": source, "title": titles[source], "chunk_count": chunk_counts[source]}
        for source in sorted(chunk_counts)
    ]


def chunk_count() -> int:
    return get_vectorstore()._collection.count()