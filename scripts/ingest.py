"""Index every supported file in data/corpus/ into the vector store."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.ingestion.indexer import ingest_documents, list_documents
from app.ingestion.loader import SUPPORTED_SUFFIXES, load_file


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    corpus_dir = get_settings().corpus_dir
    paths = sorted(
        p for p in corpus_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not paths:
        sys.exit(f"no documents in {corpus_dir}, run scripts/fetch_corpus.py first")

    counts = ingest_documents([load_file(p) for p in paths])

    print(f"Ingested {len(counts)} documents, {sum(counts.values())} chunks")
    for doc in list_documents():
        print(f"  {doc['source']:<20} {doc['chunk_count']:>3} chunks  {doc['title']}")


if __name__ == "__main__":
    main()
