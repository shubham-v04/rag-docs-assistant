import logging
import uuid

from app.graph.builder import run_graph
from app.ingestion.indexer import chunk_count
from app.services.feedback import FeedbackStore

logger = logging.getLogger(__name__)


class IndexEmptyError(Exception):
    pass


class QueryFailedError(Exception):
    pass


def answer_question(question: str, graph, store: FeedbackStore) -> dict:
    if chunk_count() == 0:
        raise IndexEmptyError

    try:
        result = run_graph(question, graph=graph)
    except Exception as exc:
        logger.exception("graph run failed")
        raise QueryFailedError from exc

    query_id = uuid.uuid4().hex
    store.save_query(query_id, question, result["answer"], result["no_answer"])
    return {"query_id": query_id, **result}
