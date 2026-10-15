import logging
import uuid

from app.graph.builder import run_graph
from app.ingestion.indexer import chunk_count
from app.services.feedback import FeedbackStore
from app.services.memory import ConversationMemory

logger = logging.getLogger(__name__)


class IndexEmptyError(Exception):
    pass


class QueryFailedError(Exception):
    pass


def answer_question(
    question: str,
    graph,
    store: FeedbackStore,
    memory: ConversationMemory | None = None,
    session_id: str | None = None,
) -> dict:
    if chunk_count() == 0:
        raise IndexEmptyError

    # memory is None when the feature is switched off; the session id is then ignored
    use_memory = memory is not None and session_id is not None
    history = memory.get(session_id) if use_memory else []

    try:
        result = run_graph(question, graph=graph, history=history)
    except Exception as exc:
        logger.exception("graph run failed")
        raise QueryFailedError from exc

    query_id = uuid.uuid4().hex
    store.save_query(query_id, question, result["answer"], result["no_answer"])
    if use_memory:
        memory.add(session_id, question, result["answer"])
    return {
        "query_id": query_id,
        "session_id": session_id if use_memory else None,
        **result,
    }
