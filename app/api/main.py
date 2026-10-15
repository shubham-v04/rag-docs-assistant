import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import get_settings
from app.graph.builder import get_graph
from app.ingestion.indexer import get_vectorstore
from app.services.feedback import FeedbackStore

logger = logging.getLogger(__name__)


def load_graph():
    try:
        return get_graph()
    except RuntimeError as exc:
        # the API should still start so /ingest, /documents and /feedback keep working
        logger.warning("graph not available, /query will return 503: %s", exc)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.feedback = FeedbackStore(get_settings().feedback_db)
    get_vectorstore()
    app.state.graph = load_graph()
    yield


def create_app() -> FastAPI:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    app = FastAPI(title="Docs Assistant", version="0.1.0", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
