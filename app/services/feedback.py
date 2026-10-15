import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    query_id   TEXT PRIMARY KEY,
    question   TEXT NOT NULL,
    answer     TEXT NOT NULL,
    no_answer  INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id   TEXT NOT NULL REFERENCES queries(query_id),
    rating     TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    comment    TEXT,
    created_at TEXT NOT NULL
);
"""


class UnknownQueryError(Exception):
    pass


@dataclass
class FeedbackRecord:
    id: int
    query_id: str
    rating: str
    comment: str | None
    created_at: str


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class FeedbackStore:
    """Remembers answered queries and the feedback given on them.

    Queries are stored too, so a query_id handed out before a restart still accepts
    feedback afterwards. Each call opens its own connection because the API runs
    endpoints in worker threads and sqlite connections are not shared across threads.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def save_query(
        self, query_id: str, question: str, answer: str, no_answer: bool
    ) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO queries (query_id, question, answer, no_answer, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (query_id, question, answer, int(no_answer), _now()),
            )

    def add_feedback(
        self, query_id: str, rating: str, comment: str | None
    ) -> FeedbackRecord:
        created_at = _now()
        with closing(self._connect()) as conn, conn:
            try:
                cursor = conn.execute(
                    "INSERT INTO feedback (query_id, rating, comment, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (query_id, rating, comment, created_at),
                )
            except sqlite3.IntegrityError as exc:
                if "FOREIGN KEY" not in str(exc):
                    raise
                raise UnknownQueryError(query_id) from exc
        return FeedbackRecord(cursor.lastrowid, query_id, rating, comment, created_at)
