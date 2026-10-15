import threading
from collections import OrderedDict, deque

from app.graph.state import Turn

MAX_TURNS = 6
MAX_SESSIONS = 1000


class ConversationMemory:
    """Last few turns per session, kept in process memory.

    This is a deliberate simplification: history is lost on restart and is not shared
    between worker processes. A real deployment would put it in Redis or a database.
    The number of sessions is capped so memory cannot grow forever; when the cap is hit
    the least recently used session is dropped.
    """

    def __init__(self, max_turns: int = MAX_TURNS, max_sessions: int = MAX_SESSIONS):
        self.max_turns = max_turns
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, deque[Turn]] = OrderedDict()
        # endpoints run in worker threads
        self._lock = threading.Lock()

    def get(self, session_id: str) -> list[Turn]:
        with self._lock:
            turns = self._sessions.get(session_id)
            if turns is None:
                return []
            self._sessions.move_to_end(session_id)
            return [Turn(question=t["question"], answer=t["answer"]) for t in turns]

    def add(self, session_id: str, question: str, answer: str) -> None:
        with self._lock:
            turns = self._sessions.setdefault(session_id, deque(maxlen=self.max_turns))
            turns.append(Turn(question=question, answer=answer))
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
