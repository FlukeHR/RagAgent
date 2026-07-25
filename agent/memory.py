from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from agent.conversation import ConversationState
from config.settings import BASE_DIR, Settings


class ConversationMemory:
    """Bounded SQLite server-side conversation memory with TTL and deletion."""

    def __init__(self, settings: Settings) -> None:
        configured = Path(settings.harness.memory_db_path)
        self.path = configured if configured.is_absolute() else BASE_DIR / configured
        self.ttl = settings.harness.memory_ttl_seconds
        self.max_sessions = settings.harness.memory_max_sessions
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    session_id TEXT PRIMARY KEY,
                    updated_at REAL NOT NULL,
                    state_json TEXT NOT NULL,
                    history_json TEXT NOT NULL
                )
                """
            )

    def load(self, session_id: str) -> tuple[ConversationState, list[dict]]:
        if not self._valid_session_id(session_id):
            raise ValueError("invalid session_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT updated_at, state_json, history_json "
                "FROM conversations WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return ConversationState(), []
            if self.ttl > 0 and row[0] < time.time() - self.ttl:
                connection.execute(
                    "DELETE FROM conversations WHERE session_id = ?",
                    (session_id,),
                )
                return ConversationState(), []
            return (
                ConversationState.from_dict(json.loads(row[1])),
                list(json.loads(row[2])),
            )

    def save(
        self,
        session_id: str,
        state: ConversationState,
        history: list[dict],
    ) -> None:
        if not self._valid_session_id(session_id):
            raise ValueError("invalid session_id")
        payload_state = json.dumps(state.__dict__, ensure_ascii=False)
        payload_history = json.dumps(history, ensure_ascii=False)
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations(session_id, updated_at, state_json, history_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                  updated_at = excluded.updated_at,
                  state_json = excluded.state_json,
                  history_json = excluded.history_json
                """,
                (session_id, now, payload_state, payload_history),
            )
            if self.ttl > 0:
                connection.execute(
                    "DELETE FROM conversations WHERE updated_at < ?",
                    (now - self.ttl,),
                )
            if self.max_sessions > 0:
                connection.execute(
                    """
                    DELETE FROM conversations
                    WHERE session_id IN (
                      SELECT session_id FROM conversations
                      ORDER BY updated_at DESC
                      LIMIT -1 OFFSET ?
                    )
                    """,
                    (self.max_sessions,),
                )

    def delete(self, session_id: str) -> None:
        if not self._valid_session_id(session_id):
            raise ValueError("invalid session_id")
        with self._lock, self._connect() as connection:
            connection.execute(
                "DELETE FROM conversations WHERE session_id = ?",
                (session_id,),
            )

    @staticmethod
    def _valid_session_id(value: str) -> bool:
        return bool(value) and len(value) <= 128 and all(
            char.isalnum() or char in "-_" for char in value
        )
