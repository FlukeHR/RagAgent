from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import BASE_DIR, Settings


@dataclass
class ConversationState:
    goal: str = ""
    constraints: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    cited_papers: list[str] = field(default_factory=list)
    summary: str = ""

    @classmethod
    def from_dict(cls, value: dict | None) -> "ConversationState":
        fields = cls.__dataclass_fields__
        data = {key: item for key, item in (value or {}).items() if key in fields}
        return cls(**data)


@dataclass(frozen=True)
class PreparedConversation:
    history: list[dict]
    summary: str
    dropped_messages: int


class ConversationManager:
    """Bound and summarize conversation history without classifying intent."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def prepare(self, history: list[dict]) -> PreparedConversation:
        valid = [
            {"role": item.get("role"), "content": str(item.get("content") or "")}
            for item in history
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]
        message_limit = self.settings.agent.history_max_messages
        recent_limit = min(message_limit, self.settings.agent.recent_history_messages)
        recent = valid[-recent_limit:]
        summary = self._summarize(valid[:-recent_limit])
        budget = self.settings.agent.history_max_chars
        bounded: list[dict] = []
        used = len(summary)
        for item in reversed(recent):
            remaining = budget - used
            if remaining <= 0:
                break
            content = str(item.get("content") or "")[:remaining]
            bounded.append({"role": item["role"], "content": content})
            used += len(content)
        bounded.reverse()
        if summary:
            bounded.insert(
                0,
                {
                    "role": "assistant",
                    "content": f"[较早对话的系统摘要]\n{summary}",
                },
            )
        return PreparedConversation(
            history=bounded,
            summary=summary,
            dropped_messages=max(0, len(valid) - len(recent)),
        )

    def update_state(
        self,
        state: ConversationState,
        sources: list[dict],
        summary: str,
        planned_goal: str,
    ) -> ConversationState:
        """Persist the goal selected by the semantic planner and cited papers."""

        if planned_goal:
            state.goal = planned_goal[:500]
        state.summary = summary
        for source in sources:
            paper_id = source.get("paper_id")
            if paper_id and paper_id not in state.cited_papers:
                state.cited_papers.append(paper_id)
        state.cited_papers = state.cited_papers[-50:]
        return state

    def _summarize(self, messages: list[dict]) -> str:
        if not messages:
            return ""
        lines = []
        for item in messages:
            role = "用户" if item["role"] == "user" else "助手"
            content = re.sub(r"\s+", " ", item["content"]).strip()
            lines.append(f"{role}: {content[:300]}")
        return "\n".join(lines)[-self.settings.agent.history_summary_max_chars :]


class ConversationMemory:
    """Bounded SQLite server-side conversation memory with TTL and deletion."""

    def __init__(self, settings: Settings) -> None:
        configured = Path(settings.agent.memory_db_path)
        self.path = configured if configured.is_absolute() else BASE_DIR / configured
        self.ttl = settings.agent.memory_ttl_seconds
        self.max_sessions = settings.agent.memory_max_sessions
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
