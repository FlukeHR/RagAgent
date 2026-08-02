from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from config.settings import BASE_DIR, Settings


@dataclass(frozen=True)
class UserRecord:
    """One local application account."""

    user_id: str
    username: str
    display_name: str
    password_hash: str
    failed_attempts: int
    locked_until: float
    created_at: float


@dataclass(frozen=True)
class AuthSessionRecord:
    """A server-side browser session; only a hash of the cookie is stored."""

    session_id: str
    user_id: str
    token_hash: str
    csrf_token: str
    user_agent: str
    created_at: float
    last_seen_at: float
    expires_at: float


class AppStore:
    """Transactional SQLite storage for the local multi-user application."""

    def __init__(self, settings: Settings, path: str | Path | None = None) -> None:
        configured = Path(path or settings.app.database_path)
        self.path = configured if configured.is_absolute() else BASE_DIR / configured
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open a bounded WAL connection with foreign-key enforcement."""

        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._write_lock, self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    failed_attempts INTEGER NOT NULL DEFAULT 0,
                    locked_until REAL NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    token_hash TEXT NOT NULL UNIQUE,
                    csrf_token TEXT NOT NULL,
                    user_agent TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS auth_sessions_user_idx
                    ON auth_sessions(user_id, expires_at);

                CREATE TABLE IF NOT EXISTS model_profiles (
                    profile_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    api_base TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    key_nonce TEXT NOT NULL,
                    key_ciphertext TEXT NOT NULL,
                    key_last4 TEXT NOT NULL,
                    key_version INTEGER NOT NULL DEFAULT 1,
                    is_default INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(user_id, name)
                );
                CREATE INDEX IF NOT EXISTS model_profiles_user_idx
                    ON model_profiles(user_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    model_profile_id TEXT REFERENCES model_profiles(profile_id) ON DELETE SET NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS conversations_user_idx
                    ON conversations(user_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    status TEXT,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    steps_json TEXT NOT NULL DEFAULT '[]',
                    trace_json TEXT NOT NULL DEFAULT '[]',
                    actions_json TEXT NOT NULL DEFAULT '[]',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_conversation_idx
                    ON messages(conversation_id, created_at, message_id);

                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    storage_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    origin TEXT NOT NULL CHECK(origin IN ('upload', 'arxiv')),
                    arxiv_id TEXT,
                    status TEXT NOT NULL,
                    error TEXT,
                    page_count INTEGER,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(user_id, storage_id),
                    UNIQUE(user_id, arxiv_id)
                );
                CREATE INDEX IF NOT EXISTS papers_user_idx
                    ON papers(user_id, status, updated_at DESC);

                CREATE TABLE IF NOT EXISTS ingest_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    query TEXT NOT NULL,
                    candidates_json TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    consumed INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    job_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    paper_id TEXT REFERENCES papers(paper_id) ON DELETE CASCADE,
                    proposal_id TEXT REFERENCES ingest_proposals(proposal_id) ON DELETE SET NULL,
                    query TEXT NOT NULL DEFAULT '',
                    arxiv_id TEXT,
                    status TEXT NOT NULL,
                    error TEXT,
                    result_json TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ingest_jobs_user_idx
                    ON ingest_jobs(user_id, updated_at DESC);
                """
            )

    # Users and authentication -------------------------------------------------
    def create_user(self, username: str, display_name: str, password_hash: str) -> UserRecord:
        """Create one local user with a case-insensitively unique username."""

        user_id = uuid.uuid4().hex
        now = time.time()
        with self._write_lock, self.connect() as connection:
            connection.execute(
                """INSERT INTO users
                   (user_id, username, display_name, password_hash, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, username, display_name, password_hash, now, now),
            )
        user = self.get_user(user_id)
        assert user is not None
        return user

    def get_user(self, user_id: str) -> UserRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
        return self._user(row)

    def get_user_by_username(self, username: str) -> UserRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,)
            ).fetchone()
        return self._user(row)

    def update_login_state(
        self, user_id: str, *, failed_attempts: int, locked_until: float
    ) -> None:
        with self._write_lock, self.connect() as connection:
            connection.execute(
                """UPDATE users SET failed_attempts = ?, locked_until = ?, updated_at = ?
                   WHERE user_id = ?""",
                (failed_attempts, locked_until, time.time(), user_id),
            )

    def update_password(self, user_id: str, password_hash: str) -> None:
        with self._write_lock, self.connect() as connection:
            connection.execute(
                """UPDATE users SET password_hash = ?, failed_attempts = 0,
                   locked_until = 0, updated_at = ? WHERE user_id = ?""",
                (password_hash, time.time(), user_id),
            )

    def update_display_name(self, user_id: str, display_name: str) -> None:
        with self._write_lock, self.connect() as connection:
            connection.execute(
                "UPDATE users SET display_name = ?, updated_at = ? WHERE user_id = ?",
                (display_name, time.time(), user_id),
            )

    def create_auth_session(
        self,
        user_id: str,
        token_hash: str,
        csrf_token: str,
        user_agent: str,
        expires_at: float,
    ) -> AuthSessionRecord:
        session_id = uuid.uuid4().hex
        now = time.time()
        with self._write_lock, self.connect() as connection:
            connection.execute(
                """INSERT INTO auth_sessions
                   (session_id, user_id, token_hash, csrf_token, user_agent,
                    created_at, last_seen_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    user_id,
                    token_hash,
                    csrf_token,
                    user_agent[:300],
                    now,
                    now,
                    expires_at,
                ),
            )
        session = self.get_auth_session(token_hash)
        assert session is not None
        return session

    def get_auth_session(self, token_hash: str) -> AuthSessionRecord | None:
        now = time.time()
        with self._write_lock, self.connect() as connection:
            connection.execute("DELETE FROM auth_sessions WHERE expires_at <= ?", (now,))
            row = connection.execute(
                "SELECT * FROM auth_sessions WHERE token_hash = ?", (token_hash,)
            ).fetchone()
            if row is not None:
                connection.execute(
                    "UPDATE auth_sessions SET last_seen_at = ? WHERE session_id = ?",
                    (now, row["session_id"]),
                )
        return self._auth_session(row)

    def list_auth_sessions(self, user_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT session_id, user_agent, created_at, last_seen_at, expires_at
                   FROM auth_sessions WHERE user_id = ? AND expires_at > ?
                   ORDER BY last_seen_at DESC""",
                (user_id, time.time()),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_auth_session(self, token_hash: str) -> None:
        with self._write_lock, self.connect() as connection:
            connection.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash,))

    def delete_other_auth_sessions(self, user_id: str, keep_token_hash: str) -> None:
        with self._write_lock, self.connect() as connection:
            connection.execute(
                "DELETE FROM auth_sessions WHERE user_id = ? AND token_hash != ?",
                (user_id, keep_token_hash),
            )

    def delete_all_auth_sessions(self, user_id: str) -> None:
        with self._write_lock, self.connect() as connection:
            connection.execute("DELETE FROM auth_sessions WHERE user_id = ?", (user_id,))

    # Model profiles -----------------------------------------------------------
    def list_model_profiles(self, user_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM model_profiles WHERE user_id = ?
                   ORDER BY is_default DESC, updated_at DESC""",
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_model_profile(self, user_id: str, profile_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM model_profiles WHERE profile_id = ? AND user_id = ?",
                (profile_id, user_id),
            ).fetchone()
        return dict(row) if row is not None else None

    def create_model_profile(self, user_id: str, values: dict[str, Any]) -> dict[str, Any]:
        profile_id = str(values.get("profile_id") or uuid.uuid4().hex)
        now = time.time()
        with self._write_lock, self.connect() as connection:
            if values.get("is_default"):
                connection.execute(
                    "UPDATE model_profiles SET is_default = 0 WHERE user_id = ?",
                    (user_id,),
                )
            connection.execute(
                """INSERT INTO model_profiles
                   (profile_id, user_id, name, provider, api_base, model_name,
                    key_nonce, key_ciphertext, key_last4, key_version,
                    is_default, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
                (
                    profile_id,
                    user_id,
                    values["name"],
                    values["provider"],
                    values["api_base"],
                    values["model_name"],
                    values["key_nonce"],
                    values["key_ciphertext"],
                    values["key_last4"],
                    int(bool(values.get("is_default"))),
                    now,
                    now,
                ),
            )
            if not connection.execute(
                "SELECT 1 FROM model_profiles WHERE user_id = ? AND is_default = 1",
                (user_id,),
            ).fetchone():
                connection.execute(
                    "UPDATE model_profiles SET is_default = 1 WHERE profile_id = ?",
                    (profile_id,),
                )
        profile = self.get_model_profile(user_id, profile_id)
        assert profile is not None
        return profile

    def update_model_profile(
        self, user_id: str, profile_id: str, values: dict[str, Any]
    ) -> dict[str, Any] | None:
        current = self.get_model_profile(user_id, profile_id)
        if current is None:
            return None
        merged = {**current, **values}
        now = time.time()
        with self._write_lock, self.connect() as connection:
            if merged.get("is_default"):
                connection.execute(
                    "UPDATE model_profiles SET is_default = 0 WHERE user_id = ?",
                    (user_id,),
                )
            connection.execute(
                """UPDATE model_profiles SET name = ?, provider = ?, api_base = ?,
                   model_name = ?, key_nonce = ?, key_ciphertext = ?, key_last4 = ?,
                   key_version = ?, is_default = ?, updated_at = ?
                   WHERE profile_id = ? AND user_id = ?""",
                (
                    merged["name"],
                    merged["provider"],
                    merged["api_base"],
                    merged["model_name"],
                    merged["key_nonce"],
                    merged["key_ciphertext"],
                    merged["key_last4"],
                    int(merged.get("key_version", 1)),
                    int(bool(merged.get("is_default"))),
                    now,
                    profile_id,
                    user_id,
                ),
            )
        return self.get_model_profile(user_id, profile_id)

    def delete_model_profile(self, user_id: str, profile_id: str) -> bool:
        with self._write_lock, self.connect() as connection:
            row = connection.execute(
                "SELECT is_default FROM model_profiles WHERE profile_id = ? AND user_id = ?",
                (profile_id, user_id),
            ).fetchone()
            if row is None:
                return False
            connection.execute(
                "DELETE FROM model_profiles WHERE profile_id = ? AND user_id = ?",
                (profile_id, user_id),
            )
            if bool(row["is_default"]):
                fallback = connection.execute(
                    """SELECT profile_id FROM model_profiles WHERE user_id = ?
                       ORDER BY updated_at DESC LIMIT 1""",
                    (user_id,),
                ).fetchone()
                if fallback:
                    connection.execute(
                        "UPDATE model_profiles SET is_default = 1 WHERE profile_id = ?",
                        (fallback["profile_id"],),
                    )
        return True

    # Conversations ------------------------------------------------------------
    def create_conversation(
        self, user_id: str, title: str, model_profile_id: str | None
    ) -> dict[str, Any]:
        conversation_id = uuid.uuid4().hex
        now = time.time()
        with self._write_lock, self.connect() as connection:
            connection.execute(
                """INSERT INTO conversations
                   (conversation_id, user_id, title, model_profile_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (conversation_id, user_id, title, model_profile_id, now, now),
            )
        conversation = self.get_conversation(user_id, conversation_id)
        assert conversation is not None
        return conversation

    def list_conversations(self, user_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT c.*, COUNT(m.message_id) AS message_count
                   FROM conversations c LEFT JOIN messages m
                     ON m.conversation_id = c.conversation_id
                   WHERE c.user_id = ? GROUP BY c.conversation_id
                   ORDER BY c.updated_at DESC""",
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(
        self, user_id: str, conversation_id: str, *, include_messages: bool = True
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            if include_messages:
                messages = connection.execute(
                    """SELECT * FROM messages WHERE conversation_id = ? AND user_id = ?
                       ORDER BY created_at, message_id""",
                    (conversation_id, user_id),
                ).fetchall()
                result["messages"] = [self._message(item) for item in messages]
        return result

    def update_conversation(
        self,
        user_id: str,
        conversation_id: str,
        *,
        title: str | None = None,
        model_profile_id: str | None = None,
        update_profile: bool = False,
    ) -> dict[str, Any] | None:
        current = self.get_conversation(user_id, conversation_id, include_messages=False)
        if current is None:
            return None
        with self._write_lock, self.connect() as connection:
            connection.execute(
                """UPDATE conversations SET title = ?, model_profile_id = ?, updated_at = ?
                   WHERE conversation_id = ? AND user_id = ?""",
                (
                    title if title is not None else current["title"],
                    model_profile_id if update_profile else current["model_profile_id"],
                    time.time(),
                    conversation_id,
                    user_id,
                ),
            )
        return self.get_conversation(user_id, conversation_id)

    def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        with self._write_lock, self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, user_id),
            )
        return cursor.rowcount > 0

    def add_message(
        self,
        user_id: str,
        conversation_id: str,
        role: str,
        content: str,
        *,
        status: str | None = None,
        sources: list[dict[str, Any]] | None = None,
        steps: list[str] | None = None,
        trace: list[dict[str, Any]] | None = None,
        actions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        message_id = uuid.uuid4().hex
        now = time.time()
        with self._write_lock, self.connect() as connection:
            connection.execute(
                """INSERT INTO messages
                   (message_id, conversation_id, user_id, role, content, status,
                    sources_json, steps_json, trace_json, actions_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    message_id,
                    conversation_id,
                    user_id,
                    role,
                    content,
                    status,
                    json.dumps(sources or [], ensure_ascii=False),
                    json.dumps(steps or [], ensure_ascii=False),
                    json.dumps(trace or [], ensure_ascii=False),
                    json.dumps(actions or [], ensure_ascii=False),
                    now,
                ),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ? AND user_id = ?",
                (now, conversation_id, user_id),
            )
        return {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "status": status,
            "sources": sources or [],
            "steps": steps or [],
            "trace": trace or [],
            "actions": actions or [],
            "created_at": now,
        }

    # Papers, proposals, and jobs ---------------------------------------------
    def paper_count(self, user_id: str) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM papers WHERE user_id = ?", (user_id,)
            ).fetchone()
        return int(row["count"] if row else 0)

    def create_paper(
        self,
        user_id: str,
        storage_id: str,
        title: str,
        original_filename: str,
        origin: str,
        *,
        arxiv_id: str | None = None,
        status: str = "queued",
        page_count: int | None = None,
    ) -> dict[str, Any]:
        paper_id = uuid.uuid4().hex
        now = time.time()
        with self._write_lock, self.connect() as connection:
            connection.execute(
                """INSERT INTO papers
                   (paper_id, user_id, storage_id, title, original_filename, origin,
                    arxiv_id, status, page_count, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    paper_id,
                    user_id,
                    storage_id,
                    title,
                    original_filename,
                    origin,
                    arxiv_id,
                    status,
                    page_count,
                    now,
                    now,
                ),
            )
        paper = self.get_paper(user_id, paper_id)
        assert paper is not None
        return paper

    def list_papers(self, user_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM papers WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_paper(self, user_id: str, paper_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM papers WHERE paper_id = ? AND user_id = ?",
                (paper_id, user_id),
            ).fetchone()
        return dict(row) if row is not None else None

    def update_paper(
        self,
        user_id: str,
        paper_id: str,
        *,
        status: str,
        error: str | None = None,
        page_count: int | None = None,
        title: str | None = None,
    ) -> None:
        current = self.get_paper(user_id, paper_id)
        if current is None:
            return
        with self._write_lock, self.connect() as connection:
            connection.execute(
                """UPDATE papers SET status = ?, error = ?, page_count = ?, title = ?,
                   updated_at = ? WHERE paper_id = ? AND user_id = ?""",
                (
                    status,
                    error[:1000] if error else None,
                    page_count if page_count is not None else current["page_count"],
                    title if title is not None else current["title"],
                    time.time(),
                    paper_id,
                    user_id,
                ),
            )

    def delete_paper_record(self, user_id: str, paper_id: str) -> bool:
        with self._write_lock, self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM papers WHERE paper_id = ? AND user_id = ?",
                (paper_id, user_id),
            )
        return cursor.rowcount > 0

    def create_proposal(
        self, user_id: str, query: str, candidates: list[dict[str, Any]], ttl: int
    ) -> dict[str, Any]:
        proposal_id = uuid.uuid4().hex
        now = time.time()
        expires_at = now + ttl
        with self._write_lock, self.connect() as connection:
            connection.execute(
                """INSERT INTO ingest_proposals
                   (proposal_id, user_id, query, candidates_json, expires_at, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    proposal_id,
                    user_id,
                    query[:1000],
                    json.dumps(candidates, ensure_ascii=False),
                    expires_at,
                    now,
                ),
            )
        return {
            "proposal_id": proposal_id,
            "query": query[:1000],
            "candidates": candidates,
            "expires_at": expires_at,
            "consumed": False,
        }

    def consume_proposal(
        self,
        user_id: str,
        proposal_id: str,
        arxiv_id: str,
        *,
        max_pending_user: int,
        max_pending_global: int,
    ) -> dict[str, Any]:
        with self._write_lock, self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT * FROM ingest_proposals
                   WHERE proposal_id = ? AND user_id = ?""",
                (proposal_id, user_id),
            ).fetchone()
            if row is None:
                raise KeyError("unknown ingest proposal")
            if bool(row["consumed"]):
                raise ValueError("ingest proposal was already consumed")
            if float(row["expires_at"]) <= time.time():
                raise ValueError("ingest proposal expired")
            candidates = list(json.loads(row["candidates_json"]))
            candidate = next(
                (item for item in candidates if item.get("arxiv_id") == arxiv_id), None
            )
            if candidate is None:
                raise ValueError("confirmed arXiv ID is outside the proposal")
            pending_user = connection.execute(
                """SELECT COUNT(*) AS count FROM ingest_jobs
                   WHERE user_id = ? AND status IN ('queued','parsing','indexing')""",
                (user_id,),
            ).fetchone()["count"]
            pending_global = connection.execute(
                """SELECT COUNT(*) AS count FROM ingest_jobs
                   WHERE status IN ('queued','parsing','indexing')"""
            ).fetchone()["count"]
            if int(pending_user) >= max_pending_user:
                raise RuntimeError("user ingest queue is full")
            if int(pending_global) >= max_pending_global:
                raise RuntimeError("global ingest queue is full")
            connection.execute(
                "UPDATE ingest_proposals SET consumed = 1 WHERE proposal_id = ?",
                (proposal_id,),
            )
        return candidate

    def create_job(
        self,
        user_id: str,
        paper_id: str,
        *,
        proposal_id: str | None = None,
        query: str = "",
        arxiv_id: str | None = None,
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        now = time.time()
        with self._write_lock, self.connect() as connection:
            connection.execute(
                """INSERT INTO ingest_jobs
                   (job_id, user_id, paper_id, proposal_id, query, arxiv_id,
                    status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?)""",
                (job_id, user_id, paper_id, proposal_id, query, arxiv_id, now, now),
            )
        job = self.get_job(user_id, job_id)
        assert job is not None
        return job

    def get_job(self, user_id: str, job_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM ingest_jobs WHERE job_id = ? AND user_id = ?",
                (job_id, user_id),
            ).fetchone()
        return self._job(row)

    def list_jobs(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM ingest_jobs WHERE user_id = ?
                   ORDER BY updated_at DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        jobs: list[dict[str, Any]] = []
        for row in rows:
            job = self._job(row)
            assert job is not None
            jobs.append(job)
        return jobs

    def pending_job_counts(self, user_id: str) -> tuple[int, int]:
        """Return active job counts for one user and for the whole local app."""

        with self.connect() as connection:
            user_count = connection.execute(
                """SELECT COUNT(*) AS count FROM ingest_jobs WHERE user_id = ?
                   AND status IN ('queued','parsing','indexing')""",
                (user_id,),
            ).fetchone()["count"]
            global_count = connection.execute(
                """SELECT COUNT(*) AS count FROM ingest_jobs
                   WHERE status IN ('queued','parsing','indexing')"""
            ).fetchone()["count"]
        return int(user_count), int(global_count)

    def update_job(
        self,
        user_id: str,
        job_id: str,
        status: str,
        *,
        error: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        if status not in {"queued", "parsing", "indexing", "succeeded", "failed"}:
            raise ValueError("invalid ingest job status")
        with self._write_lock, self.connect() as connection:
            connection.execute(
                """UPDATE ingest_jobs SET status = ?, error = ?, result_json = ?,
                   updated_at = ? WHERE job_id = ? AND user_id = ?""",
                (
                    status,
                    error[:1000] if error else None,
                    json.dumps(result, ensure_ascii=False) if result is not None else None,
                    time.time(),
                    job_id,
                    user_id,
                ),
            )

    def fail_incomplete_jobs(self) -> None:
        with self._write_lock, self.connect() as connection:
            connection.execute(
                """UPDATE ingest_jobs SET status = 'failed',
                   error = 'application restarted during job', updated_at = ?
                   WHERE status IN ('queued','parsing','indexing')""",
                (time.time(),),
            )
            connection.execute(
                """UPDATE papers SET status = 'failed',
                   error = 'application restarted during job', updated_at = ?
                   WHERE status IN ('queued','parsing','indexing')""",
                (time.time(),),
            )

    def dashboard(self, user_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            paper_rows = connection.execute(
                """SELECT status, COUNT(*) AS count FROM papers
                   WHERE user_id = ? GROUP BY status""",
                (user_id,),
            ).fetchall()
            active_jobs = connection.execute(
                """SELECT COUNT(*) AS count FROM ingest_jobs WHERE user_id = ?
                   AND status IN ('queued','parsing','indexing')""",
                (user_id,),
            ).fetchone()["count"]
        conversations = self.list_conversations(user_id)[:5]
        jobs = self.list_jobs(user_id, limit=5)
        return {
            "papers": {str(row["status"]): int(row["count"]) for row in paper_rows},
            "active_jobs": int(active_jobs),
            "recent_conversations": conversations,
            "recent_jobs": jobs,
        }

    @staticmethod
    def _user(row: sqlite3.Row | None) -> UserRecord | None:
        if row is None:
            return None
        return UserRecord(
            user_id=str(row["user_id"]),
            username=str(row["username"]),
            display_name=str(row["display_name"]),
            password_hash=str(row["password_hash"]),
            failed_attempts=int(row["failed_attempts"]),
            locked_until=float(row["locked_until"]),
            created_at=float(row["created_at"]),
        )

    @staticmethod
    def _auth_session(row: sqlite3.Row | None) -> AuthSessionRecord | None:
        if row is None:
            return None
        return AuthSessionRecord(
            session_id=str(row["session_id"]),
            user_id=str(row["user_id"]),
            token_hash=str(row["token_hash"]),
            csrf_token=str(row["csrf_token"]),
            user_agent=str(row["user_agent"]),
            created_at=float(row["created_at"]),
            last_seen_at=float(row["last_seen_at"]),
            expires_at=float(row["expires_at"]),
        )

    @staticmethod
    def _message(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "message_id": str(row["message_id"]),
            "conversation_id": str(row["conversation_id"]),
            "role": str(row["role"]),
            "content": str(row["content"]),
            "status": str(row["status"]) if row["status"] else None,
            "sources": json.loads(row["sources_json"] or "[]"),
            "steps": json.loads(row["steps_json"] or "[]"),
            "trace": json.loads(row["trace_json"] or "[]"),
            "actions": json.loads(row["actions_json"] or "[]"),
            "created_at": float(row["created_at"]),
        }

    @staticmethod
    def _job(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        result["result"] = json.loads(result.pop("result_json")) if result.get("result_json") else None
        return result
