from __future__ import annotations

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
    user_id: str
    username: str
    display_name: str
    password_hash: str
    failed_attempts: int
    locked_until: float
    created_at: float


@dataclass(frozen=True)
class AuthSessionRecord:
    session_id: str
    user_id: str
    token_hash: str
    csrf_token: str
    user_agent: str
    created_at: float
    last_seen_at: float
    expires_at: float


class AppStore:
    """SQLite storage for accounts, browser sessions, and model profiles."""

    def __init__(self, settings: Settings, path: str | Path | None = None) -> None:
        configured = Path(path or settings.app.database_path)
        self.path = configured if configured.is_absolute() else BASE_DIR / configured
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open a WAL connection and commit or roll back as one unit."""

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
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS documents_user_idx
                    ON documents(user_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS document_pages (
                    document_id TEXT NOT NULL REFERENCES documents(document_id)
                        ON DELETE CASCADE,
                    page_number INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    PRIMARY KEY (document_id, page_number)
                );
                """
            )

    def create_user(self, username: str, display_name: str, password_hash: str) -> UserRecord:
        """Create one local user."""

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
        """Create a revocable browser session."""

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
        """Create a model profile, making the first profile the default."""

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
        """Update an owned model profile."""

        current = self.get_model_profile(user_id, profile_id)
        if current is None:
            return None
        merged = {**current, **values}
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
                    time.time(),
                    profile_id,
                    user_id,
                ),
            )
        return self.get_model_profile(user_id, profile_id)

    def delete_model_profile(self, user_id: str, profile_id: str) -> bool:
        """Delete an owned profile and promote a replacement default if needed."""

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
            if row["is_default"]:
                fallback = connection.execute(
                    """SELECT profile_id FROM model_profiles WHERE user_id = ?
                       ORDER BY updated_at DESC LIMIT 1""",
                    (user_id,),
                ).fetchone()
                if fallback is not None:
                    connection.execute(
                        "UPDATE model_profiles SET is_default = 1 WHERE profile_id = ?",
                        (fallback["profile_id"],),
                    )
        return True

    def create_document(self, user_id: str, filename: str) -> dict[str, Any]:
        """Create a pending user-owned PDF record."""

        document_id = uuid.uuid4().hex
        now = time.time()
        with self._write_lock, self.connect() as connection:
            connection.execute(
                """INSERT INTO documents
                   (document_id, user_id, filename, status, created_at, updated_at)
                   VALUES (?, ?, ?, 'processing', ?, ?)""",
                (document_id, user_id, filename, now, now),
            )
        document = self.get_document(user_id, document_id)
        assert document is not None
        return document

    def finish_document(
        self,
        user_id: str,
        document_id: str,
        pages: list[tuple[int, str, int, int]],
    ) -> None:
        """Publish all rendered page metadata in one transaction."""

        with self._write_lock, self.connect() as connection:
            connection.executemany(
                """INSERT INTO document_pages
                   (document_id, page_number, text, width, height)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (document_id, page_number, text, width, height)
                    for page_number, text, width, height in pages
                ],
            )
            connection.execute(
                """UPDATE documents SET page_count = ?, status = 'ready', error = NULL,
                   updated_at = ? WHERE document_id = ? AND user_id = ?""",
                (len(pages), time.time(), document_id, user_id),
            )

    def fail_document(self, user_id: str, document_id: str, error: str) -> None:
        """Record a safe parsing failure without deleting local evidence."""

        with self._write_lock, self.connect() as connection:
            connection.execute(
                """UPDATE documents SET status = 'failed', error = ?, updated_at = ?
                   WHERE document_id = ? AND user_id = ?""",
                (error[:300], time.time(), document_id, user_id),
            )

    def list_documents(self, user_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT document_id, filename, page_count, status, error,
                          created_at, updated_at
                   FROM documents WHERE user_id = ? ORDER BY updated_at DESC""",
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_document(self, user_id: str, document_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT document_id, filename, page_count, status, error,
                          created_at, updated_at
                   FROM documents WHERE document_id = ? AND user_id = ?""",
                (document_id, user_id),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_document_pages(
        self, user_id: str, document_id: str
    ) -> list[dict[str, Any]]:
        if self.get_document(user_id, document_id) is None:
            return []
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT page_number, text, width, height FROM document_pages
                   WHERE document_id = ? ORDER BY page_number""",
                (document_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_document_page(
        self, user_id: str, document_id: str, page_number: int
    ) -> dict[str, Any] | None:
        if self.get_document(user_id, document_id) is None:
            return None
        with self.connect() as connection:
            row = connection.execute(
                """SELECT page_number, text, width, height FROM document_pages
                   WHERE document_id = ? AND page_number = ?""",
                (document_id, page_number),
            ).fetchone()
        return dict(row) if row is not None else None

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
