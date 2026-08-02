from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator

from config.settings import BASE_DIR, Settings


@dataclass(frozen=True)
class IngestProposal:
    proposal_id: str
    query: str
    arxiv_ids: tuple[str, ...]
    expires_at: float
    consumed: bool = False


@dataclass(frozen=True)
class IngestJob:
    job_id: str
    proposal_id: str
    query: str
    arxiv_ids: tuple[str, ...]
    status: str
    created_at: float
    updated_at: float
    error: str | None = None
    result: dict[str, Any] | None = None


class IngestJobStore:
    """SQLite persistence for bounded proposals and asynchronous ingest jobs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.path = (BASE_DIR / settings.mineru.job_db_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS ingest_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    arxiv_ids TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    consumed INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    job_id TEXT PRIMARY KEY,
                    proposal_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    arxiv_ids TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    error TEXT,
                    result TEXT
                );
                """
            )

    def fail_incomplete_jobs(self) -> None:
        """Mark jobs from a previous process as retryable failures."""

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE ingest_jobs
                SET status = 'failed', error = 'application restarted during job',
                    updated_at = ?
                WHERE status IN ('queued', 'parsing', 'indexing')
                """,
                (time.time(),),
            )

    def create_proposal(self, query: str, arxiv_ids: list[str]) -> IngestProposal:
        unique = tuple(dict.fromkeys(arxiv_ids))
        if not unique:
            raise ValueError("proposal requires at least one arXiv ID")
        proposal = IngestProposal(
            proposal_id=uuid.uuid4().hex,
            query=query[:1000],
            arxiv_ids=unique,
            expires_at=time.time() + self.settings.mineru.proposal_ttl_seconds,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingest_proposals
                (proposal_id, query, arxiv_ids, expires_at, consumed)
                VALUES (?, ?, ?, ?, 0)
                """,
                (
                    proposal.proposal_id,
                    proposal.query,
                    json.dumps(proposal.arxiv_ids),
                    proposal.expires_at,
                ),
            )
        return proposal

    def confirm(self, proposal_id: str, selected_ids: list[str]) -> IngestJob:
        selected = tuple(dict.fromkeys(selected_ids))
        if not selected:
            raise ValueError("at least one arXiv ID must be confirmed")
        if len(selected) > self.settings.arxiv.max_ingest_papers:
            raise ValueError("confirmed arXiv IDs exceed the per-job limit")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM ingest_proposals WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
            if row is None:
                raise KeyError("unknown ingest proposal")
            if bool(row["consumed"]):
                raise ValueError("ingest proposal was already consumed")
            if float(row["expires_at"]) <= time.time():
                raise ValueError("ingest proposal expired")
            allowed = set(json.loads(row["arxiv_ids"]))
            if not set(selected).issubset(allowed):
                raise ValueError("confirmed arXiv ID is outside the proposal")
            pending = connection.execute(
                """
                SELECT COUNT(*) FROM ingest_jobs
                WHERE status IN ('queued', 'parsing', 'indexing')
                """
            ).fetchone()[0]
            if int(pending) >= self.settings.mineru.max_pending_jobs:
                raise RuntimeError("ingest job queue is full")
            now = time.time()
            job = IngestJob(
                job_id=uuid.uuid4().hex,
                proposal_id=proposal_id,
                query=str(row["query"]),
                arxiv_ids=selected,
                status="queued",
                created_at=now,
                updated_at=now,
            )
            connection.execute(
                "UPDATE ingest_proposals SET consumed = 1 WHERE proposal_id = ?",
                (proposal_id,),
            )
            connection.execute(
                """
                INSERT INTO ingest_jobs
                (job_id, proposal_id, query, arxiv_ids, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.proposal_id,
                    job.query,
                    json.dumps(job.arxiv_ids),
                    job.status,
                    now,
                    now,
                ),
            )
        return job

    def get_job(self, job_id: str) -> IngestJob | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ingest_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            return None
        return IngestJob(
            job_id=str(row["job_id"]),
            proposal_id=str(row["proposal_id"]),
            query=str(row["query"]),
            arxiv_ids=tuple(json.loads(row["arxiv_ids"])),
            status=str(row["status"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            error=str(row["error"]) if row["error"] else None,
            result=json.loads(row["result"]) if row["result"] else None,
        )

    def update_job(
        self,
        job_id: str,
        status: str,
        *,
        error: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        if status not in {"queued", "parsing", "indexing", "succeeded", "failed"}:
            raise ValueError(f"invalid ingest status: {status}")
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE ingest_jobs
                SET status = ?, updated_at = ?, error = ?, result = ?
                WHERE job_id = ?
                """,
                (
                    status,
                    time.time(),
                    error[:1000] if error else None,
                    json.dumps(result, ensure_ascii=False) if result is not None else None,
                    job_id,
                ),
            )

    def retry(self, job_id: str) -> IngestJob:
        existing = self.get_job(job_id)
        if existing is None:
            raise KeyError("unknown ingest job")
        if existing.status != "failed":
            raise ValueError("only failed jobs can be retried")
        now = time.time()
        retried = IngestJob(
            job_id=uuid.uuid4().hex,
            proposal_id=existing.proposal_id,
            query=existing.query,
            arxiv_ids=existing.arxiv_ids,
            status="queued",
            created_at=now,
            updated_at=now,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingest_jobs
                (job_id, proposal_id, query, arxiv_ids, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    retried.job_id,
                    retried.proposal_id,
                    retried.query,
                    json.dumps(retried.arxiv_ids),
                    retried.status,
                    now,
                    now,
                ),
            )
        return retried


class IngestJobManager:
    """Run confirmed ingest jobs directly through the bounded library service."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = IngestJobStore(settings)
        self.store.fail_incomplete_jobs()
        self.executor = ThreadPoolExecutor(
            max_workers=settings.mineru.max_concurrent_jobs,
            thread_name_prefix="mineru-ingest",
        )

    def confirm(self, proposal_id: str, arxiv_ids: list[str]) -> IngestJob:
        job = self.store.confirm(proposal_id, arxiv_ids)
        self.executor.submit(self._run, job.job_id)
        return job

    def retry(self, job_id: str) -> IngestJob:
        job = self.store.retry(job_id)
        self.executor.submit(self._run, job.job_id)
        return job

    def _run(self, job_id: str) -> None:
        from services.library_service import PaperLibraryService

        job = self.store.get_job(job_id)
        if job is None:
            return
        try:
            self.store.update_job(job_id, "parsing")
            report = PaperLibraryService(self.settings).ingest_arxiv(
                list(job.arxiv_ids),
                progress=lambda status: self.store.update_job(job_id, status),
            )
            if report.failed:
                failed = ", ".join(report.failed)
                raise RuntimeError(f"arXiv ingest failed for: {failed}")
            self.store.update_job(job_id, "succeeded", result=asdict(report))
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.store.update_job(job_id, "failed", error=message)


def job_to_dict(job: IngestJob) -> dict[str, Any]:
    """Serialize a persisted job for API responses."""

    payload = asdict(job)
    payload["arxiv_ids"] = list(job.arxiv_ids)
    return payload
