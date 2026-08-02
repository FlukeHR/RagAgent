from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import fitz
import requests

from config.settings import Settings
from indexing.build_index import build_index
from retrieval.documents import PaperRepository, arxiv_storage_id, normalize_arxiv_id
from services.app_store import AppStore
from services.user_scope import AgentPool, scoped_settings, user_paths


class UserIngestManager:
    """Single-worker PDF/arXiv ingestion for user-scoped local libraries."""

    def __init__(self, settings: Settings, store: AppStore, agents: AgentPool) -> None:
        self.settings = settings
        self.store = store
        self.agents = agents
        self._library_lock = threading.RLock()
        self.executor = ThreadPoolExecutor(
            max_workers=settings.mineru.max_concurrent_jobs,
            thread_name_prefix="user-mineru-ingest",
        )
        self.store.fail_incomplete_jobs()

    def queue_paper(self, user_id: str, paper_id: str) -> dict[str, Any]:
        """Queue a validated local PDF for MinerU parsing and indexing."""

        self._check_capacity(user_id)
        paper = self.store.get_paper(user_id, paper_id)
        if paper is None:
            raise KeyError("unknown paper")
        job = self.store.create_job(user_id, paper_id)
        self.executor.submit(self._run, user_id, str(job["job_id"]))
        return job

    def queue_arxiv(
        self,
        user_id: str,
        proposal_id: str,
        query: str,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        """Create one private paper record and queue a confirmed arXiv download."""

        self._check_capacity(user_id)
        if self.store.paper_count(user_id) >= self.settings.app.max_papers_per_user:
            raise RuntimeError("paper library quota reached")
        aid = normalize_arxiv_id(str(candidate["arxiv_id"]))
        paper = self.store.create_paper(
            user_id,
            arxiv_storage_id(aid),
            str(candidate.get("title") or aid),
            f"{arxiv_storage_id(aid)}.pdf",
            "arxiv",
            arxiv_id=aid,
            status="queued",
        )
        job = self.store.create_job(
            user_id,
            str(paper["paper_id"]),
            proposal_id=proposal_id,
            query=query,
            arxiv_id=aid,
        )
        self.executor.submit(self._run, user_id, str(job["job_id"]))
        return job

    def retry(self, user_id: str, job_id: str) -> dict[str, Any]:
        """Queue a replacement job for a failed paper without changing ownership."""

        existing = self.store.get_job(user_id, job_id)
        if existing is None:
            raise KeyError("unknown ingest job")
        if existing["status"] != "failed":
            raise ValueError("only failed jobs can be retried")
        paper_id = str(existing["paper_id"])
        self.store.update_paper(user_id, paper_id, status="queued", error=None)
        self._check_capacity(user_id)
        job = self.store.create_job(
            user_id,
            paper_id,
            proposal_id=existing.get("proposal_id"),
            query=str(existing.get("query") or ""),
            arxiv_id=existing.get("arxiv_id"),
        )
        self.executor.submit(self._run, user_id, str(job["job_id"]))
        return job

    def delete_paper(self, user_id: str, paper_id: str) -> None:
        """Delete an explicitly selected user paper and rebuild only that user's index."""

        paper = self.store.get_paper(user_id, paper_id)
        if paper is None:
            raise KeyError("unknown paper")
        if paper["status"] in {"queued", "parsing", "indexing", "deleting"}:
            raise ValueError("paper is busy")
        self.store.update_paper(user_id, paper_id, status="deleting", error=None)
        settings = scoped_settings(self.settings, user_id)
        paths = user_paths(self.settings, user_id)
        repository = PaperRepository(paths.papers)
        target = repository.target(str(paper["storage_id"]))
        with self._library_lock:
            target.unlink(missing_ok=True)
            target.with_suffix(".mineru.json").unlink(missing_ok=True)
            self.store.delete_paper_record(user_id, paper_id)
            ready_files = [
                item
                for item in repository.iter_files((".pdf",))
                if item.with_suffix(".mineru.json").exists()
            ]
            if ready_files:
                build_index(settings, incremental=False)
            else:
                self._clear_flat_index(paths.indexes)
            self.agents.invalidate(user_id)

    def _run(self, user_id: str, job_id: str) -> None:
        job = self.store.get_job(user_id, job_id)
        if job is None:
            return
        paper_id = str(job["paper_id"])
        paper = self.store.get_paper(user_id, paper_id)
        if paper is None:
            return
        settings = scoped_settings(self.settings, user_id)
        paths = user_paths(self.settings, user_id)
        target = PaperRepository(paths.papers).target(str(paper["storage_id"]))
        try:
            with self._library_lock:
                self._set_stage(user_id, paper_id, job_id, "parsing")
                if job.get("arxiv_id") and not target.exists():
                    self._download_arxiv(str(job["arxiv_id"]), target)
                page_count = self._validate_pdf(target)
                count = build_index(
                    settings,
                    incremental=True,
                    progress=lambda stage: self._set_stage(
                        user_id, paper_id, job_id, stage
                    ),
                )
                self.store.update_paper(
                    user_id,
                    paper_id,
                    status="ready",
                    error=None,
                    page_count=page_count,
                )
                self.store.update_job(
                    user_id,
                    job_id,
                    "succeeded",
                    result={"indexed_chunks": count},
                )
                self.agents.invalidate(user_id)
        except Exception as exc:  # noqa: BLE001 - persisted as a safe product error
            message = self._safe_error(exc)
            self.store.update_paper(user_id, paper_id, status="failed", error=message)
            self.store.update_job(user_id, job_id, "failed", error=message)

    def _set_stage(self, user_id: str, paper_id: str, job_id: str, stage: str) -> None:
        if stage not in {"parsing", "indexing"}:
            return
        self.store.update_paper(user_id, paper_id, status=stage, error=None)
        self.store.update_job(user_id, job_id, stage)

    def _check_capacity(self, user_id: str) -> None:
        user_count, global_count = self.store.pending_job_counts(user_id)
        if user_count >= self.settings.app.max_pending_jobs_per_user:
            raise RuntimeError("user ingest queue is full")
        if global_count >= self.settings.mineru.max_pending_jobs:
            raise RuntimeError("global ingest queue is full")

    def _download_arxiv(self, arxiv_id: str, target: Path) -> None:
        max_bytes = int(self.settings.arxiv.max_pdf_mb * 1024 * 1024)
        part = target.with_suffix(".pdf.part")
        try:
            with requests.get(
                f"https://arxiv.org/pdf/{normalize_arxiv_id(arxiv_id)}",
                stream=True,
                timeout=self.settings.arxiv.request_timeout_seconds,
                headers={"User-Agent": "paper-rag-agent/1.0"},
            ) as response:
                response.raise_for_status()
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > max_bytes:
                    raise ValueError("arXiv PDF exceeds size limit")
                size = 0
                with part.open("wb") as handle:
                    for block in response.iter_content(chunk_size=1 << 16):
                        if not block:
                            continue
                        size += len(block)
                        if size > max_bytes:
                            raise ValueError("arXiv PDF exceeds size limit")
                        handle.write(block)
                with part.open("rb") as handle:
                    magic = handle.read(5)
                if magic != b"%PDF-":
                    raise ValueError("arXiv response is not a PDF")
                os.replace(part, target)
        finally:
            part.unlink(missing_ok=True)

    def _validate_pdf(self, path: Path) -> int:
        if not path.exists() or path.stat().st_size > self.settings.mineru.max_pdf_mb * 1024 * 1024:
            raise ValueError("PDF is missing or exceeds size limit")
        with path.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise ValueError("file is not a PDF")
        with fitz.open(path) as document:
            page_count = int(document.page_count)
        if page_count <= 0 or page_count > self.settings.mineru.max_pages:
            raise ValueError("PDF page count is outside the configured limit")
        return page_count

    @staticmethod
    def _clear_flat_index(index_dir: Path) -> None:
        root = index_dir.resolve()
        if not root.exists():
            return
        for item in root.iterdir():
            if item.is_file() and item.resolve().parent == root:
                item.unlink(missing_ok=True)

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        if isinstance(exc, ValueError):
            return str(exc)[:500]
        name = type(exc).__name__
        if "MinerU" in name or "MinerU" in str(exc):
            return "MinerU 解析失败，请确认本地解析服务可用后重试"
        if isinstance(exc, requests.RequestException):
            return "arXiv 下载失败，请检查网络后重试"
        return f"入库失败（{name}）"
