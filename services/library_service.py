from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import BASE_DIR, Settings
from indexing.build_index import build_index
from indexing.prune import prune_library, touch_papers
from retrieval.repository import (
    PaperRepository,
    arxiv_storage_id,
    normalize_arxiv_id,
)


_LIBRARY_LOCK = threading.Lock()


@dataclass
class IngestReport:
    downloaded: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    storage_ids: list[str] = field(default_factory=list)
    indexed_chunks: int = 0


class PaperLibraryService:
    """Download validated PDFs and atomically update the unified paper index."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.data_dir = BASE_DIR / settings.project.data_root
        self.index_dir = BASE_DIR / settings.index.index_root
        self.repository = PaperRepository(self.data_dir)

    def ingest_arxiv(self, arxiv_ids: list[str]) -> IngestReport:
        normalized: list[str] = []
        for value in arxiv_ids[: self.settings.arxiv.max_ingest_papers]:
            aid = normalize_arxiv_id(value)
            if aid not in normalized:
                normalized.append(aid)

        report = IngestReport(storage_ids=[arxiv_storage_id(aid) for aid in normalized])
        if not normalized:
            return report
        with _LIBRARY_LOCK:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            for aid, storage_id in zip(normalized, report.storage_ids):
                target = self.repository.target(storage_id)
                if target.exists():
                    report.reused.append(aid)
                elif self._download_pdf(aid, target):
                    report.downloaded.append(aid)
                else:
                    report.failed.append(aid)

            if report.downloaded or not (self.index_dir / "manifest.json").exists():
                if any(self.repository.iter_files((".pdf",))):
                    report.indexed_chunks = build_index(
                        self.settings,
                        incremental=True,
                    )
            else:
                from retrieval.vector_store import VectorStore

                store = VectorStore(str(self.index_dir))
                store.load()
                report.indexed_chunks = len(store.chunks)

            touched = set(report.storage_ids)
            if touched:
                touch_papers(self.settings, touched)
            prune_library(self.settings, protect=touched)
            if (self.index_dir / "manifest.json").exists():
                from retrieval.vector_store import VectorStore

                current = VectorStore(str(self.index_dir))
                current.load()
                report.indexed_chunks = len(current.chunks)
        return report

    def _download_pdf(self, aid: str, target: Path) -> bool:
        import requests

        max_bytes = int(self.settings.arxiv.max_pdf_mb * 1024 * 1024)
        part = target.with_suffix(target.suffix + ".part")
        try:
            with requests.get(
                f"https://arxiv.org/pdf/{aid}",
                timeout=self.settings.arxiv.request_timeout_seconds,
                stream=True,
                headers={"User-Agent": "paper-rag-agent/1.0"},
            ) as response:
                response.raise_for_status()
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > max_bytes:
                    return False
                size = 0
                first = b""
                with part.open("wb") as handle:
                    for block in response.iter_content(chunk_size=1 << 16):
                        if not block:
                            continue
                        if not first:
                            first = block[:5]
                        size += len(block)
                        if size > max_bytes:
                            return False
                        handle.write(block)
                if first != b"%PDF-":
                    return False
                os.replace(part, target)
                return True
        except (OSError, requests.RequestException, ValueError):
            return False
        finally:
            part.unlink(missing_ok=True)
