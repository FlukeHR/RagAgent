from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz

from config.settings import BASE_DIR, Settings
from services.app_store import AppStore


_SAFE_ID = re.compile(r"^[a-f0-9]{32}$")
_ASCII_WORD = re.compile(r"[a-z0-9]{2,}")
_CJK_RUN = re.compile(r"[\u3400-\u9fff]+")


class DocumentService:
    """Store small PDFs as searchable text plus rendered page images."""

    def __init__(self, settings: Settings, store: AppStore) -> None:
        self.settings = settings
        self.store = store
        configured = Path(settings.app.documents_root)
        self.root = (configured if configured.is_absolute() else BASE_DIR / configured).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def ingest(self, user_id: str, filename: str, content: bytes) -> dict[str, Any]:
        """Validate, render, and publish one bounded text-based PDF."""

        if not content.startswith(b"%PDF-"):
            raise ValueError("文件不是有效的 PDF")
        if len(content) > self.settings.app.max_pdf_mb * 1024 * 1024:
            raise ValueError("PDF 文件过大")
        safe_name = Path(filename or "document.pdf").name[:200]
        record = self.store.create_document(user_id, safe_name)
        document_id = str(record["document_id"])
        directory = self._document_dir(user_id, document_id)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "source.pdf").write_bytes(content)
        try:
            with fitz.open(stream=content, filetype="pdf") as pdf:
                if not 0 < pdf.page_count <= self.settings.app.max_pdf_pages:
                    raise ValueError(
                        f"PDF 页数必须在 1 到 {self.settings.app.max_pdf_pages} 之间"
                    )
                pages: list[tuple[int, str, int, int]] = []
                for page_number, page in enumerate(pdf, start=1):
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                    (directory / f"page-{page_number}.png").write_bytes(
                        pixmap.tobytes("png")
                    )
                    pages.append(
                        (
                            page_number,
                            page.get_text("text", sort=True)[:50000],
                            pixmap.width,
                            pixmap.height,
                        )
                    )
            self.store.finish_document(user_id, document_id, pages)
        except Exception as exc:
            self.store.fail_document(user_id, document_id, type(exc).__name__)
            if isinstance(exc, ValueError):
                raise
            raise ValueError("PDF 解析失败") from exc
        result = self.store.get_document(user_id, document_id)
        assert result is not None
        return result

    def search_pages(
        self,
        user_id: str,
        document_id: str,
        query: str,
        exclude: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the most lexically relevant pages from one small document."""

        pages = [
            page
            for page in self.store.list_document_pages(user_id, document_id)
            if int(page["page_number"]) not in (exclude or set())
        ]
        if not pages:
            return []
        terms = _search_terms(query)
        for page in pages:
            text = str(page["text"]).lower()
            page["score"] = sum(text.count(term) for term in terms)
        # ponytail: linear scan is intentional for the 30-page ceiling; add FTS only if raised.
        return sorted(
            pages,
            key=lambda page: (int(page["score"]), -int(page["page_number"])),
            reverse=True,
        )[: self.settings.app.candidate_pages]

    def page_image(
        self, user_id: str, document_id: str, page_number: int
    ) -> Path | None:
        """Resolve an owned rendered page without accepting a path from the caller."""

        if self.store.get_document_page(user_id, document_id, page_number) is None:
            return None
        path = self._document_dir(user_id, document_id) / f"page-{page_number}.png"
        return path if path.is_file() else None

    def _document_dir(self, user_id: str, document_id: str) -> Path:
        if not _SAFE_ID.fullmatch(user_id) or not _SAFE_ID.fullmatch(document_id):
            raise ValueError("invalid document owner or id")
        user_root = (self.root / user_id).resolve()
        if user_root.parent != self.root:
            raise ValueError("user path escapes document root")
        target = (user_root / document_id).resolve()
        if target.parent != user_root:
            raise ValueError("document path escapes user root")
        return target


def _search_terms(value: str) -> set[str]:
    """Extract cheap English words and CJK bigrams for bounded page ranking."""

    lowered = value.lower()
    terms = set(_ASCII_WORD.findall(lowered))
    for run in _CJK_RUN.findall(lowered):
        terms.update(run[index : index + 2] for index in range(len(run) - 1))
    return terms
