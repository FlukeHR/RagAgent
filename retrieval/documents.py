from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


BBox = tuple[float, float, float, float]
_ARXIV_ID = re.compile(
    r"^(?:\d{4}\.\d{4,5}|[A-Za-z][A-Za-z0-9._-]*/\d{7})(?:v\d+)?$"
)


@dataclass
class TextBlock:
    """A bounded text block extracted from one PDF page."""

    page_number: int
    text: str
    bbox: BBox | None = None


@dataclass
class PaperPage:
    """Canonical page model shared by parsing, loading, chunking, and tools."""

    page_number: int
    text: str
    is_scanned_like: bool = False
    ocr_text: str | None = None
    vlm_summary: str | None = None
    blocks: list[TextBlock] = field(default_factory=list)

    @property
    def primary_text(self) -> str:
        if self.text.strip():
            return self.text.strip()
        if self.ocr_text and self.ocr_text.strip():
            return self.ocr_text.strip()
        if self.vlm_summary and self.vlm_summary.strip():
            return self.vlm_summary.strip()
        return ""

    @property
    def indexed_text(self) -> str:
        parts: list[str] = []
        if self.text.strip():
            parts.append(self.text.strip())
        if self.ocr_text and self.ocr_text.strip():
            parts.append(f"[OCR]\n{self.ocr_text.strip()}")
        if self.vlm_summary and self.vlm_summary.strip():
            parts.append(f"[VLM]\n{self.vlm_summary.strip()}")
        return "\n\n".join(parts)

    @property
    def dominant_modality(self) -> str:
        if self.text.strip():
            return "text"
        if self.ocr_text and self.ocr_text.strip():
            return "ocr"
        if self.vlm_summary and self.vlm_summary.strip():
            return "vlm"
        return "text"


@dataclass
class PaperElement:
    """Canonical table, figure, formula, or layout element."""

    element_id: str
    element_type: str
    page_start: int
    page_end: int
    text: str = ""
    markdown: str | None = None
    modality: str = "text"
    bbox: BBox | None = None
    caption: str | None = None
    footnote: str | None = None
    summary: str | None = None
    source: str = "sidecar"
    heading_path: str | None = None
    order: int = 0

    @property
    def content(self) -> str:
        parts: list[str] = []
        if self.caption:
            parts.append(f"Caption: {self.caption.strip()}")
        if self.summary:
            parts.append(f"Summary: {self.summary.strip()}")
        body = self.markdown or self.text
        if body and (not self.summary or body.strip() != self.summary.strip()):
            parts.append(body.strip())
        if self.footnote:
            parts.append(f"Footnote: {self.footnote.strip()}")
        return "\n".join(dict.fromkeys(part for part in parts if part))


@dataclass
class PaperSection:
    title: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    modality: str = "text"
    heading_path: str | None = None
    heading_level: int | None = None


@dataclass
class PaperDocument:
    paper_id: str
    title: str
    source: str
    sections: list[PaperSection]
    pages: list[PaperPage] = field(default_factory=list)
    elements: list[PaperElement] = field(default_factory=list)
    parser_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedPDF:
    pages: list[PaperPage]
    sections: list[PaperSection] = field(default_factory=list)
    elements: list[PaperElement] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    parser_metadata: dict[str, Any] = field(default_factory=dict)


class InvalidPaperId(ValueError):
    """Raised when an identifier could escape or bypass the paper repository."""


class PaperRepository:
    """Resolve paper identifiers inside one flat, bounded paper directory."""

    def __init__(self, root: str | Path, max_id_chars: int = 160) -> None:
        self.root = Path(root).resolve()
        self.max_id_chars = max_id_chars

    def validate_id(self, paper_id: str) -> str:
        value = str(paper_id or "").strip()
        if not value or len(value) > self.max_id_chars:
            raise InvalidPaperId("paper_id is empty or too long")
        if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
            raise InvalidPaperId("paper_id must be a flat file identifier")
        if Path(value).name != value:
            raise InvalidPaperId("paper_id contains an invalid path component")
        return value

    def resolve(
        self,
        paper_id: str,
        suffixes: tuple[str, ...] = (".pdf", ".txt", ".md"),
        *,
        must_exist: bool = True,
    ) -> Path | None:
        safe_id = self.validate_id(paper_id)
        for suffix in suffixes:
            candidate = (self.root / f"{safe_id}{suffix}").resolve()
            if candidate.parent != self.root:
                raise InvalidPaperId("paper path escapes the repository")
            if not must_exist or candidate.exists():
                return candidate
        return None

    def target(self, paper_id: str, suffix: str = ".pdf") -> Path:
        safe_id = self.validate_id(paper_id)
        if not suffix.startswith(".") or "/" in suffix or "\\" in suffix:
            raise InvalidPaperId("invalid file suffix")
        target = (self.root / f"{safe_id}{suffix}").resolve()
        if target.parent != self.root:
            raise InvalidPaperId("paper path escapes the repository")
        return target

    def iter_files(
        self, suffixes: tuple[str, ...] = (".pdf", ".txt", ".md")
    ) -> Iterable[Path]:
        if not self.root.exists():
            return
        allowed = {suffix.lower() for suffix in suffixes}
        for path in sorted(self.root.iterdir()):
            if path.is_file() and path.suffix.lower() in allowed:
                yield path


def normalize_arxiv_id(value: str) -> str:
    """Validate a modern or legacy arXiv identifier without URL/path input."""

    aid = str(value or "").strip()
    if not _ARXIV_ID.fullmatch(aid):
        raise InvalidPaperId(f"invalid arXiv id: {aid!r}")
    return aid


def arxiv_storage_id(aid: str) -> str:
    """Map legacy arXiv IDs to a flat file identifier."""

    return normalize_arxiv_id(aid).replace("/", "__")
