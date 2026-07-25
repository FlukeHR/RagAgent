from __future__ import annotations

import re
from pathlib import Path


_ARXIV_ID = re.compile(
    r"^(?:\d{4}\.\d{4,5}|[A-Za-z][A-Za-z0-9._-]*/\d{7})(?:v\d+)?$"
)


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
        """Return a validated write target inside the flat repository."""

        safe_id = self.validate_id(paper_id)
        if not suffix.startswith(".") or "/" in suffix or "\\" in suffix:
            raise InvalidPaperId("invalid file suffix")
        target = (self.root / f"{safe_id}{suffix}").resolve()
        if target.parent != self.root:
            raise InvalidPaperId("paper path escapes the repository")
        return target

    def iter_files(self, suffixes: tuple[str, ...] = (".pdf", ".txt", ".md")):
        if not self.root.exists():
            return
        allowed = {suffix.lower() for suffix in suffixes}
        for path in sorted(self.root.iterdir()):
            if path.is_file() and path.suffix.lower() in allowed:
                yield path


def normalize_arxiv_id(value: str) -> str:
    """Validate a modern or legacy arXiv identifier without accepting URL/path input."""

    aid = str(value or "").strip()
    if not _ARXIV_ID.fullmatch(aid):
        raise InvalidPaperId(f"invalid arXiv id: {aid!r}")
    return aid


def arxiv_storage_id(aid: str) -> str:
    """Map legacy IDs containing '/' to a flat reversible-enough file identifier."""

    return normalize_arxiv_id(aid).replace("/", "__")
