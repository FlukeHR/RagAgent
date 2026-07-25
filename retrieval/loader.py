from __future__ import annotations

from pathlib import Path
from typing import Iterable

from retrieval.models import (
    PDFPageImage,
    PaperDocument,
    PaperElement,
    PaperPage as PageText,
    PaperSection,
)
from retrieval.pdf_service import PDFService
from retrieval.pdf_parse import PDFParseProvider, provider_from_config
from retrieval.normalizer import DocumentNormalizer
from retrieval.repository import PaperRepository


class PaperLoader:
    """加载并解析本地论文（PDF / txt / md），按章节切分。"""

    min_scan_text_chars = 50

    def __init__(
        self,
        root_dir: str,
        include_suffixes: tuple[str, ...] = (".pdf", ".txt", ".md"),
        pdf_provider: PDFParseProvider | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.include_suffixes = include_suffixes
        self.pdf_provider = pdf_provider
        self.repository = PaperRepository(root_dir)
        self.normalizer = DocumentNormalizer()

    def iter_files(self) -> Iterable[Path]:
        yield from self.repository.iter_files(self.include_suffixes)

    def load(self) -> list[PaperDocument]:
        docs: list[PaperDocument] = []
        for file in self.iter_files():
            doc = self.load_file(file)
            if doc is not None:
                docs.append(doc)
        return docs

    def load_file(self, file: Path) -> PaperDocument | None:
        pages: list[PageText] | None = None
        elements: list[PaperElement] | None = None
        if file.suffix.lower() == ".pdf":
            parsed = self._parse_pdf(file, self.pdf_provider)
            pages = parsed[0]
            elements = parsed[1]
            text = "\n".join(
                part
                for part in (
                    "\n".join(p.indexed_text for p in pages),
                    "\n".join(e.content for e in elements or []),
                )
                if part.strip()
            )
            sections = self.normalizer.sections_from_pages(pages)
        else:
            text = self._read(file)
            sections = self.normalizer.sections_from_text(text)
        if not text or not text.strip():
            return None
        return PaperDocument(
            paper_id=file.stem,
            title=self.normalizer.title(text, fallback=file.stem),
            source=str(file),
            sections=sections,
            pages=pages or [],
            elements=elements or [],
        )

    # ---------- 读取 ----------
    def _read(self, file: Path) -> str:
        if file.suffix.lower() == ".pdf":
            return self._read_pdf(file)
        try:
            return file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return file.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _read_pdf(file: Path) -> str:
        return "\n".join(p.indexed_text for p in PaperLoader._read_pdf_pages(file))

    @staticmethod
    def _read_pdf_pages(file: Path) -> list[PageText]:
        return PaperLoader._parse_pdf(file, None)[0]

    @staticmethod
    def _parse_pdf(
        file: Path, provider: PDFParseProvider | None
    ) -> tuple[list[PageText], list[PaperElement]]:
        parsed = (provider or provider_from_config()).parse(file)
        return parsed.pages, parsed.elements

    @staticmethod
    def render_pdf_page(
        file: Path,
        page_number: int,
        max_side: int = 1600,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> PDFPageImage:
        """Render one PDF page as a bounded image for OCR/VLM follow-up tools."""
        return PDFService.render_page(file, page_number, max_side=max_side, bbox=bbox)

    # Compatibility wrappers; normalization now lives in DocumentNormalizer.
    def _extract_title(self, text: str, fallback: str) -> str:
        return self.normalizer.title(text, fallback)

    def _split_sections(self, text: str) -> list[PaperSection]:
        return self.normalizer.sections_from_text(text)

    def _split_sections_from_pages(self, pages: list[PageText]) -> list[PaperSection]:
        return self.normalizer.sections_from_pages(pages)

    def _is_heading(self, line: str) -> bool:
        return self.normalizer.is_heading(line)
