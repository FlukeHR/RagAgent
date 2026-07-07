from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from retrieval.pdf_parse import ParsedBlock, PDFParseProvider, provider_from_config


@dataclass
class PaperSection:
    title: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    modality: str = "text"
    heading_path: str | None = None


@dataclass
class PageText:
    page_number: int
    text: str
    is_scanned_like: bool = False
    ocr_text: str | None = None
    vlm_summary: str | None = None
    blocks: list[ParsedBlock] | None = None

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
class PDFPageImage:
    page_number: int
    mime_type: str
    data: bytes
    width: int
    height: int


@dataclass
class PaperElement:
    element_id: str
    element_type: str
    page_start: int
    page_end: int
    content: str
    modality: str = "text"
    bbox: tuple[float, float, float, float] | None = None
    caption: str | None = None
    summary: str | None = None


@dataclass
class PaperDocument:
    paper_id: str          # 论文标识，默认取文件名（不含扩展名）
    title: str             # 论文标题
    source: str            # 原始文件路径
    sections: list[PaperSection]
    pages: list[PageText] | None = None
    elements: list[PaperElement] | None = None


# 常见论文章节关键词（小写匹配）
_SECTION_KEYWORDS = (
    "abstract",
    "introduction",
    "related work",
    "background",
    "preliminaries",
    "method",
    "methodology",
    "approach",
    "model",
    "architecture",
    "experiment",
    "experiments",
    "experimental setup",
    "results",
    "evaluation",
    "analysis",
    "ablation",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "future work",
    "references",
    "acknowledgment",
    "acknowledgments",
    "acknowledgements",
    "appendix",
)

# 形如 "1 Introduction" / "2.1 Method" / "3. Results"
_NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z].{0,70}$")


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

    def iter_files(self) -> Iterable[Path]:
        if not self.root_dir.exists():
            return
        for file in sorted(self.root_dir.rglob("*")):
            if file.is_file() and file.suffix.lower() in self.include_suffixes:
                yield file

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
            sections = self._split_sections_from_pages(pages)
        else:
            text = self._read(file)
            sections = self._split_sections(text)
        if not text or not text.strip():
            return None
        return PaperDocument(
            paper_id=file.stem,
            title=self._extract_title(text, fallback=file.stem),
            source=str(file),
            sections=sections,
            pages=pages,
            elements=elements,
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
        pages = [
            PageText(
                page_number=p.page_number,
                text=p.text,
                is_scanned_like=p.is_scanned_like,
                ocr_text=p.ocr_text,
                vlm_summary=p.vlm_summary,
                blocks=p.blocks,
            )
            for p in parsed.pages
        ]
        elements = [
            PaperElement(
                element_id=e.element_id,
                element_type=e.element_type,
                page_start=e.page_start,
                page_end=e.page_end,
                content=e.content,
                modality=e.modality,
                bbox=e.bbox,
                caption=e.caption,
                summary=e.summary,
            )
            for e in parsed.elements
        ]
        return pages, elements

    @staticmethod
    def render_pdf_page(
        file: Path,
        page_number: int,
        max_side: int = 1600,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> PDFPageImage:
        """Render one PDF page as a bounded image for OCR/VLM follow-up tools."""
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise RuntimeError(
                "解析 PDF 需要 PyMuPDF，请先安装：pip install pymupdf"
            ) from exc
        with fitz.open(str(file)) as doc:
            if page_number < 1 or page_number > len(doc):
                raise ValueError(f"page_number out of range: {page_number}")
            if max_side <= 0:
                raise ValueError("max_side must be positive")
            page = doc[page_number - 1]
            clip = None
            if bbox is not None:
                clip = fitz.Rect(*bbox) & page.rect
                if clip.is_empty:
                    raise ValueError(f"bbox outside page: {bbox}")
            rect = clip or page.rect
            side = max(float(rect.width), float(rect.height), 1.0)
            zoom = min(2.0, max_side / side)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False, clip=clip)
            try:
                data = pix.tobytes("jpg")
                mime = "image/jpeg"
            except Exception:
                data = pix.tobytes("png")
                mime = "image/png"
            return PDFPageImage(
                page_number=page_number,
                mime_type=mime,
                data=data,
                width=pix.width,
                height=pix.height,
            )

    # ---------- 标题 ----------
    @staticmethod
    def _extract_title(text: str, fallback: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            # 跳过过短或全大写页眉，取第一行像样的标题
            if len(stripped) >= 8 and not stripped.lower().startswith("arxiv"):
                return stripped[:200]
        return fallback

    # ---------- 章节切分 ----------
    def _split_sections(self, text: str) -> list[PaperSection]:
        lines = text.splitlines()
        sections: list[PaperSection] = []
        current_title = "Body"
        current_lines: list[str] = []

        def flush() -> None:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append(PaperSection(title=current_title, text=body))

        for line in lines:
            if self._is_heading(line):
                flush()
                current_title = line.strip()[:80]
                current_lines = []
            else:
                current_lines.append(line)
        flush()

        if not sections:
            sections = [PaperSection(title="Body", text=text.strip())]
        return sections

    def _split_sections_from_pages(self, pages: list[PageText]) -> list[PaperSection]:
        sections: list[PaperSection] = []
        current_title = "Body"
        current_lines: list[str] = []
        current_start: int | None = pages[0].page_number if pages else None
        current_end: int | None = current_start
        current_modality = pages[0].dominant_modality if pages else "text"

        def flush() -> None:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append(
                    PaperSection(
                        title=current_title,
                        text=body,
                        page_start=current_start,
                        page_end=current_end,
                        modality=current_modality,
                        heading_path=current_title,
                    )
                )

        for page in pages:
            page_modality = page.dominant_modality
            for line in page.indexed_text.splitlines():
                if self._is_heading(line):
                    flush()
                    current_title = line.strip()[:80]
                    current_lines = []
                    current_start = page.page_number
                    current_end = page.page_number
                    current_modality = page_modality
                else:
                    if not current_lines:
                        current_modality = page_modality
                    current_lines.append(line)
                    current_end = page.page_number
            if current_lines:
                current_end = page.page_number
        flush()

        if not sections:
            text = "\n".join(p.indexed_text for p in pages).strip()
            if text:
                sections = [
                    PaperSection(
                        title="Body",
                        text=text,
                        page_start=pages[0].page_number if pages else None,
                        page_end=pages[-1].page_number if pages else None,
                        modality=pages[0].dominant_modality if pages else "text",
                        heading_path="Body",
                    )
                ]
        return sections

    @staticmethod
    def _is_heading(line: str) -> bool:
        s = line.strip()
        if not s or len(s) > 80:
            return False
        if _NUMBERED_HEADING.match(s):
            return True
        low = s.lower().rstrip(":.")
        return low in _SECTION_KEYWORDS
