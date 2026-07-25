from __future__ import annotations

from dataclasses import dataclass, field


BBox = tuple[float, float, float, float]


@dataclass
class TextBlock:
    """A bounded text block extracted from one PDF page."""

    page_number: int
    text: str
    bbox: BBox | None = None


@dataclass
class PaperPage:
    """Canonical page model shared by parsing, loading, chunking and tools."""

    page_number: int
    text: str
    is_scanned_like: bool = False
    ocr_text: str | None = None
    vlm_summary: str | None = None
    blocks: list[TextBlock] = field(default_factory=list)

    @property
    def primary_text(self) -> str:
        """Prefer native text; OCR/VLM become fallbacks instead of duplicate section text."""

        if self.text.strip():
            return self.text.strip()
        if self.ocr_text and self.ocr_text.strip():
            return self.ocr_text.strip()
        if self.vlm_summary and self.vlm_summary.strip():
            return self.vlm_summary.strip()
        return ""

    @property
    def indexed_text(self) -> str:
        """Full page view used by explicit page-reading tools."""

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
    """Canonical table/figure/formula/layout element."""

    element_id: str
    element_type: str
    page_start: int
    page_end: int
    text: str = ""
    modality: str = "text"
    bbox: BBox | None = None
    caption: str | None = None
    summary: str | None = None
    source: str = "sidecar"

    @property
    def content(self) -> str:
        parts: list[str] = []
        if self.caption:
            parts.append(f"Caption: {self.caption.strip()}")
        if self.summary:
            parts.append(f"Summary: {self.summary.strip()}")
        if self.text:
            parts.append(self.text.strip())
        return "\n".join(part for part in parts if part)


@dataclass
class PaperSection:
    title: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    modality: str = "text"
    heading_path: str | None = None


@dataclass
class PaperDocument:
    paper_id: str
    title: str
    source: str
    sections: list[PaperSection]
    pages: list[PaperPage] = field(default_factory=list)
    elements: list[PaperElement] = field(default_factory=list)


@dataclass
class ParsedPDF:
    pages: list[PaperPage]
    elements: list[PaperElement] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)


@dataclass
class PDFPageImage:
    page_number: int
    mime_type: str
    data: bytes
    width: int
    height: int
