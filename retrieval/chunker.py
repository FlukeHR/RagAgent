from __future__ import annotations

from dataclasses import dataclass

from retrieval.loader import PaperDocument


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    paper_title: str
    section: str
    content: str
    source: str
    page_start: int | None = None
    page_end: int | None = None
    element_type: str = "text"
    modality: str = "text"
    bbox: tuple[float, float, float, float] | None = None
    chunk_context: str | None = None
    heading_path: str | None = None


class PaperChunker:
    """按字符窗口切块，尽量在段落/句子边界断开，并保留章节元数据。"""

    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 150) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, docs: list[PaperDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in docs:
            idx = 0
            for section in doc.sections:
                for piece in self._split_text(section.text):
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc.paper_id}::{idx}",
                            paper_id=doc.paper_id,
                            paper_title=doc.title,
                            section=section.title,
                            content=piece,
                            source=doc.source,
                            page_start=section.page_start,
                            page_end=section.page_end,
                            modality=section.modality,
                            chunk_context=self._context(
                                doc.title,
                                section.title,
                                section.page_start,
                                section.page_end,
                                "text",
                                section.modality,
                            ),
                            heading_path=section.heading_path or section.title,
                        )
                    )
                    idx += 1
        return chunks

    def split_elements(self, docs: list[PaperDocument]) -> list[Chunk]:
        """Create chunks from parsed tables, figures, formulas and layout sidecars."""
        chunks: list[Chunk] = []
        for doc in docs:
            for element in doc.elements or []:
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.paper_id}::{element.element_type}::{element.element_id}",
                        paper_id=doc.paper_id,
                        paper_title=doc.title,
                        section=self._element_section(element.element_type, element.page_start),
                        content=element.content,
                        source=doc.source,
                        page_start=element.page_start,
                        page_end=element.page_end,
                        element_type=element.element_type,
                        modality=element.modality,
                        bbox=element.bbox,
                        chunk_context=self._context(
                            doc.title,
                            self._element_section(element.element_type, element.page_start),
                            element.page_start,
                            element.page_end,
                            element.element_type,
                            element.modality,
                        ),
                        heading_path=self._element_section(element.element_type, element.page_start),
                    )
                )
        return chunks

    def split_pages(self, docs: list[PaperDocument]) -> list[Chunk]:
        """Create page-level text chunks for hybrid page + semantic retrieval."""
        chunks: list[Chunk] = []
        for doc in docs:
            for page in doc.pages or []:
                text = page.text.strip()
                if text:
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc.paper_id}::page::{page.page_number}",
                            paper_id=doc.paper_id,
                            paper_title=doc.title,
                            section=f"Page {page.page_number}",
                            content=text,
                            source=doc.source,
                            page_start=page.page_number,
                            page_end=page.page_number,
                            element_type="page",
                            modality="text",
                            bbox=(page.blocks[0].bbox if page.blocks and page.blocks[0].bbox else None),
                            chunk_context=self._context(
                                doc.title, f"Page {page.page_number}", page.page_number, page.page_number, "page", "text"
                            ),
                            heading_path=f"Page {page.page_number}",
                        )
                    )
                if page.ocr_text and page.ocr_text.strip():
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc.paper_id}::ocr::{page.page_number}",
                            paper_id=doc.paper_id,
                            paper_title=doc.title,
                            section=f"Page {page.page_number} OCR",
                            content=page.ocr_text.strip(),
                            source=doc.source,
                            page_start=page.page_number,
                            page_end=page.page_number,
                            element_type="ocr",
                            modality="ocr",
                            chunk_context=self._context(
                                doc.title, f"Page {page.page_number} OCR", page.page_number, page.page_number, "ocr", "ocr"
                            ),
                            heading_path=f"Page {page.page_number} OCR",
                        )
                    )
                if page.vlm_summary and page.vlm_summary.strip():
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc.paper_id}::vlm::{page.page_number}",
                            paper_id=doc.paper_id,
                            paper_title=doc.title,
                            section=f"Page {page.page_number} VLM summary",
                            content=page.vlm_summary.strip(),
                            source=doc.source,
                            page_start=page.page_number,
                            page_end=page.page_number,
                            element_type="vlm",
                            modality="vlm",
                            chunk_context=self._context(
                                doc.title,
                                f"Page {page.page_number} VLM summary",
                                page.page_number,
                                page.page_number,
                                "vlm",
                                "vlm",
                            ),
                            heading_path=f"Page {page.page_number} VLM summary",
                        )
                    )
        return chunks

    @staticmethod
    def _context(
        paper_title: str,
        section: str,
        page_start: int | None,
        page_end: int | None,
        element_type: str,
        modality: str,
    ) -> str:
        page = ""
        if page_start is not None and page_end is not None:
            page = f"第 {page_start} 页" if page_start == page_end else f"第 {page_start}-{page_end} 页"
        elif page_start is not None:
            page = f"第 {page_start} 页"
        bits = [f"《{paper_title}》", section]
        if page:
            bits.append(page)
        bits.append(f"{element_type}/{modality}")
        return "，".join(bits)

    @staticmethod
    def _element_section(element_type: str, page: int) -> str:
        label = {
            "table": "Table",
            "figure": "Figure",
            "formula": "Formula",
            "text_block": "Text block",
            "image": "Image",
            "page_image": "Page image",
        }.get(element_type, element_type.title())
        return f"{label} on page {page}"

    def _split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        pieces: list[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + self.chunk_size, n)
            if end < n:
                end = self._soft_boundary(text, start, end)
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            if end >= n:
                break
            start = max(end - self.chunk_overlap, start + 1)
        return pieces

    def _soft_boundary(self, text: str, start: int, end: int) -> int:
        """在窗口尾部回退到最近的段落/句号边界，避免切断句子。"""
        window = text[start:end]
        for sep in ("\n\n", "\n", ". ", "。"):
            pos = window.rfind(sep)
            # 只在边界不至于让块过短时采用
            if pos != -1 and pos >= int(self.chunk_size * 0.5):
                return start + pos + len(sep)
        return end
