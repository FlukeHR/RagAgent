from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from retrieval.models import PaperDocument


def content_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


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
    parent_id: str | None = None
    content_hash: str = ""
    granularity: str = "section"

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = content_hash(self.content)
        if self.parent_id is None:
            page = self.page_start if self.page_start == self.page_end else self.section
            self.parent_id = f"{self.paper_id}::{page}"


class ChunkStrategy(Protocol):
    def build(self, document: PaperDocument, chunker: "PaperChunker") -> list[Chunk]: ...


class SectionChunkStrategy:
    def build(self, document: PaperDocument, chunker: "PaperChunker") -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0
        for section in document.sections:
            for piece in chunker.split_text(section.text):
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.paper_id}::section::{index}",
                        paper_id=document.paper_id,
                        paper_title=document.title,
                        section=section.title,
                        content=piece,
                        source=document.source,
                        page_start=section.page_start,
                        page_end=section.page_end,
                        modality=section.modality,
                        chunk_context=chunker.context(
                            document.title,
                            section.title,
                            section.page_start,
                            section.page_end,
                            "text",
                            section.modality,
                        ),
                        heading_path=section.heading_path or section.title,
                        parent_id=f"{document.paper_id}::section::{section.title}",
                        granularity="section",
                    )
                )
                index += 1
        return chunks


class PageChunkStrategy:
    def build(self, document: PaperDocument, chunker: "PaperChunker") -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in document.pages:
            parent = f"{document.paper_id}::page::{page.page_number}"
            candidates = [
                ("page", "text", page.text.strip()),
                ("ocr", "ocr", (page.ocr_text or "").strip()),
                ("vlm", "vlm", (page.vlm_summary or "").strip()),
            ]
            seen: set[str] = set()
            for element_type, modality, text in candidates:
                if not text:
                    continue
                digest = content_hash(text)
                if digest in seen:
                    continue
                seen.add(digest)
                suffix = "page" if element_type == "page" else element_type
                title = (
                    f"Page {page.page_number}"
                    if element_type == "page"
                    else f"Page {page.page_number} {element_type.upper()}"
                )
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.paper_id}::{suffix}::{page.page_number}",
                        paper_id=document.paper_id,
                        paper_title=document.title,
                        section=title,
                        content=text,
                        source=document.source,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        element_type=element_type,
                        modality=modality,
                        bbox=None,
                        chunk_context=chunker.context(
                            document.title,
                            title,
                            page.page_number,
                            page.page_number,
                            element_type,
                            modality,
                        ),
                        heading_path=title,
                        parent_id=parent,
                        content_hash=digest,
                        granularity="page",
                    )
                )
        return chunks


class ElementChunkStrategy:
    def build(self, document: PaperDocument, chunker: "PaperChunker") -> list[Chunk]:
        chunks: list[Chunk] = []
        for element in document.elements:
            section = chunker.element_section(element.element_type, element.page_start)
            chunks.append(
                Chunk(
                    chunk_id=(
                        f"{document.paper_id}::{element.element_type}::{element.element_id}"
                    ),
                    paper_id=document.paper_id,
                    paper_title=document.title,
                    section=section,
                    content=element.content,
                    source=document.source,
                    page_start=element.page_start,
                    page_end=element.page_end,
                    element_type=element.element_type,
                    modality=element.modality,
                    bbox=element.bbox,
                    chunk_context=chunker.context(
                        document.title,
                        section,
                        element.page_start,
                        element.page_end,
                        element.element_type,
                        element.modality,
                    ),
                    heading_path=section,
                    parent_id=f"{document.paper_id}::page::{element.page_start}",
                    granularity="element",
                )
            )
        return chunks


class PaperChunker:
    """Composable section/page/element chunk builder with exact-content deduplication."""

    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        strategies: list[ChunkStrategy] | None = None,
    ) -> None:
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategies = strategies or [
            SectionChunkStrategy(),
            PageChunkStrategy(),
            ElementChunkStrategy(),
        ]

    def build(self, documents: list[PaperDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []
        seen: set[tuple[str, str]] = set()
        for document in documents:
            for strategy in self.strategies:
                for chunk in strategy.build(document, self):
                    key = (chunk.paper_id, chunk.content_hash)
                    if key in seen:
                        continue
                    seen.add(key)
                    chunks.append(chunk)
        return chunks

    # Compatibility entry points used by external callers.
    def split(self, documents: list[PaperDocument]) -> list[Chunk]:
        return self._build_with(SectionChunkStrategy(), documents)

    def split_pages(self, documents: list[PaperDocument]) -> list[Chunk]:
        return self._build_with(PageChunkStrategy(), documents)

    def split_elements(self, documents: list[PaperDocument]) -> list[Chunk]:
        return self._build_with(ElementChunkStrategy(), documents)

    def _build_with(
        self,
        strategy: ChunkStrategy,
        documents: list[PaperDocument],
    ) -> list[Chunk]:
        return [chunk for document in documents for chunk in strategy.build(document, self)]

    @staticmethod
    def context(
        paper_title: str,
        section: str,
        page_start: int | None,
        page_end: int | None,
        element_type: str,
        modality: str,
    ) -> str:
        page = ""
        if page_start is not None and page_end is not None:
            page = (
                f"第 {page_start} 页"
                if page_start == page_end
                else f"第 {page_start}-{page_end} 页"
            )
        bits = [f"《{paper_title}》", section]
        if page:
            bits.append(page)
        bits.append(f"{element_type}/{modality}")
        return "，".join(bits)

    @staticmethod
    def element_section(element_type: str, page: int) -> str:
        label = {
            "table": "Table",
            "figure": "Figure",
            "formula": "Formula",
            "text_block": "Text block",
            "image": "Image",
            "page_image": "Page image",
        }.get(element_type, element_type.title())
        return f"{label} on page {page}"

    def split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]
        pieces: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                end = self._soft_boundary(text, start, end)
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            if end >= len(text):
                break
            start = max(end - self.chunk_overlap, start + 1)
        return pieces

    def _soft_boundary(self, text: str, start: int, end: int) -> int:
        window = text[start:end]
        for separator in ("\n\n", "\n", ". ", "。"):
            position = window.rfind(separator)
            if position >= int(self.chunk_size * 0.5):
                return start + position + len(separator)
        return end
