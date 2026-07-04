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
                        )
                    )
                    idx += 1
        return chunks

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
