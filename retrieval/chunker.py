from __future__ import annotations

from dataclasses import dataclass

from retrieval.loader import CodeDocument


@dataclass
class CodeChunk:
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    content: str


class CodeChunker:
    def __init__(self, chunk_size: int = 120, chunk_overlap: int = 20) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, docs: list[CodeDocument]) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []
        for doc in docs:
            lines = doc.content.splitlines()
            if not lines:
                continue

            step = self.chunk_size - self.chunk_overlap
            chunk_idx = 0
            for start in range(0, len(lines), step):
                end = min(start + self.chunk_size, len(lines))
                chunk_lines = lines[start:end]
                if not chunk_lines:
                    continue
                chunk = CodeChunk(
                    chunk_id=f"{doc.path}::{chunk_idx}",
                    file_path=doc.path,
                    start_line=start + 1,
                    end_line=end,
                    content="\n".join(chunk_lines),
                )
                chunks.append(chunk)
                chunk_idx += 1
                if end >= len(lines):
                    break
        return chunks
