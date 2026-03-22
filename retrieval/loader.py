from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class CodeDocument:
    path: str
    content: str


class CodeLoader:
    def __init__(self, root_dir: str, include_suffixes: tuple[str, ...] = (".py",)) -> None:
        self.root_dir = Path(root_dir)
        self.include_suffixes = include_suffixes

    def iter_files(self) -> Iterable[Path]:
        for file in self.root_dir.rglob("*"):
            if file.is_file() and file.suffix in self.include_suffixes:
                yield file

    def load(self) -> list[CodeDocument]:
        documents: list[CodeDocument] = []
        for file in self.iter_files():
            try:
                content = file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            documents.append(CodeDocument(path=str(file), content=content))
        return documents
