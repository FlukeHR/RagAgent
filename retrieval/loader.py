from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class PaperSection:
    title: str
    text: str


@dataclass
class PaperDocument:
    paper_id: str          # 论文标识，默认取文件名（不含扩展名）
    title: str             # 论文标题
    source: str            # 原始文件路径
    sections: list[PaperSection]


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

    def __init__(
        self, root_dir: str, include_suffixes: tuple[str, ...] = (".pdf", ".txt", ".md")
    ) -> None:
        self.root_dir = Path(root_dir)
        self.include_suffixes = include_suffixes

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
        text = self._read(file)
        if not text or not text.strip():
            return None
        return PaperDocument(
            paper_id=file.stem,
            title=self._extract_title(text, fallback=file.stem),
            source=str(file),
            sections=self._split_sections(text),
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
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise RuntimeError(
                "解析 PDF 需要 PyMuPDF，请先安装：pip install pymupdf"
            ) from exc
        parts: list[str] = []
        with fitz.open(str(file)) as doc:
            for page in doc:
                parts.append(page.get_text("text"))
        return "\n".join(parts)

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

    @staticmethod
    def _is_heading(line: str) -> bool:
        s = line.strip()
        if not s or len(s) > 80:
            return False
        if _NUMBERED_HEADING.match(s):
            return True
        low = s.lower().rstrip(":.")
        return low in _SECTION_KEYWORDS
