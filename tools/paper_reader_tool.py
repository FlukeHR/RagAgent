from __future__ import annotations

from pathlib import Path

from config.settings import BASE_DIR, Settings
from retrieval.loader import PaperDocument, PaperLoader
from tools.base import ToolResult


class PaperReaderTool:
    """精读本地某篇论文的指定章节。"""

    name = "read_paper_section"

    def __init__(self, settings: Settings, max_chars: int = 4000) -> None:
        self.data_dir = BASE_DIR / settings.project.data_root
        self.loader = PaperLoader(str(self.data_dir))
        self.max_chars = max_chars

    @staticmethod
    def schema() -> dict:
        return {
            "name": "read_paper_section",
            "description": (
                "精读本地某篇论文的指定章节全文。通常先用 search_local_papers 得到 paper_id，"
                "再用本工具深入阅读某章节（如 Method、Experiments）。"
                "省略 section 时返回该论文的章节目录与摘要。"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "论文标识（search_local_papers 返回的 paper_id）",
                    },
                    "section": {
                        "type": "string",
                        "description": "章节名（模糊匹配，如 'method'）；省略则返回章节目录",
                    },
                },
                "required": ["paper_id"],
            },
        }

    def run(
        self, paper_id: str, section: str | None = None, _id_base: int = 0
    ) -> ToolResult:
        doc = self._load(paper_id)
        if doc is None:
            return ToolResult(
                text=f"未找到 paper_id={paper_id} 对应的本地论文。", sources=[]
            )

        sid = f"S{_id_base + 1}"
        if not section:
            toc = "\n".join(f"- {s.title}" for s in doc.sections)
            abstract = next(
                (s.text for s in doc.sections if "abstract" in s.title.lower()),
                doc.sections[0].text,
            )
            text = (
                f"[{sid}]《{doc.title}》章节目录（引用时用 {sid}）:\n{toc}\n\n"
                f"摘要/开头:\n{abstract[: self.max_chars]}"
            )
            return ToolResult(
                text=text,
                sources=[self._src(doc, "TOC", sid, snippet=abstract[:600])],
            )

        matched = [s for s in doc.sections if section.lower() in s.title.lower()]
        if not matched:
            toc = ", ".join(s.title for s in doc.sections)
            return ToolResult(
                text=f"未找到章节 '{section}'。可用章节: {toc}", sources=[]
            )
        sec = matched[0]
        return ToolResult(
            text=f"[{sid}]《{doc.title}》｜章节 {sec.title}（引用时用 {sid}）:\n{sec.text[: self.max_chars]}",
            sources=[self._src(doc, sec.title, sid, snippet=sec.text[:600])],
        )

    def _load(self, paper_id: str) -> PaperDocument | None:
        for suffix in (".pdf", ".txt", ".md"):
            f = self.data_dir / f"{paper_id}{suffix}"
            if f.exists():
                return self.loader.load_file(Path(f))
        return None

    @staticmethod
    def _src(
        doc: PaperDocument, section: str, sid: str, snippet: str | None = None
    ) -> dict:
        return {
            "id": sid,
            "chunk_id": f"{doc.paper_id}::{section}",
            "paper_id": doc.paper_id,
            "paper_title": doc.title,
            "section": section,
            "source": doc.source,
            "page_start": (
                next((s.page_start for s in doc.sections if s.title == section), None)
                if section != "TOC"
                else None
            ),
            "page_end": (
                next((s.page_end for s in doc.sections if s.title == section), None)
                if section != "TOC"
                else None
            ),
            "element_type": "text",
            "modality": "text",
            "bbox": None,
            "chunk_context": (
                f"《{doc.title}》，{section}"
                if section != "TOC"
                else f"《{doc.title}》，章节目录"
            ),
            "heading_path": section,
            "score": None,
            "snippet": snippet,
        }
