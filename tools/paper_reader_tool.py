from __future__ import annotations

from config.settings import BASE_DIR, Settings
from retrieval.loader import PaperDocument, PaperLoader
from retrieval.repository import PaperRepository
from tools.base import EvidenceSource, ToolPolicy, ToolResult


class PaperReaderTool:
    """Read one bounded section from a paper in the local repository."""

    name = "read_paper_section"
    policy = ToolPolicy(side_effects="read", idempotent=True)

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.data_dir = BASE_DIR / settings.project.data_root
        self.repository = PaperRepository(self.data_dir)
        self.loader = PaperLoader(str(self.data_dir))

    @staticmethod
    def schema() -> dict:
        return {
            "name": "read_paper_section",
            "description": (
                "精读本地某篇论文的指定章节。通常先用 search_local_papers 得到 paper_id；"
                "省略 section 时只返回章节目录与摘要。"
            ),
            "input_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 160,
                        "description": "search_local_papers 返回的论文标识",
                    },
                    "section": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 200,
                        "description": "章节名，模糊匹配；省略则返回目录",
                    },
                },
                "required": ["paper_id"],
            },
        }

    def run(self, paper_id: str, section: str | None = None) -> ToolResult:
        doc = self._load(paper_id)
        if doc is None:
            return ToolResult(text=f"未找到 paper_id={paper_id} 对应的本地论文。")

        max_chars = self.settings.harness.tool_result_max_chars
        if not section:
            toc = "\n".join(f"- {item.title}" for item in doc.sections)
            abstract = next(
                (item.text for item in doc.sections if "abstract" in item.title.lower()),
                doc.sections[0].text if doc.sections else "",
            )
            return ToolResult(
                text=(
                    "{{cite:0}}《"
                    f"{doc.title}》章节目录：\n{toc}\n\n摘要/开头：\n{abstract[:max_chars]}"
                ),
                sources=[self._source(doc, "TOC", abstract)],
            )

        matched = [item for item in doc.sections if section.lower() in item.title.lower()]
        if not matched:
            toc = ", ".join(item.title for item in doc.sections)
            return ToolResult(text=f"未找到章节 '{section}'。可用章节：{toc}")
        selected = matched[0]
        return ToolResult(
            text=f"{{{{cite:0}}}}《{doc.title}》｜章节 {selected.title}\n"
            f"{selected.text[:max_chars]}",
            sources=[self._source(doc, selected.title, selected.text)],
        )

    def _load(self, paper_id: str) -> PaperDocument | None:
        path = self.repository.resolve(paper_id)
        return self.loader.load_file(path) if path is not None else None

    def _source(self, doc: PaperDocument, section: str, snippet: str) -> EvidenceSource:
        selected = next((item for item in doc.sections if item.title == section), None)
        return EvidenceSource(
            chunk_id=f"{doc.paper_id}::{section}",
            paper_id=doc.paper_id,
            paper_title=doc.title,
            section=section,
            source=doc.source,
            page_start=selected.page_start if selected else None,
            page_end=selected.page_end if selected else None,
            element_type="text",
            modality="text",
            chunk_context=f"《{doc.title}》，{section}",
            heading_path=section,
            snippet=snippet[: self.settings.harness.source_snippet_chars],
            quality_rank=2,
        )
