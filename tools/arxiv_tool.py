from __future__ import annotations

from config.settings import Settings
from services.arxiv_service import ArxivSearchService
from tools.base import EvidenceSource, ToolPolicy, ToolResult


class ArxivTool:
    """Search arXiv abstracts without downloading full text."""

    name = "search_arxiv"
    policy = ToolPolicy(side_effects="network", idempotent=True)

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.service = ArxivSearchService(settings)

    def schema(self) -> dict:
        return {
            "name": "search_arxiv",
            "description": (
                "在线检索 arXiv 摘要；本地证据不足或用户要求最新研究时使用。"
                "需要全文时再调用 ingest_arxiv_papers。"
            ),
            "input_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1000,
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": self.settings.arxiv.max_results,
                    },
                },
                "required": ["query"],
            },
        }

    def run(self, query: str, max_results: int | None = None) -> ToolResult:
        papers = self.service.search(query, max_results)
        if not papers:
            return ToolResult(text=f"arXiv 未检索到与 '{query}' 相关的论文。")
        blocks: list[str] = []
        sources: list[EvidenceSource] = []
        for index, paper in enumerate(papers):
            cite = f"{{{{cite:{index}}}}}"
            authors = ", ".join(paper.authors[:5])
            published = paper.published.date().isoformat() if paper.published else "unknown"
            blocks.append(
                f"{cite} {paper.title}\n作者: {authors}\n发表: {published}\n"
                f"arxiv_id: {paper.arxiv_id}  链接: {paper.entry_url}\n摘要: {paper.summary}"
            )
            sources.append(
                EvidenceSource(
                    chunk_id=paper.arxiv_id,
                    paper_id=paper.arxiv_id,
                    paper_title=paper.title,
                    section="Abstract",
                    source=paper.entry_url,
                    element_type="abstract",
                    modality="text",
                    chunk_context=f"arXiv abstract for {paper.title}",
                    heading_path="Abstract",
                    snippet=paper.summary[: self.settings.harness.source_snippet_chars],
                    published_at=published,
                    quality_rank=1,
                )
            )
        return ToolResult(text="\n\n".join(blocks), sources=sources)
