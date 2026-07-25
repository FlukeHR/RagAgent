from __future__ import annotations

from config.settings import Settings
from tools.base import ToolResult


class ArxivTool:
    """arXiv 在线摘要检索工具，不负责下载或入库。"""

    name = "search_arxiv"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.max_results = settings.arxiv.max_results

    @staticmethod
    def schema() -> dict:
        return {
            "name": "search_arxiv",
            "description": (
                "在线检索 arXiv 最新论文，返回标题、作者、发表日期、摘要与链接。"
                "当本地论文库不足以回答、或用户询问最新研究进展时使用。"
                "本工具只侦察摘要；需要全文时再调用 ingest_arxiv_papers。"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "arXiv 检索关键词"},
                    "max_results": {
                        "type": "integer",
                        "description": "返回论文数量，省略则用默认配置",
                    },
                },
                "required": ["query"],
            },
        }

    def run(
        self,
        query: str,
        max_results: int | None = None,
        _id_base: int = 0,
    ) -> ToolResult:
        import arxiv

        n = max_results or self.max_results
        search = arxiv.Search(
            query=query, max_results=n, sort_by=arxiv.SortCriterion.Relevance
        )
        client = arxiv.Client()

        blocks: list[str] = []
        sources: list[dict] = []
        for i, result in enumerate(client.results(search), start=1):
            aid = result.get_short_id()
            sid = f"S{_id_base + i}"
            authors = ", ".join(a.name for a in result.authors[:5])
            summary = " ".join(result.summary.split())
            block = (
                f"[{sid}] {result.title}\n"
                f"作者: {authors}\n"
                f"发表: {result.published.date()}\n"
                f"arxiv_id: {aid}  链接: {result.entry_id}\n"
                f"摘要: {summary}"
            )
            blocks.append(block)
            src = {
                "id": sid,
                "chunk_id": aid,
                "paper_id": aid,
                "paper_title": result.title,
                "section": "Abstract",
                "source": result.entry_id,
                "element_type": "abstract",
                "modality": "text",
                "bbox": None,
                "chunk_context": f"arXiv abstract for {result.title}",
                "heading_path": "Abstract",
                "score": None,
                "snippet": summary[:600],
            }
            sources.append(src)

        if not blocks:
            return ToolResult(text=f"arXiv 未检索到与 '{query}' 相关的论文。", sources=[])
        return ToolResult(text="\n\n".join(blocks), sources=sources)
