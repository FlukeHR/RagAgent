from __future__ import annotations

from config.settings import BASE_DIR, Settings
from retrieval.retriever import Retriever
from tools.base import ToolResult


class PaperSearchTool:
    """本地论文库语义检索工具。"""

    name = "search_local_papers"

    def __init__(self, settings: Settings, collection: str) -> None:
        index_dir = BASE_DIR / settings.index.index_root / collection
        self.retriever = Retriever(settings=settings, index_dir=str(index_dir))

    @staticmethod
    def schema() -> dict:
        return {
            "name": "search_local_papers",
            "description": (
                "在本地论文库中做语义检索，返回最相关的论文片段（含论文标题与所属章节）。"
                "当用户的问题可能由已收录论文回答时，优先使用本工具。"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询，可用关键词或自然语言问题",
                    }
                },
                "required": ["query"],
            },
        }

    def run(self, query: str) -> ToolResult:
        results = self.retriever.search(query)
        if not results:
            return ToolResult(text="本地论文库未检索到相关片段。", sources=[])

        blocks: list[str] = []
        sources: list[dict] = []
        for i, r in enumerate(results, start=1):
            c = r.chunk
            blocks.append(
                f"[来源{i}] 《{c.paper_title}》· {c.section} "
                f"(paper_id={c.paper_id}, score={r.score:.3f})\n{c.content}"
            )
            sources.append(
                {
                    "paper_id": c.paper_id,
                    "paper_title": c.paper_title,
                    "section": c.section,
                    "source": c.source,
                    "score": round(float(r.score), 4),
                }
            )
        return ToolResult(text="\n\n".join(blocks), sources=sources)
