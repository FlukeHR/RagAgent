from __future__ import annotations

from config.settings import BASE_DIR, Settings
from retrieval.retriever import Retriever
from tools.base import EvidenceSource, ToolPolicy, ToolResult


class PaperSearchTool:
    """本地论文库语义检索工具。"""

    name = "search_local_papers"
    policy = ToolPolicy(side_effects="read", idempotent=True)

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        index_dir = BASE_DIR / settings.index.index_root
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
                "additionalProperties": False,
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1000,
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
        sources: list[EvidenceSource] = []
        for i, r in enumerate(results, start=1):
            c = r.chunk
            cite = f"{{{{cite:{i - 1}}}}}"
            page_label = (
                f"｜页码 {c.page_start}" if c.page_start == c.page_end and c.page_start is not None
                else f"｜页码 {c.page_start}-{c.page_end}" if c.page_start is not None and c.page_end is not None
                else ""
            )
            blocks.append(
                f"{cite}《{c.paper_title}》｜章节 {c.section}{page_label}｜类型 {c.element_type}｜模态 {c.modality}｜论文ID {c.paper_id}"
                f"\n上下文：{c.chunk_context or c.section}"
                f"（论文ID 仅供 read_paper_section 调用）\n{c.content}"
            )
            sources.append(
                EvidenceSource.from_chunk(
                    c,
                    score=r.score,
                    snippet_chars=self.settings.harness.source_snippet_chars,
                    confidence=r.confidence,
                    score_backend=r.backend,
                    dense_score=r.dense_score,
                    sparse_score=r.sparse_score,
                    fusion_score=r.fusion_score,
                    lexical_anchor_score=r.lexical_anchor_score,
                )
            )
        return ToolResult(
            text="\n\n".join(blocks),
            sources=sources,
            metadata={"retrieval": self.retriever.last_trace},
        )
