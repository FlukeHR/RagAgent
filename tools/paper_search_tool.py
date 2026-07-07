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

    def run(self, query: str, _id_base: int = 0) -> ToolResult:
        results = self.retriever.search(query)
        if not results:
            return ToolResult(text="本地论文库未检索到相关片段。", sources=[])

        blocks: list[str] = []
        sources: list[dict] = []
        for i, r in enumerate(results, start=1):
            c = r.chunk
            sid = f"S{_id_base + i}"  # 全局唯一引用编号，模型据此引用
            page_label = (
                f"｜页码 {c.page_start}" if c.page_start == c.page_end and c.page_start is not None
                else f"｜页码 {c.page_start}-{c.page_end}" if c.page_start is not None and c.page_end is not None
                else ""
            )
            blocks.append(
                f"[{sid}]《{c.paper_title}》｜章节 {c.section}{page_label}｜类型 {c.element_type}｜模态 {c.modality}｜论文ID {c.paper_id}"
                f"\n上下文：{c.chunk_context or c.section}"
                f"（论文ID 仅供 read_paper_section 调用，引用时请用 {sid}）\n{c.content}"
            )
            sources.append(
                {
                    "id": sid,
                    "chunk_id": c.chunk_id,
                    "paper_id": c.paper_id,
                    "paper_title": c.paper_title,
                    "section": c.section,
                    "source": c.source,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "element_type": c.element_type,
                    "modality": c.modality,
                    "bbox": c.bbox,
                    "chunk_context": c.chunk_context,
                    "heading_path": c.heading_path,
                    "score": round(float(r.score), 4),
                    "snippet": c.content[:600],  # 供 /preview 在 PDF 中精确定位高亮
                }
            )
        return ToolResult(text="\n\n".join(blocks), sources=sources)
