from __future__ import annotations

from pydantic import BaseModel, Field


class Turn(BaseModel):
    """对话历史中的一轮（仅纯文本，不含工具内部块）。"""

    role: str = Field(..., description="user | assistant")
    content: str = Field(..., description="该轮文本内容")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    collection: str | None = Field(None, description="目标论文集合名称")
    history: list[Turn] = Field(
        default_factory=list, description="本会话之前的对话轮次，用于历史注入与指代消解"
    )


class SourceItem(BaseModel):
    id: str | None = None  # 引用编号，如 S1；前端按它把 [S1] 标记映射到本来源
    chunk_id: str | None = None
    paper_id: str
    paper_title: str
    section: str
    source: str
    page_start: int | None = None
    page_end: int | None = None
    element_type: str | None = None
    modality: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    chunk_context: str | None = None
    heading_path: str | None = None
    score: float | None = None
    snippet: str | None = None  # 引用原文片段，前端预览时用于在 PDF 中定位高亮
    collection: str | None = None  # 该来源 PDF 所在集合目录（arXiv 下载的指向 arxiv），供前端预览定位
    image_mime_type: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    image_base64: str | None = None


class AskResponse(BaseModel):
    collection: str
    answer: str
    steps: list[str]
    sources: list[SourceItem]
    trace: list[dict] = []  # 结构化可观测事件（LLM 轮次/工具调用/核查/预算）


class CollectionsResponse(BaseModel):
    collections: list[str]


class IngestArxivRequest(BaseModel):
    query: str = Field(..., min_length=1, description="arXiv 检索关键词")
    collection: str | None = Field(None, description="下载入库的集合名，默认 arxiv")
    max_results: int | None = Field(None, description="下载论文数量")


class IngestArxivResponse(BaseModel):
    collection: str
    downloaded: list[str]
    indexed_chunks: int


class TitleRequest(BaseModel):
    messages: list[Turn] = Field(..., description="用于概括标题的对话轮次（通常是首轮问答）")


class TitleResponse(BaseModel):
    title: str
