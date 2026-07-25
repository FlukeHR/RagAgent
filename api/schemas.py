from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal


class Turn(BaseModel):
    """对话历史中的一轮（仅纯文本，不含工具内部块）。"""

    role: Literal["user", "assistant"]
    content: str = Field(..., description="该轮文本内容")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    session_id: str | None = Field(
        None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
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
    confidence: float | None = None
    score_backend: str | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    snippet: str | None = None  # 引用原文片段，供回答展示与生成侧评估使用
    image_mime_type: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    image_base64: str | None = None
    published_at: str | None = None
    support_status: str | None = None
    quality_rank: int = 0


class AskResponse(BaseModel):
    answer: str
    status: str = "answered"
    steps: list[str]
    sources: list[SourceItem]
    trace: list[dict] = Field(default_factory=list)


class IngestArxivRequest(BaseModel):
    query: str = Field(..., min_length=1, description="arXiv 检索关键词")
    max_results: int | None = Field(None, ge=1, le=20, description="下载论文数量")


class IngestArxivResponse(BaseModel):
    downloaded: list[str]
    indexed_chunks: int


class TitleRequest(BaseModel):
    messages: list[Turn] = Field(..., description="用于概括标题的对话轮次（通常是首轮问答）")


class TitleResponse(BaseModel):
    title: str


class ConversationResponse(BaseModel):
    session_id: str
    state: dict
    history: list[Turn]
