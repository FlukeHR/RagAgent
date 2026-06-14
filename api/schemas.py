from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    collection: str | None = Field(None, description="目标论文集合名称")


class SourceItem(BaseModel):
    id: str | None = None  # 引用编号，如 S1；前端按它把 [S1] 标记映射到本来源
    chunk_id: str | None = None
    paper_id: str
    paper_title: str
    section: str
    source: str
    score: float | None = None
    snippet: str | None = None  # 引用原文片段，前端预览时用于在 PDF 中定位高亮


class AskResponse(BaseModel):
    collection: str
    answer: str
    steps: list[str]
    sources: list[SourceItem]


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
