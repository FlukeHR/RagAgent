from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    repo_name: str | None = Field(None, description="目标仓库名称")


class SourceItem(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    score: float


class AskResponse(BaseModel):
    repo_name: str
    answer: str
    steps: list[str]
    sources: list[SourceItem]


class ReposResponse(BaseModel):
    repositories: list[str]
