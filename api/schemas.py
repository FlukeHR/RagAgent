from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Turn(BaseModel):
    """One plain-text conversation turn."""

    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = Field(
        None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    history: list[Turn] = Field(default_factory=list)


class SourceItem(BaseModel):
    id: str | None = None
    chunk_id: str | None = None
    paper_id: str
    paper_title: str
    section: str
    source: str | None = Field(default=None, exclude=True)
    source_kind: Literal["library_pdf", "external_url"] = "library_pdf"
    citation_url: str | None = None
    preview_kind: Literal["pdf", "web"] = "pdf"
    page_start: int | None = None
    page_end: int | None = None
    element_type: str | None = None
    modality: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    bbox_space: str | None = None
    element_id: str | None = None
    parser_metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_context: str | None = None
    heading_path: str | None = None
    score: float | None = None
    confidence: float | None = None
    score_backend: str | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    lexical_anchor_score: float = 0.0
    snippet: str | None = None
    published_at: str | None = None
    support_status: str | None = None
    quality_rank: int = 0
    origin_tools: list[str] = Field(default_factory=list)


class AskResponse(BaseModel):
    answer: str
    status: str = "answered"
    steps: list[str]
    sources: list[SourceItem]
    trace: list[dict[str, Any]] = Field(default_factory=list)
    suggested_actions: list[dict[str, Any]] = Field(default_factory=list)


class ArxivProposalRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    max_results: int | None = Field(None, ge=1, le=20)


class ArxivCandidate(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    summary: str
    published: str | None = None
    entry_url: str | None = None


class ArxivProposalResponse(BaseModel):
    proposal_id: str
    expires_at: float
    candidates: list[ArxivCandidate]


class ConfirmIngestRequest(BaseModel):
    proposal_id: str = Field(..., min_length=32, max_length=64, pattern=r"^[a-f0-9]+$")
    arxiv_ids: list[str] = Field(..., min_length=1, max_length=1)


class IngestJobResponse(BaseModel):
    job_id: str
    proposal_id: str
    query: str
    arxiv_ids: list[str]
    status: Literal["queued", "parsing", "indexing", "succeeded", "failed"]
    created_at: float
    updated_at: float
    error: str | None = None
    result: dict[str, Any] | None = None


class TitleRequest(BaseModel):
    messages: list[Turn]


class TitleResponse(BaseModel):
    title: str


class ConversationResponse(BaseModel):
    session_id: str
    state: dict[str, Any]
    history: list[Turn]


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=8, max_length=256)
    display_name: str = Field(default="", max_length=80)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=32)
    password: str = Field(..., min_length=1, max_length=256)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=256)
    new_password: str = Field(..., min_length=8, max_length=256)


class AccountUpdateRequest(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=80)


class AuthUser(BaseModel):
    user_id: str
    username: str
    display_name: str


class AuthResponse(BaseModel):
    user: AuthUser
    csrf_token: str
    expires_at: float


class ModelProfileCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    provider: str = Field(default="openai-compatible", min_length=1, max_length=40)
    api_base: str = Field(..., min_length=1, max_length=500)
    model_name: str = Field(..., min_length=1, max_length=160)
    api_key: str = Field(..., min_length=1, max_length=1000)
    is_default: bool = False


class ModelProfileUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    provider: str | None = Field(default=None, min_length=1, max_length=40)
    api_base: str | None = Field(default=None, min_length=1, max_length=500)
    model_name: str | None = Field(default=None, min_length=1, max_length=160)
    api_key: str | None = Field(default=None, min_length=1, max_length=1000)
    is_default: bool | None = None


class ModelProfileResponse(BaseModel):
    profile_id: str
    name: str
    provider: str
    api_base: str
    model_name: str
    key_last4: str
    is_default: bool
    created_at: float
    updated_at: float


class SessionCreateRequest(BaseModel):
    title: str = Field(default="新对话", max_length=80)
    model_profile_id: str | None = Field(default=None, max_length=64)


class SessionUpdateRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=80)
    model_profile_id: str | None = Field(default=None, max_length=64)


class SessionAskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=20000)


class SessionImportItem(BaseModel):
    title: str = Field(default="旧对话", max_length=80)
    messages: list[Turn] = Field(default_factory=list, max_length=100)


class SessionImportRequest(BaseModel):
    sessions: list[SessionImportItem] = Field(default_factory=list, max_length=50)


class ArxivConfirmRequest(BaseModel):
    proposal_id: str = Field(..., min_length=32, max_length=64)
    arxiv_id: str = Field(..., min_length=1, max_length=80)
