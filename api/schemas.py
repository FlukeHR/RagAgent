from __future__ import annotations

from pydantic import BaseModel, Field


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
    name: str | None = Field(None, min_length=1, max_length=80)
    provider: str | None = Field(None, min_length=1, max_length=40)
    api_base: str | None = Field(None, min_length=1, max_length=500)
    model_name: str | None = Field(None, min_length=1, max_length=160)
    api_key: str | None = Field(None, min_length=1, max_length=1000)
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


class DocumentAskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    model_profile_id: str | None = None


class EvidenceSource(BaseModel):
    id: str
    claim: str
    page: int
    bbox: tuple[float, float, float, float]


class DocumentAskResponse(BaseModel):
    answer: str
    status: str
    sources: list[EvidenceSource]
