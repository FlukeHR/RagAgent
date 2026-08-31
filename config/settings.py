from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppConfig:
    """Local account, credential, and endpoint settings."""

    database_path: str = "./data/app.sqlite3"
    secrets_root: str = "./data/secrets"
    session_cookie_name: str = "paper_rag_session"
    session_ttl_seconds: int = 604800
    password_min_length: int = 10
    max_login_failures: int = 5
    login_lock_seconds: int = 900
    max_model_profiles_per_user: int = 10
    documents_root: str = "./data/documents"
    max_pdf_mb: int = 20
    max_pdf_pages: int = 30
    candidate_pages: int = 3
    model_timeout_seconds: float = 120.0
    model_max_tokens: int = 1200
    enforce_public_dns_for_model_endpoints: bool = False
    allowed_local_llm_endpoints: list[str] = field(default_factory=list)


@dataclass
class Settings:
    app: AppConfig


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "config.yaml"


def load_settings(path: str | Path | None = None) -> Settings:
    """Load and validate the small remaining application configuration."""

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    settings = Settings(app=AppConfig(**raw.get("app", {})))
    if settings.app.session_ttl_seconds <= 0:
        raise ValueError("app.session_ttl_seconds must be positive")
    if settings.app.password_min_length < 8:
        raise ValueError("app.password_min_length must be at least 8")
    if settings.app.max_login_failures <= 0 or settings.app.login_lock_seconds <= 0:
        raise ValueError("login limits must be positive")
    if settings.app.max_model_profiles_per_user <= 0:
        raise ValueError("app.max_model_profiles_per_user must be positive")
    if settings.app.max_pdf_mb <= 0 or settings.app.max_pdf_pages <= 0:
        raise ValueError("PDF limits must be positive")
    if (
        settings.app.candidate_pages <= 0
        or settings.app.model_timeout_seconds <= 0
        or settings.app.model_max_tokens <= 0
    ):
        raise ValueError("candidate page and model timeout limits must be positive")
    return settings
