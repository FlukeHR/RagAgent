from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlparse

from fastapi import Depends, HTTPException, Request, Response, status

from config.settings import Settings, load_settings
from services.app_store import AppStore, AuthSessionRecord, UserRecord
from services.security import AuthService, SecretBox, secure_compare
from services.user_ingest import UserIngestManager
from services.user_scope import AgentPool


@dataclass(frozen=True)
class AuthContext:
    """Authenticated user and server-side session bound to one request."""

    user: UserRecord
    session: AuthSessionRecord
    raw_token: str


@lru_cache(maxsize=1)
def settings() -> Settings:
    return load_settings()


@lru_cache(maxsize=1)
def store() -> AppStore:
    return AppStore(settings())


@lru_cache(maxsize=1)
def auth_service() -> AuthService:
    return AuthService(settings(), store())


@lru_cache(maxsize=1)
def secret_box() -> SecretBox:
    return SecretBox(settings())


@lru_cache(maxsize=1)
def agent_pool() -> AgentPool:
    return AgentPool(settings(), secret_box())


@lru_cache(maxsize=1)
def ingest_manager() -> UserIngestManager:
    return UserIngestManager(settings(), store(), agent_pool())


def current_auth(request: Request) -> AuthContext:
    """Require a valid local HttpOnly session cookie."""

    raw_token = request.cookies.get(settings().app.session_cookie_name)
    authenticated = auth_service().authenticate(raw_token)
    if authenticated is None or raw_token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录")
    user, session = authenticated
    return AuthContext(user=user, session=session, raw_token=raw_token)


def require_csrf(
    request: Request,
    context: AuthContext = Depends(current_auth),
) -> AuthContext:
    """Require a same-origin request and a session-bound CSRF token."""

    origin = request.headers.get("origin")
    if not origin:
        raise HTTPException(status_code=403, detail="请求来源无效")
    parsed = urlparse(origin)
    host = request.headers.get("host", "")
    if parsed.netloc != host or parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=403, detail="请求来源无效")
    token = request.headers.get("x-csrf-token", "")
    if not token or not secure_compare(token, context.session.csrf_token):
        raise HTTPException(status_code=403, detail="CSRF 校验失败")
    return context


def set_session_cookie(response: Response, raw_token: str) -> None:
    """Set the local-only session cookie with strict browser defaults."""

    config = settings().app
    response.set_cookie(
        config.session_cookie_name,
        raw_token,
        max_age=config.session_ttl_seconds,
        httponly=True,
        secure=False,
        samesite="strict",
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(settings().app.session_cookie_name, path="/")
