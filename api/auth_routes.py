from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from api.dependencies import (
    AuthContext,
    auth_service,
    clear_session_cookie,
    current_auth,
    require_csrf,
    set_session_cookie,
    store,
)
from api.schemas import (
    AccountUpdateRequest,
    AuthResponse,
    AuthUser,
    ChangePasswordRequest,
    LoginRequest,
    RegisterRequest,
)


router = APIRouter(prefix="/api", tags=["authentication"])


def _response(context: AuthContext) -> AuthResponse:
    return AuthResponse(
        user=AuthUser(
            user_id=context.user.user_id,
            username=context.user.username,
            display_name=context.user.display_name,
        ),
        csrf_token=context.session.csrf_token,
        expires_at=context.session.expires_at,
    )


@router.post("/auth/register", response_model=AuthResponse)
def register(payload: RegisterRequest, request: Request, response: Response) -> AuthResponse:
    """Create a local username/password account and sign it in."""

    try:
        result = auth_service().register(
            payload.username,
            payload.password,
            payload.display_name,
            request.headers.get("user-agent", ""),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    set_session_cookie(response, result.token)
    return _response(
        AuthContext(user=result.user, session=result.session, raw_token=result.token)
    )


@router.post("/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest, request: Request, response: Response) -> AuthResponse:
    """Authenticate an existing local account."""

    try:
        result = auth_service().login(
            payload.username,
            payload.password,
            request.headers.get("user-agent", ""),
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    set_session_cookie(response, result.token)
    return _response(
        AuthContext(user=result.user, session=result.session, raw_token=result.token)
    )


@router.get("/auth/session", response_model=AuthResponse)
def session(context: AuthContext = Depends(current_auth)) -> AuthResponse:
    """Return the authenticated account and its CSRF token."""

    return _response(context)


@router.post("/auth/logout", status_code=204)
def logout(
    response: Response,
    context: AuthContext = Depends(require_csrf),
) -> None:
    store().delete_auth_session(auth_service().token_hash(context.raw_token))
    clear_session_cookie(response)


@router.patch("/account", response_model=AuthResponse)
def update_account(
    payload: AccountUpdateRequest,
    context: AuthContext = Depends(require_csrf),
) -> AuthResponse:
    store().update_display_name(context.user.user_id, payload.display_name.strip())
    user = store().get_user(context.user.user_id)
    assert user is not None
    return _response(AuthContext(user=user, session=context.session, raw_token=context.raw_token))


@router.post("/account/change-password", status_code=204)
def change_password(
    payload: ChangePasswordRequest,
    context: AuthContext = Depends(require_csrf),
) -> None:
    try:
        auth_service().change_password(
            context.user,
            payload.current_password,
            payload.new_password,
            auth_service().token_hash(context.raw_token),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/account/sessions")
def account_sessions(
    context: AuthContext = Depends(current_auth),
) -> list[dict[str, object]]:
    """List this account's active local browser sessions."""

    active = store().list_auth_sessions(context.user.user_id)
    for item in active:
        item["current"] = item["session_id"] == context.session.session_id
    return active


@router.post("/account/logout-all", status_code=204)
def logout_all(
    response: Response,
    context: AuthContext = Depends(require_csrf),
) -> None:
    store().delete_all_auth_sessions(context.user.user_id)
    clear_session_cookie(response)
