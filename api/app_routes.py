from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import AuthContext, current_auth, require_csrf, secret_box, settings, store
from api.schemas import ModelProfileCreate, ModelProfileResponse, ModelProfileUpdate
from services.security import validate_model_endpoint


router = APIRouter(prefix="/api", tags=["application"])


def _profile_response(profile: dict[str, Any]) -> ModelProfileResponse:
    return ModelProfileResponse(
        profile_id=str(profile["profile_id"]),
        name=str(profile["name"]),
        provider=str(profile["provider"]),
        api_base=str(profile["api_base"]),
        model_name=str(profile["model_name"]),
        key_last4=str(profile["key_last4"]),
        is_default=bool(profile["is_default"]),
        created_at=float(profile["created_at"]),
        updated_at=float(profile["updated_at"]),
    )


@router.get("/dashboard")
def dashboard(context: AuthContext = Depends(current_auth)) -> dict[str, int]:
    return {
        "model_profiles": len(store().list_model_profiles(context.user.user_id)),
        "documents": len(store().list_documents(context.user.user_id)),
    }


@router.get("/model-profiles", response_model=list[ModelProfileResponse])
def list_model_profiles(
    context: AuthContext = Depends(current_auth),
) -> list[ModelProfileResponse]:
    return [
        _profile_response(profile)
        for profile in store().list_model_profiles(context.user.user_id)
    ]


@router.post("/model-profiles", response_model=ModelProfileResponse, status_code=201)
def create_model_profile(
    payload: ModelProfileCreate,
    context: AuthContext = Depends(require_csrf),
) -> ModelProfileResponse:
    profiles = store().list_model_profiles(context.user.user_id)
    if len(profiles) >= settings().app.max_model_profiles_per_user:
        raise HTTPException(status_code=409, detail="模型配置数量已达上限")
    try:
        api_base = validate_model_endpoint(payload.api_base, settings())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    profile_id = uuid.uuid4().hex
    nonce, ciphertext = secret_box().encrypt(
        context.user.user_id, profile_id, payload.api_key
    )
    try:
        profile = store().create_model_profile(
            context.user.user_id,
            {
                "profile_id": profile_id,
                "name": payload.name.strip(),
                "provider": payload.provider.strip(),
                "api_base": api_base,
                "model_name": payload.model_name.strip(),
                "key_nonce": nonce,
                "key_ciphertext": ciphertext,
                "key_last4": payload.api_key[-4:],
                "is_default": payload.is_default or not profiles,
            },
        )
    except Exception as exc:
        if "UNIQUE constraint failed" in str(exc):
            raise HTTPException(status_code=409, detail="模型配置名称已存在") from exc
        raise
    return _profile_response(profile)


@router.patch("/model-profiles/{profile_id}", response_model=ModelProfileResponse)
def update_model_profile(
    profile_id: str,
    payload: ModelProfileUpdate,
    context: AuthContext = Depends(require_csrf),
) -> ModelProfileResponse:
    current = store().get_model_profile(context.user.user_id, profile_id)
    if current is None:
        raise HTTPException(status_code=404, detail="模型配置不存在")
    values = payload.model_dump(exclude_unset=True, exclude={"api_key"})
    try:
        values["api_base"] = validate_model_endpoint(
            str(values.get("api_base") or current["api_base"]), settings()
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if payload.api_key is not None:
        nonce, ciphertext = secret_box().encrypt(
            context.user.user_id, profile_id, payload.api_key
        )
        values.update(
            {
                "key_nonce": nonce,
                "key_ciphertext": ciphertext,
                "key_last4": payload.api_key[-4:],
                "key_version": int(current["key_version"]) + 1,
            }
        )
    try:
        profile = store().update_model_profile(context.user.user_id, profile_id, values)
    except Exception as exc:
        if "UNIQUE constraint failed" in str(exc):
            raise HTTPException(status_code=409, detail="模型配置名称已存在") from exc
        raise
    assert profile is not None
    return _profile_response(profile)


@router.delete("/model-profiles/{profile_id}", status_code=204)
def delete_model_profile(
    profile_id: str,
    context: AuthContext = Depends(require_csrf),
) -> None:
    if not store().delete_model_profile(context.user.user_id, profile_id):
        raise HTTPException(status_code=404, detail="模型配置不存在")
