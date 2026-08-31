from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.dependencies import (
    AuthContext,
    current_auth,
    document_service,
    evidence_agent,
    require_csrf,
    secret_box,
    settings,
    store,
)
from api.schemas import DocumentAskRequest, DocumentAskResponse


router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("")
def list_documents(context: AuthContext = Depends(current_auth)) -> list[dict[str, object]]:
    return document_service().store.list_documents(context.user.user_id)


@router.post("", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    context: AuthContext = Depends(require_csrf),
) -> dict[str, object]:
    limit = settings().app.max_pdf_mb * 1024 * 1024
    content = await file.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(status_code=413, detail="PDF 文件过大")
    try:
        return document_service().ingest(
            context.user.user_id, file.filename or "document.pdf", content
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{document_id}/pages/{page_number}", response_class=FileResponse)
def document_page(
    document_id: str,
    page_number: int,
    context: AuthContext = Depends(current_auth),
) -> FileResponse:
    path = document_service().page_image(
        context.user.user_id, document_id, page_number
    )
    if path is None:
        raise HTTPException(status_code=404, detail="页面不存在")
    return FileResponse(path, media_type="image/png")


@router.post("/{document_id}/ask", response_model=DocumentAskResponse)
def ask_document(
    document_id: str,
    payload: DocumentAskRequest,
    context: AuthContext = Depends(require_csrf),
) -> DocumentAskResponse:
    document = store().get_document(context.user.user_id, document_id)
    if document is None or document["status"] != "ready":
        raise HTTPException(status_code=404, detail="文档不存在或尚未就绪")
    profile = (
        store().get_model_profile(context.user.user_id, payload.model_profile_id)
        if payload.model_profile_id
        else next(iter(store().list_model_profiles(context.user.user_id)), None)
    )
    if profile is None:
        raise HTTPException(status_code=409, detail="请先添加支持图片输入的模型配置")
    try:
        api_key = secret_box().decrypt(
            context.user.user_id,
            str(profile["profile_id"]),
            str(profile["key_nonce"]),
            str(profile["key_ciphertext"]),
        )
        return DocumentAskResponse.model_validate(
            evidence_agent().ask(
                context.user.user_id,
                document_id,
                payload.question.strip(),
                profile,
                api_key,
            )
        )
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="视觉模型请求超时") from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail="视觉模型请求失败") from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
