from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import fitz
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.dependencies import (
    AuthContext,
    agent_pool,
    current_auth,
    ingest_manager,
    require_csrf,
    secret_box,
    settings,
    store,
)
from api.schemas import (
    ArxivCandidate,
    ArxivConfirmRequest,
    ArxivProposalRequest,
    ArxivProposalResponse,
    AskResponse,
    ModelProfileCreate,
    ModelProfileResponse,
    ModelProfileUpdate,
    SessionAskRequest,
    SessionCreateRequest,
    SessionImportRequest,
    SessionUpdateRequest,
    SourceItem,
)
from config.settings import LLMConfig
from llm.model import LLMClient
from retrieval.documents import PaperRepository
from services import ArxivSearchService
from services.security import validate_model_endpoint
from services.user_scope import scoped_settings, user_paths


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


def _safe_title(value: str, fallback: str = "未命名论文") -> str:
    text = re.sub(r"\s+", " ", value).strip()
    return text[:200] or fallback


@router.get("/dashboard")
def dashboard(context: AuthContext = Depends(current_auth)) -> dict[str, Any]:
    """Return compact user-owned dashboard statistics and recent activity."""

    result = store().dashboard(context.user.user_id)
    result["model_profiles"] = len(store().list_model_profiles(context.user.user_id))
    return result


# Model profiles ---------------------------------------------------------------
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
        profile = store().update_model_profile(
            context.user.user_id, profile_id, values
        )
    except Exception as exc:
        if "UNIQUE constraint failed" in str(exc):
            raise HTTPException(status_code=409, detail="模型配置名称已存在") from exc
        raise
    assert profile is not None
    agent_pool().invalidate(context.user.user_id, profile_id)
    return _profile_response(profile)


@router.delete("/model-profiles/{profile_id}", status_code=204)
def delete_model_profile(
    profile_id: str,
    context: AuthContext = Depends(require_csrf),
) -> None:
    if not store().delete_model_profile(context.user.user_id, profile_id):
        raise HTTPException(status_code=404, detail="模型配置不存在")
    agent_pool().invalidate(context.user.user_id, profile_id)


@router.post("/model-profiles/{profile_id}/test")
def test_model_profile(
    profile_id: str,
    context: AuthContext = Depends(require_csrf),
) -> dict[str, Any]:
    """Run one explicit, user-triggered minimal model request."""

    profile = store().get_model_profile(context.user.user_id, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="模型配置不存在")
    try:
        api_key = secret_box().decrypt(
            context.user.user_id,
            profile_id,
            str(profile["key_nonce"]),
            str(profile["key_ciphertext"]),
        )
        config: LLMConfig = scoped_settings(
            settings(),
            context.user.user_id,
            model_name=str(profile["model_name"]),
            api_base=str(profile["api_base"]),
            api_key=api_key,
        ).llm
        started = time.perf_counter()
        output = LLMClient(config).generate(
            "Reply with OK only.", system="This is a connection test. Reply with OK only."
        )
        return {
            "ok": bool(output.strip()),
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    except Exception as exc:  # noqa: BLE001 - provider details must not leak secrets
        raise HTTPException(
            status_code=502,
            detail=f"模型连接失败（{type(exc).__name__}）",
        ) from exc


# Conversations ---------------------------------------------------------------
@router.get("/sessions")
def list_sessions(context: AuthContext = Depends(current_auth)) -> list[dict[str, Any]]:
    return store().list_conversations(context.user.user_id)


@router.post("/sessions", status_code=201)
def create_session(
    payload: SessionCreateRequest,
    context: AuthContext = Depends(require_csrf),
) -> dict[str, Any]:
    profile_id = payload.model_profile_id
    if profile_id is None:
        profiles = store().list_model_profiles(context.user.user_id)
        profile_id = str(profiles[0]["profile_id"]) if profiles else None
    elif store().get_model_profile(context.user.user_id, profile_id) is None:
        raise HTTPException(status_code=404, detail="模型配置不存在")
    return store().create_conversation(
        context.user.user_id, _safe_title(payload.title, "新对话"), profile_id
    )


@router.get("/sessions/{conversation_id}")
def get_session(
    conversation_id: str,
    context: AuthContext = Depends(current_auth),
) -> dict[str, Any]:
    conversation = store().get_conversation(context.user.user_id, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    return conversation


@router.patch("/sessions/{conversation_id}")
def update_session(
    conversation_id: str,
    payload: SessionUpdateRequest,
    context: AuthContext = Depends(require_csrf),
) -> dict[str, Any]:
    fields = payload.model_fields_set
    if "model_profile_id" in fields and payload.model_profile_id is not None:
        if store().get_model_profile(
            context.user.user_id, payload.model_profile_id
        ) is None:
            raise HTTPException(status_code=404, detail="模型配置不存在")
    conversation = store().update_conversation(
        context.user.user_id,
        conversation_id,
        title=_safe_title(payload.title) if payload.title is not None else None,
        model_profile_id=payload.model_profile_id,
        update_profile="model_profile_id" in fields,
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    return conversation


@router.delete("/sessions/{conversation_id}", status_code=204)
def delete_session(
    conversation_id: str,
    context: AuthContext = Depends(require_csrf),
) -> None:
    if not store().delete_conversation(context.user.user_id, conversation_id):
        raise HTTPException(status_code=404, detail="对话不存在")


@router.post("/sessions/{conversation_id}/ask", response_model=AskResponse)
def ask_session(
    conversation_id: str,
    payload: SessionAskRequest,
    context: AuthContext = Depends(require_csrf),
) -> AskResponse:
    conversation = store().get_conversation(context.user.user_id, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    profile_id = conversation.get("model_profile_id")
    if not profile_id:
        raise HTTPException(status_code=409, detail="请先为对话选择模型配置")
    profile = store().get_model_profile(context.user.user_id, str(profile_id))
    if profile is None:
        raise HTTPException(status_code=409, detail="当前模型配置已被删除")
    previous = list(conversation.get("messages") or [])
    history = [
        {"role": item["role"], "content": item["content"]}
        for item in previous
        if item.get("role") in {"user", "assistant"}
    ]
    store().add_message(
        context.user.user_id, conversation_id, "user", payload.question
    )
    if not previous and str(conversation["title"]) == "新对话":
        store().update_conversation(
            context.user.user_id,
            conversation_id,
            title=_safe_title(payload.question[:24], "新对话"),
        )
    try:
        result = agent_pool().get(context.user.user_id, profile).ask(
            payload.question,
            history=history,
            session_id=None,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail="当前论文库尚未建立可用索引") from exc
    except Exception as exc:  # noqa: BLE001 - normalize provider/runtime failures
        raise HTTPException(
            status_code=500, detail=f"问答执行失败（{type(exc).__name__}）"
        ) from exc
    sources = _present_sources(context.user.user_id, result.sources)
    actions = _suggested_actions(payload.question, sources)
    store().add_message(
        context.user.user_id,
        conversation_id,
        "assistant",
        result.answer,
        status=result.status,
        sources=sources,
        steps=result.steps,
        trace=result.trace,
        actions=actions,
    )
    return AskResponse(
        answer=result.answer,
        status=result.status,
        steps=result.steps,
        sources=[SourceItem.model_validate(source) for source in sources],
        trace=result.trace,
        suggested_actions=actions,
    )


@router.post("/sessions/import")
def import_sessions(
    payload: SessionImportRequest,
    context: AuthContext = Depends(require_csrf),
) -> dict[str, int]:
    imported = 0
    for item in payload.sessions:
        conversation = store().create_conversation(
            context.user.user_id, _safe_title(item.title, "旧对话"), None
        )
        for message in item.messages:
            store().add_message(
                context.user.user_id,
                str(conversation["conversation_id"]),
                message.role,
                message.content[:20000],
            )
        imported += 1
    return {"imported": imported}


# Papers and ingestion --------------------------------------------------------
@router.get("/papers")
def list_papers(context: AuthContext = Depends(current_auth)) -> list[dict[str, Any]]:
    return store().list_papers(context.user.user_id)


@router.post("/papers/upload", status_code=202)
def upload_paper(
    file: UploadFile,
    context: AuthContext = Depends(require_csrf),
) -> dict[str, Any]:
    """Stream one PDF to a user-owned temporary file and queue bounded ingestion."""

    if store().paper_count(context.user.user_id) >= settings().app.max_papers_per_user:
        raise HTTPException(status_code=409, detail="论文数量已达上限")
    user_pending, global_pending = store().pending_job_counts(context.user.user_id)
    if user_pending >= settings().app.max_pending_jobs_per_user:
        raise HTTPException(status_code=429, detail="你的入库队列已满")
    if global_pending >= settings().mineru.max_pending_jobs:
        raise HTTPException(status_code=429, detail="全局入库队列已满")
    filename = Path(file.filename or "paper.pdf").name
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")
    paths = user_paths(settings(), context.user.user_id)
    storage_id = uuid.uuid4().hex
    target = PaperRepository(paths.papers).target(storage_id)
    part = target.with_suffix(".pdf.part")
    max_bytes = int(settings().mineru.max_pdf_mb * 1024 * 1024)
    size = 0
    try:
        with part.open("wb") as handle:
            while block := file.file.read(1 << 16):
                size += len(block)
                if size > max_bytes:
                    raise ValueError("PDF 超过大小限制")
                handle.write(block)
        with part.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise ValueError("文件不是有效 PDF")
        with fitz.open(part) as document:
            page_count = int(document.page_count)
        if page_count <= 0 or page_count > settings().mineru.max_pages:
            raise ValueError("PDF 页数超过限制")
        os.replace(part, target)
        paper = store().create_paper(
            context.user.user_id,
            storage_id,
            _safe_title(Path(filename).stem),
            filename,
            "upload",
            status="queued",
            page_count=page_count,
        )
        try:
            job = ingest_manager().queue_paper(
                context.user.user_id, str(paper["paper_id"])
            )
        except Exception:
            store().delete_paper_record(context.user.user_id, str(paper["paper_id"]))
            target.unlink(missing_ok=True)
            raise
        return {"paper": paper, "job": job}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        part.unlink(missing_ok=True)
        file.file.close()


@router.get("/papers/{paper_id}/pdf", response_class=FileResponse)
def preview_paper_pdf(
    paper_id: str,
    context: AuthContext = Depends(current_auth),
) -> FileResponse:
    paper = store().get_paper(context.user.user_id, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="论文不存在")
    repository = PaperRepository(user_paths(settings(), context.user.user_id).papers)
    target = repository.resolve(str(paper["storage_id"]), (".pdf",))
    if target is None:
        raise HTTPException(status_code=404, detail="PDF 文件不存在")
    return FileResponse(
        target,
        media_type="application/pdf",
        filename=str(paper["original_filename"]),
        content_disposition_type="inline",
    )


@router.post("/papers/{paper_id}/retry", status_code=202)
def retry_paper(
    paper_id: str,
    context: AuthContext = Depends(require_csrf),
) -> dict[str, Any]:
    paper = store().get_paper(context.user.user_id, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="论文不存在")
    job = next(
        (
            item
            for item in store().list_jobs(context.user.user_id)
            if item.get("paper_id") == paper_id and item.get("status") == "failed"
        ),
        None,
    )
    if job is None:
        raise HTTPException(status_code=409, detail="没有可重试的失败任务")
    try:
        return ingest_manager().retry(context.user.user_id, str(job["job_id"]))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/papers/{paper_id}", status_code=204)
def delete_paper(
    paper_id: str,
    context: AuthContext = Depends(require_csrf),
) -> None:
    try:
        ingest_manager().delete_paper(context.user.user_id, paper_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="论文不存在") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/arxiv/ingest/proposals", response_model=ArxivProposalResponse)
def create_arxiv_proposal(
    payload: ArxivProposalRequest,
    context: AuthContext = Depends(require_csrf),
) -> ArxivProposalResponse:
    try:
        papers = ArxivSearchService(settings()).search(payload.query, payload.max_results)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="arXiv 搜索失败") from exc
    if not papers:
        raise HTTPException(status_code=404, detail="未找到匹配的 arXiv 论文")
    candidates = [
        {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": list(paper.authors),
            "summary": paper.summary,
            "published": paper.published.isoformat() if paper.published else None,
            "entry_url": _https_url(paper.entry_url),
        }
        for paper in papers
    ]
    proposal = store().create_proposal(
        context.user.user_id,
        payload.query,
        candidates,
        settings().mineru.proposal_ttl_seconds,
    )
    return ArxivProposalResponse(
        proposal_id=str(proposal["proposal_id"]),
        expires_at=float(proposal["expires_at"]),
        candidates=[ArxivCandidate.model_validate(item) for item in candidates],
    )


@router.post("/arxiv/ingest/confirm", status_code=202)
def confirm_arxiv_ingest(
    payload: ArxivConfirmRequest,
    context: AuthContext = Depends(require_csrf),
) -> dict[str, Any]:
    try:
        candidate = store().consume_proposal(
            context.user.user_id,
            payload.proposal_id,
            payload.arxiv_id,
            max_pending_user=settings().app.max_pending_jobs_per_user,
            max_pending_global=settings().mineru.max_pending_jobs,
        )
        return ingest_manager().queue_arxiv(
            context.user.user_id,
            payload.proposal_id,
            "arXiv proposal",
            candidate,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="入库候选不存在") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


@router.get("/ingest/jobs")
def list_ingest_jobs(
    context: AuthContext = Depends(current_auth),
) -> list[dict[str, Any]]:
    return store().list_jobs(context.user.user_id)


@router.get("/ingest/jobs/{job_id}")
def get_ingest_job(
    job_id: str,
    context: AuthContext = Depends(current_auth),
) -> dict[str, Any]:
    job = store().get_job(context.user.user_id, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job


def _present_sources(user_id: str, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    papers = store().list_papers(user_id)
    by_storage = {str(item["storage_id"]): item for item in papers}
    presented: list[dict[str, Any]] = []
    for source in sources:
        item = dict(source)
        raw = str(item.pop("source", "") or "")
        parsed = urlparse(raw)
        if parsed.scheme in {"http", "https"}:
            item.update(
                {
                    "source_kind": "external_url",
                    "citation_url": _https_url(raw),
                    "preview_kind": "web",
                }
            )
        else:
            paper = by_storage.get(str(item.get("paper_id") or ""))
            item.update(
                {
                    "paper_id": str(paper["paper_id"]) if paper else "",
                    "source_kind": "library_pdf",
                    "citation_url": None,
                    "preview_kind": "pdf",
                }
            )
        presented.append(item)
    return presented


def _suggested_actions(question: str, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    arxiv_ids = [
        str(source.get("paper_id") or "")
        for source in sources
        if source.get("source_kind") == "external_url"
        and "search_arxiv" in source.get("origin_tools", [])
    ]
    if not arxiv_ids:
        return []
    return [
        {
            "type": "arxiv_ingest",
            "query": question[:1000],
            "arxiv_ids": list(dict.fromkeys(arxiv_ids)),
        }
    ]


def _https_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.hostname and parsed.hostname.endswith("arxiv.org"):
        parsed = parsed._replace(scheme="https", netloc=parsed.hostname)
    return urlunparse(parsed)
