from __future__ import annotations

import re
from functools import lru_cache

from fastapi import APIRouter, HTTPException

from agent.graph import PaperRAGAgent
from api.schemas import (
    AskRequest,
    AskResponse,
    ConversationResponse,
    IngestArxivRequest,
    IngestArxivResponse,
    SourceItem,
    TitleRequest,
    TitleResponse,
    Turn,
)
from config.settings import load_settings
from llm.model import LLMClient
from services import ArxivSearchService, PaperLibraryService

router = APIRouter()


@lru_cache(maxsize=1)
def _get_agent() -> PaperRAGAgent:
    return PaperRAGAgent(settings=load_settings())


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        agent = _get_agent()
        history = [turn.model_dump() for turn in req.history]
        result = agent.ask(
            req.question,
            history=history,
            session_id=req.session_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"内部错误: {exc}") from exc

    return AskResponse(
        answer=result.answer,
        status=result.status,
        steps=result.steps,
        sources=[SourceItem.model_validate(source) for source in result.sources],
        trace=result.trace,
    )


def _truncate_title(text: str, limit: int = 20) -> str:
    """降级标题：取首条用户消息首行前 limit 字。"""
    line = re.sub(r"\s+", " ", (text or "").strip())
    if not line:
        return "新对话"
    return line[:limit] + ("…" if len(line) > limit else "")


@router.post("/title", response_model=TitleResponse)
def make_title(req: TitleRequest) -> TitleResponse:
    """生成简短会话标题；模型不可用时截断首条用户消息。"""
    first_user = next(
        (message.content for message in req.messages if message.role == "user"),
        "",
    )
    llm = LLMClient(load_settings().llm)
    if not llm.supports_agentic():
        return TitleResponse(title=_truncate_title(first_user))

    conversation = "\n".join(
        f"{message.role}: {message.content[:500]}"
        for message in req.messages[:4]
    )
    prompt = (
        "请用不超过 12 个汉字概括下面这轮对话的主题，作为对话列表里的标题。"
        "只输出标题本身，不要标点、不要引号、不要解释。\n\n"
        f"对话：\n{conversation}\n\n标题："
    )
    try:
        output = llm.generate(
            prompt,
            system="你是对话标题生成助手，只输出简短标题。",
        )
        title = re.sub(
            r"\s+", " ", (output or "").strip().strip("\"'「」“”。.")
        ).strip()
        if (
            not title
            or len(title) > 30
            or "降级模式" in title
            or "调用失败" in title
        ):
            title = _truncate_title(first_user)
    except Exception:  # noqa: BLE001
        title = _truncate_title(first_user)
    return TitleResponse(title=title)


@router.post("/ingest_arxiv", response_model=IngestArxivResponse)
def ingest_arxiv(req: IngestArxivRequest) -> IngestArxivResponse:
    """下载 arXiv PDF 到统一论文库并增量重建索引。"""
    settings = load_settings()
    try:
        papers = ArxivSearchService(settings).search(
            req.query,
            max_results=req.max_results,
        )
        arxiv_ids = [paper.arxiv_id for paper in papers]
        arxiv_ids = arxiv_ids[: settings.arxiv.max_ingest_papers]
        if not arxiv_ids:
            raise HTTPException(
                status_code=404,
                detail="arXiv 未检索到可下载论文",
            )

        report = PaperLibraryService(settings).ingest_arxiv(arxiv_ids)
        downloaded = [*report.downloaded, *report.reused]
        if not downloaded:
            raise HTTPException(
                status_code=502,
                detail="arXiv 论文下载失败",
            )

        indexed_chunks = report.indexed_chunks
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"入库失败: {exc}") from exc

    _get_agent.cache_clear()
    return IngestArxivResponse(
        downloaded=downloaded,
        indexed_chunks=indexed_chunks,
    )


@router.get("/sessions/{session_id}", response_model=ConversationResponse)
def get_session(session_id: str) -> ConversationResponse:
    try:
        state, history = _get_agent().memory.load(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ConversationResponse(
        session_id=session_id,
        state=state.__dict__,
        history=[Turn.model_validate(item) for item in history],
    )


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str) -> None:
    try:
        _get_agent().memory.delete(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
