from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, HTTPException

from agent.graph import PaperRAGAgent
from api.schemas import (
    AskRequest,
    AskResponse,
    CollectionsResponse,
    IngestArxivRequest,
    IngestArxivResponse,
)
from config.settings import BASE_DIR, load_settings
from indexing.build_index import build_collection
from indexing.manager import IndexManager
from tools import ArxivTool

router = APIRouter()


@lru_cache(maxsize=16)
def _get_agent(collection: str) -> PaperRAGAgent:
    settings = load_settings()
    return PaperRAGAgent(settings=settings, collection=collection)


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    settings = load_settings()
    collection = req.collection or settings.project.default_collection
    manager = IndexManager(settings)

    if collection not in manager.list_collections():
        raise HTTPException(status_code=400, detail=f"集合不存在: {collection}")

    try:
        agent = _get_agent(collection)
        result = agent.ask(req.question)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"内部错误: {exc}") from exc

    return AskResponse(
        collection=collection,
        answer=result.answer,
        steps=result.steps,
        sources=result.sources,
    )


@router.get("/collections", response_model=CollectionsResponse)
def list_collections() -> CollectionsResponse:
    settings = load_settings()
    manager = IndexManager(settings)
    return CollectionsResponse(collections=manager.list_collections())


@router.post("/ingest_arxiv", response_model=IngestArxivResponse)
def ingest_arxiv(req: IngestArxivRequest) -> IngestArxivResponse:
    """在线检索 arXiv，下载 PDF 到指定集合并重建该集合索引。"""
    settings = load_settings()
    collection = req.collection or "arxiv"

    tool = ArxivTool(settings)
    tool.download_dir = BASE_DIR / settings.project.data_root / collection
    try:
        result = tool.run(req.query, max_results=req.max_results, download=True)
        downloaded = [s["paper_id"] for s in result.sources]
        if not downloaded:
            raise HTTPException(status_code=404, detail="arXiv 未检索到论文")
        n = build_collection(settings, collection)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"入库失败: {exc}") from exc

    _get_agent.cache_clear()  # 索引已更新，丢弃缓存的 agent
    return IngestArxivResponse(
        collection=collection, downloaded=downloaded, indexed_chunks=n
    )
