from __future__ import annotations

import re
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from agent.graph import PaperRAGAgent
from api.schemas import (
    AskRequest,
    AskResponse,
    CollectionsResponse,
    IngestArxivRequest,
    IngestArxivResponse,
    TitleRequest,
    TitleResponse,
)
from config.settings import BASE_DIR, load_settings
from llm.model import LLMClient
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
        history = [t.model_dump() for t in req.history]
        result = agent.ask(req.question, history=history)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"内部错误: {exc}") from exc

    return AskResponse(
        collection=collection,
        answer=result.answer,
        steps=result.steps,
        sources=result.sources,
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
    """为一段对话自动生成简短标题。无 LLM 后端 / 失败时降级为截取首条用户消息。"""
    first_user = next((m.content for m in req.messages if m.role == "user"), "")
    settings = load_settings()
    llm = LLMClient(settings.llm)
    if not llm.supports_agentic():  # 无真实后端，直接降级截断
        return TitleResponse(title=_truncate_title(first_user))

    convo = "\n".join(f"{m.role}: {m.content[:500]}" for m in req.messages[:4])
    prompt = (
        "请用不超过 12 个汉字概括下面这轮对话的主题，作为对话列表里的标题。"
        "只输出标题本身，不要标点、不要引号、不要解释。\n\n"
        f"对话：\n{convo}\n\n标题："
    )
    try:
        out = llm.generate(prompt, system="你是对话标题生成助手，只输出简短标题。")
        title = re.sub(r"\s+", " ", (out or "").strip().strip("\"'「」“”。.")).strip()
        if not title or len(title) > 30 or "降级模式" in title or "调用失败" in title:
            title = _truncate_title(first_user)
    except Exception:  # noqa: BLE001 - 生成失败不阻断，降级截断
        title = _truncate_title(first_user)
    return TitleResponse(title=title)


def _search_phrases(snippet: str, max_phrases: int = 6) -> list[str]:
    """把片段拆成若干短句，用于在 PDF 页内逐句定位（长串易因换行/连字符匹配失败）。"""
    text = re.sub(r"\s+", " ", snippet).strip()
    if not text:
        return []
    phrases = [
        p.strip()[:120]
        for p in re.split(r"(?<=[.!?。！？])\s+", text)
        if len(p.strip()) >= 15
    ]
    return phrases[:max_phrases] or [text[:120]]


def _locate(doc, snippet: str) -> tuple[list, int]:
    """返回 (高亮矩形列表, 0基页码)。逐页统计命中的短句矩形，取覆盖最高的页。

    比"首个命中页"更稳：避免某短句在靠前页偶然出现导致定位到无关页。
    """
    phrases = _search_phrases(snippet)
    if not phrases:
        return [], 0

    best_rects: list = []
    best_page = 0
    best_score = 0
    for pno in range(len(doc)):
        page = doc[pno]
        rects: list = []
        matched_phrases = 0
        for ph in phrases:
            hits = page.search_for(ph)
            if hits:
                matched_phrases += 1
                rects.extend(hits)
        if matched_phrases > best_score:
            best_score, best_rects, best_page = matched_phrases, rects, pno
    return best_rects, best_page


def _pdf_path_or_404(collection: str, paper_id: str):
    settings = load_settings()
    pdf_path = BASE_DIR / settings.project.data_root / collection / f"{paper_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"无 PDF 可预览: {paper_id}")
    try:
        import fitz  # PyMuPDF  # noqa: F401
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="缺少 PyMuPDF") from exc
    return pdf_path


@router.get("/preview/meta")
def preview_meta(
    collection: str = Query(...),
    paper_id: str = Query(...),
    snippet: str | None = Query(None),
    page_start: int | None = Query(None),
) -> dict:
    """返回 PDF 总页数与引用所在页（1 基），供前端加载整份文档并滚动定位。"""
    import fitz

    pdf_path = _pdf_path_or_404(collection, paper_id)
    with fitz.open(str(pdf_path)) as doc:
        match_page = 1
        if page_start:
            match_page = min(max(page_start, 1), len(doc))
        elif snippet:
            _, pno = _locate(doc, snippet)
            match_page = pno + 1
        return {"pages": len(doc), "match_page": match_page}


def _parse_bbox_param(bbox: str | None):
    if not bbox:
        return None
    try:
        parts = [float(x.strip()) for x in bbox.split(",")]
    except ValueError:
        return None
    if len(parts) != 4:
        return None
    return parts


@router.get("/preview/page")
def preview_page(
    collection: str = Query(...),
    paper_id: str = Query(...),
    page: int = Query(1, ge=1),
    snippet: str | None = Query(None),
    bbox: str | None = Query(None),
    zoom: float = Query(1.6, ge=0.5, le=4.0),
) -> Response:
    """渲染 PDF 指定页为 PNG；若该页含引用片段则就地高亮。"""
    import fitz

    pdf_path = _pdf_path_or_404(collection, paper_id)
    with fitz.open(str(pdf_path)) as doc:
        pno = min(page - 1, len(doc) - 1)
        pg = doc[pno]
        bbox_parts = _parse_bbox_param(bbox)
        if bbox_parts:
            rect = fitz.Rect(*bbox_parts) & pg.rect
            if not rect.is_empty:
                pg.draw_rect(rect, color=(1, 0.72, 0), width=2)
        if snippet:
            rects: list = []
            for ph in _search_phrases(snippet):
                rects.extend(pg.search_for(ph))
            if rects:
                pg.add_highlight_annot(rects)
        pix = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        png = pix.tobytes("png")
    return Response(content=png, media_type="image/png")


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
