from __future__ import annotations

import re
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from agent.graph import PaperRAGAgent
from api.schemas import (
    ArxivCandidate,
    ArxivProposalRequest,
    ArxivProposalResponse,
    AskRequest,
    AskResponse,
    ConfirmIngestRequest,
    ConversationResponse,
    IngestJobResponse,
    SourceItem,
    TitleRequest,
    TitleResponse,
    Turn,
)
from config.settings import BASE_DIR, load_settings
from llm.model import LLMClient
from retrieval.documents import InvalidPaperId, PaperRepository
from services import ArxivSearchService
from services.ingest_jobs import IngestJobManager, IngestJobStore, job_to_dict


router = APIRouter()


@lru_cache(maxsize=1)
def _get_agent() -> PaperRAGAgent:
    return PaperRAGAgent(settings=load_settings())


@lru_cache(maxsize=1)
def _get_ingest_manager() -> IngestJobManager:
    return IngestJobManager(load_settings())


@router.post("/arxiv/ingest/proposals", response_model=ArxivProposalResponse)
def create_ingest_proposal(req: ArxivProposalRequest) -> ArxivProposalResponse:
    """Search candidates and create a short-lived, non-writing proposal."""

    settings = load_settings()
    try:
        papers = ArxivSearchService(settings).search(req.query, req.max_results)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"arXiv search failed: {exc}") from exc
    if not papers:
        raise HTTPException(status_code=404, detail="No matching arXiv papers found")
    proposal = IngestJobStore(settings).create_proposal(
        req.query, [paper.arxiv_id for paper in papers]
    )
    return ArxivProposalResponse(
        proposal_id=proposal.proposal_id,
        expires_at=proposal.expires_at,
        candidates=[
            ArxivCandidate(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                authors=list(paper.authors),
                summary=paper.summary,
                published=paper.published.isoformat() if paper.published else None,
            )
            for paper in papers
        ],
    )


@router.post(
    "/arxiv/ingest/confirm", response_model=IngestJobResponse, status_code=202
)
def confirm_ingest(req: ConfirmIngestRequest) -> IngestJobResponse:
    """Consume one proposal and queue a scoped asynchronous write job."""

    try:
        job = _get_ingest_manager().confirm(req.proposal_id, req.arxiv_ids)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return IngestJobResponse.model_validate(job_to_dict(job))


@router.get("/arxiv/ingest/jobs/{job_id}", response_model=IngestJobResponse)
def get_ingest_job(job_id: str) -> IngestJobResponse:
    job = _get_ingest_manager().store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown ingest job")
    return IngestJobResponse.model_validate(job_to_dict(job))


@router.post(
    "/arxiv/ingest/jobs/{job_id}/retry",
    response_model=IngestJobResponse,
    status_code=202,
)
def retry_ingest_job(job_id: str) -> IngestJobResponse:
    try:
        job = _get_ingest_manager().retry(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return IngestJobResponse.model_validate(job_to_dict(job))


@router.get("/papers/{paper_id}/pdf", response_class=FileResponse)
def preview_pdf(paper_id: str) -> FileResponse:
    """Serve one validated repository PDF for the browser previewer."""

    settings = load_settings()
    repository = PaperRepository(BASE_DIR / settings.project.data_root)
    try:
        pdf_path = repository.resolve(paper_id, (".pdf",))
    except InvalidPaperId as exc:
        raise HTTPException(status_code=400, detail="Invalid paper identifier") from exc
    if pdf_path is None:
        raise HTTPException(status_code=404, detail=f"PDF not found: {paper_id}")
    return FileResponse(path=pdf_path, media_type="application/pdf")


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        result = _get_agent().ask(
            req.question,
            history=[turn.model_dump() for turn in req.history],
            session_id=req.session_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc
    return AskResponse(
        answer=result.answer,
        status=result.status,
        steps=result.steps,
        sources=[SourceItem.model_validate(source) for source in result.sources],
        trace=result.trace,
    )


def _truncate_title(text: str, limit: int = 20) -> str:
    line = re.sub(r"\s+", " ", (text or "").strip())
    if not line:
        return "New conversation"
    return line[:limit] + ("…" if len(line) > limit else "")


@router.post("/title", response_model=TitleResponse)
def make_title(req: TitleRequest) -> TitleResponse:
    first_user = next(
        (message.content for message in req.messages if message.role == "user"), ""
    )
    llm = LLMClient(load_settings().llm)
    if not llm.supports_agentic():
        return TitleResponse(title=_truncate_title(first_user))
    conversation = "\n".join(
        f"{message.role}: {message.content[:500]}" for message in req.messages[:4]
    )
    prompt = (
        "Create a concise conversation title of at most 12 Chinese characters or "
        "eight English words. Output only the title.\n\n"
        f"Conversation:\n{conversation}\n\nTitle:"
    )
    output = llm.generate(prompt, system="Return only a concise conversation title.")
    title = re.sub(r"\s+", " ", (output or "").strip().strip("\"'"))
    if not title or len(title) > 40 or "降級" in title:
        title = _truncate_title(first_user)
    return TitleResponse(title=title)


@router.post("/ingest_arxiv", include_in_schema=False)
def ingest_arxiv(req: ArxivProposalRequest) -> None:
    """Reject the removed one-step write endpoint."""

    del req
    raise HTTPException(
        status_code=410,
        detail=(
            "Direct ingestion was removed. Create a proposal and explicitly confirm "
            "the selected arXiv ID."
        ),
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
