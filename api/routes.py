from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, HTTPException

from agent.graph import CodeRAGAgent
from api.schemas import AskRequest, AskResponse, ReposResponse
from config.settings import load_settings
from indexing.manager import IndexManager

router = APIRouter()


@lru_cache(maxsize=16)
def _get_agent(repo_name: str) -> CodeRAGAgent:
    settings = load_settings()
    return CodeRAGAgent(settings=settings, repo_name=repo_name)


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    settings = load_settings()
    repo_name = req.repo_name or settings.project.default_repo
    manager = IndexManager(settings)

    if repo_name not in manager.list_repositories():
        raise HTTPException(status_code=400, detail=f"Repository not found: {repo_name}")

    try:
        agent = _get_agent(repo_name)
        result = agent.ask(req.question)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc

    return AskResponse(repo_name=repo_name, answer=result.answer, steps=result.steps, sources=result.sources)


@router.get("/repos", response_model=ReposResponse)
def list_repos() -> ReposResponse:
    settings = load_settings()
    manager = IndexManager(settings)
    return ReposResponse(repositories=manager.list_repositories())
