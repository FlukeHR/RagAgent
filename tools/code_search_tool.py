from __future__ import annotations

from config.settings import BASE_DIR, Settings
from retrieval.retriever import CodeRetriever, RetrievalResult


class CodeSearchTool:
    def __init__(self, settings: Settings, repo_name: str) -> None:
        index_dir = BASE_DIR / settings.index.index_root / repo_name
        self.retriever = CodeRetriever(settings=settings, index_dir=str(index_dir))

    def run(self, query: str) -> list[RetrievalResult]:
        return self.retriever.search(query)
