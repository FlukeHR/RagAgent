from __future__ import annotations

from config.settings import Settings
from retrieval.reranker import Reranker
from retrieval.retriever import RetrievalResult


class RerankTool:
    def __init__(self, settings: Settings) -> None:
        self.reranker = Reranker(
            model_name=settings.rerank.model_name,
            use_cross_encoder=settings.rerank.use_cross_encoder,
        )

    def run(self, query: str, candidates: list[RetrievalResult], top_n: int) -> list[RetrievalResult]:
        pairs = [(item.chunk, item.score) for item in candidates]
        reranked = self.reranker.rerank(query, pairs, top_n=top_n)
        return [RetrievalResult(chunk=chunk, score=score) for chunk, score in reranked]
