from __future__ import annotations

from dataclasses import dataclass

from config.settings import Settings
from retrieval.chunker import Chunk
from retrieval.embedder import Embedder
from retrieval.reranker import Reranker
from retrieval.vector_store import VectorStore


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


class Retriever:
    """向量召回 + 可选重排，面向论文 chunk。"""

    def __init__(self, settings: Settings, index_dir: str) -> None:
        self.settings = settings
        self.embedder = Embedder(
            model_name=settings.embedding.model_name,
            use_sentence_transformers=settings.embedding.use_sentence_transformers,
        )
        self.store = VectorStore(index_dir=index_dir)
        self.reranker = Reranker(
            model_name=settings.rerank.model_name,
            use_cross_encoder=settings.rerank.use_cross_encoder,
        )

    def search(self, query: str) -> list[RetrievalResult]:
        q_vec = self.embedder.encode([query])
        recall = self.store.search(q_vec, top_k=self.settings.index.top_k_recall)
        reranked = self.reranker.rerank(
            query, recall, top_n=self.settings.index.top_n_rerank
        )
        return [RetrievalResult(chunk=chunk, score=score) for chunk, score in reranked]
