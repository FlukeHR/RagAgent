from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    """稠密向量 + BM25 混合召回（RRF 融合）+ 可选重排，面向论文 chunk。"""

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
        self._bm25 = None  # 懒构建：首次检索时用 store.chunks 现建并缓存

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def _bm25_index(self):
        """查询时用 store.chunks 现建 BM25（论文集合不大，开销可忽略）。

        rank_bm25 不可用或无语料时返回 None，检索自动退化为纯稠密向量。
        """
        if self._bm25 is None:
            if not self.store.chunks:
                self.store.load()
            corpus = [self._tokenize(c.content) for c in self.store.chunks]
            try:
                from rank_bm25 import BM25Okapi

                self._bm25 = BM25Okapi(corpus) if corpus else False
            except ImportError:
                self._bm25 = False
        return self._bm25 or None

    @staticmethod
    def _rrf_fuse(
        ranked_lists: list[list[tuple[Chunk, float]]], k: int = 60
    ) -> list[tuple[Chunk, float]]:
        """Reciprocal Rank Fusion：按各路名次倒数求和，无需调权重、对分数量纲不敏感。"""
        scores: dict[str, float] = {}
        chunk_by_id: dict[str, Chunk] = {}
        for lst in ranked_lists:
            for rank, (chunk, _) in enumerate(lst):
                cid = chunk.chunk_id
                chunk_by_id[cid] = chunk
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        fused = sorted(
            ((chunk_by_id[cid], sc) for cid, sc in scores.items()),
            key=lambda x: x[1],
            reverse=True,
        )
        return fused

    def search(self, query: str) -> list[RetrievalResult]:
        top_k = self.settings.index.top_k_recall
        q_vec = self.embedder.encode([query])
        dense = self.store.search(q_vec, top_k=top_k)

        bm25 = self._bm25_index()
        if bm25 is not None:
            bm25_scores = bm25.get_scores(self._tokenize(query))
            top_idx = np.argsort(-bm25_scores)[:top_k]
            sparse = [
                (self.store.chunks[int(i)], float(bm25_scores[int(i)])) for i in top_idx
            ]
            candidates = self._rrf_fuse([dense, sparse])[:top_k]
        else:
            candidates = dense

        reranked = self.reranker.rerank(
            query, candidates, top_n=self.settings.index.top_n_rerank
        )
        return [RetrievalResult(chunk=chunk, score=score) for chunk, score in reranked]
