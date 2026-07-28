from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from retrieval.chunker import Chunk
from retrieval.reranker import Reranker
from retrieval.analyzer import QueryAnalyzer


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    confidence: float
    backend: str
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    lexical_anchor_score: float = 0.0


class ReciprocalRankFusion:
    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(
        self,
        ranked_lists: Sequence[Sequence[tuple[Chunk, float]]],
    ) -> list[tuple[Chunk, float]]:
        scores: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for ranked in ranked_lists:
            for rank, (chunk, _) in enumerate(ranked, start=1):
                by_id[chunk.chunk_id] = chunk
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                    self.k + rank
                )
        return sorted(
            ((by_id[chunk_id], score) for chunk_id, score in scores.items()),
            key=lambda item: item[1],
            reverse=True,
        )


class ParentDiversifier:
    def __init__(self, max_per_parent: int = 2) -> None:
        self.max_per_parent = max_per_parent

    def apply(
        self,
        candidates: Sequence[tuple[Chunk, float]],
        limit: int,
    ) -> list[tuple[Chunk, float]]:
        selected: list[tuple[Chunk, float]] = []
        counts: dict[str, int] = {}
        seen_content: set[str] = set()
        for chunk, score in candidates:
            content_key = chunk.content_hash
            if content_key in seen_content:
                continue
            parent = chunk.parent_id or chunk.chunk_id
            if counts.get(parent, 0) >= self.max_per_parent:
                continue
            selected.append((chunk, score))
            seen_content.add(content_key)
            counts[parent] = counts.get(parent, 0) + 1
            if len(selected) >= limit:
                break
        return selected


class ScoreCalibrator:
    @staticmethod
    def confidence(score: float, backend: str) -> float:
        value = max(-30.0, min(30.0, float(score)))
        if backend == "cross_encoder":
            return 1.0 / (1.0 + math.exp(-value))
        # Fallback scores are non-negative overlap + recall scores. This monotonic
        # mapping is intentionally separate from CE logits and must be calibrated
        # with the retrieval benchmark before changing production thresholds.
        return max(0.0, min(1.0, value / (1.0 + value)))


class RetrievalPipeline:
    """Shared fusion/diversification/reranking pipeline for production and evaluation."""

    def __init__(
        self,
        reranker: Reranker,
        rrf_k: int = 60,
        max_chunks_per_parent: int = 2,
    ) -> None:
        self.reranker = reranker
        self.fusion = ReciprocalRankFusion(rrf_k)
        self.diversifier = ParentDiversifier(max_chunks_per_parent)

    def run(
        self,
        dense: Sequence[tuple[Chunk, float]],
        sparse: Sequence[tuple[Chunk, float]] | None,
        query: str,
        recall_limit: int,
        output_limit: int,
        use_reranker: bool = True,
    ) -> list[RetrievalResult]:
        dense_scores = {chunk.chunk_id: score for chunk, score in dense}
        sparse_scores = {chunk.chunk_id: score for chunk, score in sparse or []}
        if sparse is not None:
            fused = self.fusion.fuse([dense, sparse])
        else:
            fused = list(dense)
        candidates = self.diversifier.apply(fused, recall_limit)
        if use_reranker:
            ranked = self.reranker.rerank(query, candidates, top_n=output_limit)
            backend = self.reranker.backend
        else:
            ranked = candidates[:output_limit]
            backend = "fusion" if sparse is not None else "dense"
        fusion_scores = {chunk.chunk_id: score for chunk, score in candidates}
        results: list[RetrievalResult] = []
        for chunk, score in ranked:
            anchor_score = self.reranker.analyzer.identifier_overlap(
                query,
                f"{chunk.paper_title}\n{chunk.section}\n{chunk.content}",
            )
            confidence = max(
                ScoreCalibrator.confidence(float(score), backend),
                anchor_score,
            )
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    confidence=confidence,
                    backend=backend,
                    dense_score=dense_scores.get(chunk.chunk_id),
                    sparse_score=sparse_scores.get(chunk.chunk_id),
                    fusion_score=fusion_scores.get(chunk.chunk_id),
                    rerank_score=float(score) if use_reranker else None,
                    lexical_anchor_score=anchor_score,
                )
            )
        return results


def rank_in_memory(
    query: str,
    chunks: Sequence[Chunk],
    vectors,
    query_vector,
    *,
    analyzer: QueryAnalyzer,
    reranker: Reranker,
    top_k: int,
    top_n: int,
    mode: str = "hybrid",
    use_reranker: bool = True,
    rrf_k: int = 60,
    max_chunks_per_parent: int = 2,
):
    """Use the exact production fusion/diversification/rerank stages on a test corpus."""

    import numpy as np

    dense: list[tuple[Chunk, float]] = []
    sparse: list[tuple[Chunk, float]] | None = None
    if mode in {"dense", "hybrid"}:
        scores = np.asarray(vectors) @ np.asarray(query_vector)
        indices = np.argsort(-scores)[:top_k]
        dense = [(chunks[int(index)], float(scores[index])) for index in indices]
    if mode in {"sparse", "hybrid"}:
        try:
            from rank_bm25 import BM25Okapi

            bm25 = BM25Okapi([analyzer.tokens(chunk.content) for chunk in chunks])
            scores = bm25.get_scores(analyzer.tokens(query))
            indices = np.argsort(-scores)[:top_k]
            sparse = [
                (chunks[int(index)], float(scores[index])) for index in indices
            ]
        except ImportError:
            sparse = []
    if mode == "sparse":
        dense, sparse = sparse or [], None
    if mode not in {"dense", "sparse", "hybrid"}:
        raise ValueError(f"unknown retrieval mode: {mode}")
    return RetrievalPipeline(
        reranker,
        rrf_k=rrf_k,
        max_chunks_per_parent=max_chunks_per_parent,
    ).run(
        dense,
        sparse,
        query,
        recall_limit=top_k,
        output_limit=top_n,
        use_reranker=use_reranker,
    )
