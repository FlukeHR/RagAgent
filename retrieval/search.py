from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from config.settings import Settings, resolve_model_path
from retrieval.chunker import Chunk
from retrieval.index import Embedder, VectorStore


_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*")
_CJK_RUN = re.compile(r"[\u3400-\u9fff]+")
_QUERY_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "by",
    "for",
    "how",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}


class QueryAnalyzer:
    """English/CJK analyzer shared by retrieval and evidence verification."""

    def __init__(self, cjk_ngram_size: int = 2) -> None:
        self.cjk_ngram_size = max(1, cjk_ngram_size)

    def tokens(self, text: str) -> list[str]:
        lowered = text.lower()
        tokens = _WORD.findall(lowered)
        for run in _CJK_RUN.findall(lowered):
            if len(run) <= self.cjk_ngram_size:
                tokens.append(run)
            else:
                tokens.extend(
                    run[index : index + self.cjk_ngram_size]
                    for index in range(len(run) - self.cjk_ngram_size + 1)
                )
        return tokens

    def overlap(self, left: str, right: str) -> float:
        left_tokens = set(self.tokens(left))
        if not left_tokens:
            return 0.0
        return len(left_tokens & set(self.tokens(right))) / len(left_tokens)

    @staticmethod
    def identifier_overlap(query: str, evidence: str) -> float:
        query_terms = {
            token.lower()
            for token in _WORD.findall(query)
            if token.lower() not in _QUERY_WORDS
        }
        if not query_terms:
            return 0.0
        evidence_terms = {token.lower() for token in _WORD.findall(evidence)}
        return len(query_terms & evidence_terms) / len(query_terms)


class Reranker:
    """CrossEncoder reranker with deterministic lexical fallback."""

    def __init__(
        self,
        model_name: str,
        use_cross_encoder: bool = False,
        analyzer: QueryAnalyzer | None = None,
    ) -> None:
        self.model_name = model_name
        self.use_cross_encoder = use_cross_encoder
        self.analyzer = analyzer or QueryAnalyzer()
        self._cross_encoder = None
        self.load_error: str | None = None
        self._load_attempted = False
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    def _ensure_model(self) -> None:
        if not self.use_cross_encoder:
            return
        with self._load_lock:
            if self._load_attempted or not self.use_cross_encoder:
                return
            self._load_attempted = True
            try:
                from sentence_transformers import CrossEncoder

                self._cross_encoder = CrossEncoder(resolve_model_path(self.model_name))
            except Exception as exc:
                self.load_error = str(exc)

    @property
    def backend(self) -> str:
        self._ensure_model()
        return "cross_encoder" if self._cross_encoder is not None else "token_overlap"

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[Chunk, float]],
        top_n: int,
    ) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        self._ensure_model()
        if self._cross_encoder is not None:
            pairs = [[query, item[0].content] for item in candidates]
            with self._predict_lock:
                scores = self._cross_encoder.predict(pairs)
            ranked = sorted(
                zip((item[0] for item in candidates), scores),
                key=lambda item: item[1],
                reverse=True,
            )
            return [(chunk, float(score)) for chunk, score in ranked[:top_n]]
        ranked = sorted(
            (
                (chunk, self.analyzer.overlap(query, chunk.content) + base_score)
                for chunk, base_score in candidates
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_n]


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
        self, ranked_lists: Sequence[Sequence[tuple[Chunk, float]]]
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
            if chunk.content_hash in seen_content:
                continue
            parent = chunk.parent_id or chunk.chunk_id
            if counts.get(parent, 0) >= self.max_per_parent:
                continue
            selected.append((chunk, score))
            seen_content.add(chunk.content_hash)
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
        return max(0.0, min(1.0, value / (1.0 + value)))


class RetrievalPipeline:
    """Fuse, diversify, rerank, and calibrate recalled chunks."""

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
        fused = self.fusion.fuse([dense, sparse]) if sparse is not None else list(dense)
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
                query, f"{chunk.paper_title}\n{chunk.section}\n{chunk.content}"
            )
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    confidence=max(
                        ScoreCalibrator.confidence(float(score), backend), anchor_score
                    ),
                    backend=backend,
                    dense_score=dense_scores.get(chunk.chunk_id),
                    sparse_score=sparse_scores.get(chunk.chunk_id),
                    fusion_score=fusion_scores.get(chunk.chunk_id),
                    rerank_score=float(score) if use_reranker else None,
                    lexical_anchor_score=anchor_score,
                )
            )
        return results


class Retriever:
    """Filterable Dense + BM25 + RRF + diversification + reranking search."""

    _shared_lock = threading.Lock()
    _shared_embedders: dict[tuple[str, bool, int], Embedder] = {}
    _shared_rerankers: dict[tuple[str, bool, int], Reranker] = {}

    def __init__(
        self,
        settings: Settings,
        index_dir: str,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = settings
        self.analyzer = QueryAnalyzer(settings.retrieval.cjk_ngram_size)
        self.embedder = embedder or self._shared_embedder(settings)
        self.store = VectorStore(index_dir=index_dir)
        self.reranker = self._shared_reranker(settings)
        self.pipeline = RetrievalPipeline(
            self.reranker,
            settings.index.rrf_k,
            settings.index.max_chunks_per_parent,
        )
        self._bm25 = None
        self._bm25_generation: str | None = None
        self.last_trace: dict[str, float | str | int] = {}
        self._search_lock = threading.Lock()

    @classmethod
    def _shared_embedder(cls, settings: Settings) -> Embedder:
        key = (
            settings.embedding.model_name,
            settings.embedding.use_sentence_transformers,
            settings.embedding.fallback_dimension,
        )
        with cls._shared_lock:
            embedder = cls._shared_embedders.get(key)
            if embedder is None:
                embedder = Embedder(*key)
                cls._shared_embedders[key] = embedder
            return embedder

    @classmethod
    def _shared_reranker(cls, settings: Settings) -> Reranker:
        key = (
            settings.rerank.model_name,
            settings.rerank.use_cross_encoder,
            settings.retrieval.cjk_ngram_size,
        )
        with cls._shared_lock:
            reranker = cls._shared_rerankers.get(key)
            if reranker is None:
                reranker = Reranker(
                    settings.rerank.model_name,
                    settings.rerank.use_cross_encoder,
                    analyzer=QueryAnalyzer(settings.retrieval.cjk_ngram_size),
                )
                cls._shared_rerankers[key] = reranker
            return reranker

    def prewarm(self) -> dict[str, str | int]:
        """Load immutable query models and this user's local index ahead of a request."""

        with self._search_lock:
            self._ensure_store()
            self.embedder.encode(["paper retrieval warmup"])
            self._bm25_index()
            return {
                "embedding": self.embedder.backend,
                "reranker": self.reranker.backend,
                "chunks": len(self.store.chunks),
            }

    def _ensure_store(self) -> None:
        if self.store.generation is None:
            self.store.load(self.embedder.signature)
        elif self.store.reload_if_changed(self.embedder.signature):
            self._bm25 = None
            self._bm25_generation = None

    def _bm25_index(self):
        self._ensure_store()
        if self._bm25_generation != self.store.generation:
            try:
                from rank_bm25 import BM25Okapi

                corpus = [self.analyzer.tokens(chunk.content) for chunk in self.store.chunks]
                self._bm25 = BM25Okapi(corpus) if corpus else None
            except ImportError:
                self._bm25 = None
            self._bm25_generation = self.store.generation
        return self._bm25

    def _allowed_indices(
        self,
        paper_ids: set[str] | None,
        modalities: set[str] | None,
        element_types: set[str] | None,
    ) -> list[int]:
        return [
            index
            for index, chunk in enumerate(self.store.chunks)
            if (paper_ids is None or chunk.paper_id in paper_ids)
            and (modalities is None or chunk.modality in modalities)
            and (element_types is None or chunk.element_type in element_types)
        ]

    def search(
        self,
        query: str,
        *,
        paper_ids: set[str] | None = None,
        modalities: set[str] | None = None,
        element_types: set[str] | None = None,
        mode: str = "hybrid",
        use_reranker: bool = True,
        top_k: int | None = None,
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        """Run one thread-safe retrieval over shared lazy model and index state."""

        with self._search_lock:
            return self._search(
                query,
                paper_ids=paper_ids,
                modalities=modalities,
                element_types=element_types,
                mode=mode,
                use_reranker=use_reranker,
                top_k=top_k,
                top_n=top_n,
            )

    def _search(
        self,
        query: str,
        *,
        paper_ids: set[str] | None = None,
        modalities: set[str] | None = None,
        element_types: set[str] | None = None,
        mode: str = "hybrid",
        use_reranker: bool = True,
        top_k: int | None = None,
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        started = time.perf_counter()
        self._ensure_store()
        loaded_at = time.perf_counter()
        recall_limit = top_k or self.settings.index.top_k_recall
        output_limit = top_n or self.settings.index.top_n_rerank
        allowed = self._allowed_indices(paper_ids, modalities, element_types)
        query_vector = self.embedder.encode([query])
        embedded_at = time.perf_counter()
        dense = (
            self.store.search(query_vector, recall_limit, allowed)
            if mode in {"dense", "hybrid"}
            else []
        )
        dense_at = time.perf_counter()

        sparse: list[tuple[Chunk, float]] | None = None
        if mode in {"sparse", "hybrid"}:
            bm25 = self._bm25_index()
            if bm25 is not None:
                scores = bm25.get_scores(self.analyzer.tokens(query))
                indices = np.asarray(allowed, dtype=np.int64)
                order: list[int] | np.ndarray = (
                    indices[np.argsort(-scores[indices])[:recall_limit]]
                    if len(indices)
                    else []
                )
                sparse = [
                    (self.store.chunks[int(index)], float(scores[int(index)]))
                    for index in order
                ]
            else:
                sparse = []
        if mode == "sparse":
            dense, sparse = sparse or [], None
        if mode not in {"dense", "sparse", "hybrid"}:
            raise ValueError(f"unknown retrieval mode: {mode}")
        recalled_at = time.perf_counter()
        results = self.pipeline.run(
            dense,
            sparse,
            query,
            recall_limit,
            output_limit,
            use_reranker=use_reranker,
        )
        finished = time.perf_counter()
        self.last_trace = {
            "mode": mode,
            "store_ms": round((loaded_at - started) * 1000, 2),
            "embed_ms": round((embedded_at - loaded_at) * 1000, 2),
            "dense_ms": round((dense_at - embedded_at) * 1000, 2),
            "sparse_ms": round((recalled_at - dense_at) * 1000, 2),
            "fusion_rerank_ms": round((finished - recalled_at) * 1000, 2),
            "total_ms": round((finished - started) * 1000, 2),
            "results": len(results),
        }
        return results


def prewarm_shared_models(settings: Settings) -> dict[str, str]:
    """Warm locally installed immutable models without triggering downloads."""

    embedding_path = resolve_model_path(settings.embedding.model_name)
    if settings.embedding.use_sentence_transformers and not Path(embedding_path).is_dir():
        return {"status": "skipped", "reason": "embedding_model_not_local"}
    embedder = Retriever._shared_embedder(settings)
    embedder.encode(["paper retrieval warmup"])
    reranker = Retriever._shared_reranker(settings)
    if settings.rerank.use_cross_encoder:
        reranker_path = resolve_model_path(settings.rerank.model_name)
        if Path(reranker_path).is_dir():
            _ = reranker.backend
    return {
        "status": "ready",
        "embedding": embedder.backend,
        "reranker": reranker.backend,
    }
