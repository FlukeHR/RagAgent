from __future__ import annotations

import time

import numpy as np

from config.settings import Settings
from retrieval.analyzer import QueryAnalyzer
from retrieval.chunker import Chunk
from retrieval.embedder import Embedder
from retrieval.pipeline import RetrievalPipeline, RetrievalResult
from retrieval.reranker import Reranker
from retrieval.vector_store import VectorStore


class Retriever:
    """Filterable Dense + BM25 + RRF + diversification + reranking pipeline."""

    def __init__(
        self,
        settings: Settings,
        index_dir: str,
        embedder: Embedder | None = None,
    ) -> None:
        """Create a retriever, optionally reusing an already-loaded embedder."""

        self.settings = settings
        self.analyzer = QueryAnalyzer(settings.retrieval.cjk_ngram_size)
        self.embedder = embedder or Embedder(
            settings.embedding.model_name,
            settings.embedding.use_sentence_transformers,
            settings.embedding.fallback_dimension,
        )
        self.store = VectorStore(index_dir=index_dir)
        self.reranker = Reranker(
            settings.rerank.model_name,
            settings.rerank.use_cross_encoder,
            analyzer=self.analyzer,
        )
        self.pipeline = RetrievalPipeline(
            self.reranker,
            settings.index.rrf_k,
            settings.index.max_chunks_per_parent,
        )
        self._bm25 = None
        self._bm25_generation: str | None = None
        self.last_trace: dict[str, float | str | int] = {}

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

    @staticmethod
    def _rrf_fuse(
        ranked_lists: list[list[tuple[Chunk, float]]],
        k: int = 60,
    ) -> list[tuple[Chunk, float]]:
        """Compatibility helper retained for external callers."""

        from retrieval.pipeline import ReciprocalRankFusion

        return ReciprocalRankFusion(k).fuse(ranked_lists)

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
