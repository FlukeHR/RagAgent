from __future__ import annotations

from typing import Sequence

from config.settings import resolve_model_path
from retrieval.analyzer import QueryAnalyzer
from retrieval.chunker import Chunk


class Reranker:
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

    def _ensure_model(self) -> None:
        """Load reranker weights only when reranking is actually requested."""

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
