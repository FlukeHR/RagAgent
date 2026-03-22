from __future__ import annotations

from typing import Sequence

from retrieval.chunker import CodeChunk


class Reranker:
    def __init__(self, model_name: str, use_cross_encoder: bool = False) -> None:
        self.model_name = model_name
        self.use_cross_encoder = use_cross_encoder
        self._cross_encoder = None

        if use_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder

                self._cross_encoder = CrossEncoder(model_name)
            except Exception:
                self._cross_encoder = None

    @staticmethod
    def _token_overlap_score(query: str, text: str) -> float:
        q_tokens = set(query.lower().split())
        t_tokens = set(text.lower().split())
        if not q_tokens:
            return 0.0
        return len(q_tokens & t_tokens) / len(q_tokens)

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[CodeChunk, float]],
        top_n: int,
    ) -> list[tuple[CodeChunk, float]]:
        if not candidates:
            return []

        if self._cross_encoder is not None:
            pairs = [[query, item[0].content] for item in candidates]
            ce_scores = self._cross_encoder.predict(pairs)
            ranked = sorted(
                zip((item[0] for item in candidates), ce_scores),
                key=lambda x: x[1],
                reverse=True,
            )
            return [(chunk, float(score)) for chunk, score in ranked[:top_n]]

        ranked = sorted(
            ((chunk, self._token_overlap_score(query, chunk.content) + base_score) for chunk, base_score in candidates),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_n]
