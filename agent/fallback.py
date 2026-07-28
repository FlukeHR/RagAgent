from __future__ import annotations

from config.settings import Settings
from retrieval.retriever import Retriever


class FallbackRAG:
    """Local retrieval portion of the non-agentic fallback path."""

    def __init__(self, settings: Settings, retriever: Retriever) -> None:
        self.settings = settings
        self.retriever = retriever

    def retrieve(self, query: str):
        return self.retriever.search(query)

    def sources(self, results) -> list[dict]:
        limit = self.settings.harness.source_snippet_chars
        return [
            {
                "id": f"S{index}",
                "chunk_id": result.chunk.chunk_id,
                "paper_id": result.chunk.paper_id,
                "paper_title": result.chunk.paper_title,
                "section": result.chunk.section,
                "source": result.chunk.source,
                "page_start": result.chunk.page_start,
                "page_end": result.chunk.page_end,
                "element_type": result.chunk.element_type,
                "modality": result.chunk.modality,
                "bbox": result.chunk.bbox,
                "chunk_context": result.chunk.chunk_context,
                "heading_path": result.chunk.heading_path,
                "score": round(float(result.score), 4),
                "confidence": round(float(result.confidence), 4),
                "score_backend": result.backend,
                "dense_score": result.dense_score,
                "sparse_score": result.sparse_score,
                "fusion_score": result.fusion_score,
                "lexical_anchor_score": result.lexical_anchor_score,
                "snippet": result.chunk.content[:limit],
            }
            for index, result in enumerate(results, start=1)
        ]
