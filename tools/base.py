from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvidenceSource:
    """Canonical evidence returned by every tool before citation IDs are assigned."""

    paper_id: str
    paper_title: str
    section: str
    source: str
    chunk_id: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    element_type: str | None = None
    modality: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    bbox_space: str | None = None
    element_id: str | None = None
    parser_metadata: dict[str, Any] = field(default_factory=dict)
    chunk_context: str | None = None
    heading_path: str | None = None
    score: float | None = None
    confidence: float | None = None
    score_backend: str | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    lexical_anchor_score: float = 0.0
    snippet: str | None = None
    published_at: str | None = None
    support_status: str | None = None
    quality_rank: int = 0
    origin_tools: list[str] = field(default_factory=list)
    citation_id: str | None = None

    @property
    def dedup_key(self) -> str:
        raw = "|".join(
            [
                self.modality or "",
                " ".join((self.snippet or "").lower().split()),
            ]
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["id"] = data.pop("citation_id")
        return data

    @classmethod
    def from_chunk(
        cls,
        chunk: Any,
        *,
        score: float | None,
        snippet_chars: int = 600,
        confidence: float | None = None,
        score_backend: str | None = None,
        dense_score: float | None = None,
        sparse_score: float | None = None,
        fusion_score: float | None = None,
        lexical_anchor_score: float = 0.0,
    ) -> "EvidenceSource":
        """Build canonical evidence from a retrieval chunk without losing metadata."""

        return cls(
            chunk_id=chunk.chunk_id,
            paper_id=chunk.paper_id,
            paper_title=chunk.paper_title,
            section=chunk.section,
            source=chunk.source,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            element_type=chunk.element_type,
            modality=chunk.modality,
            bbox=chunk.bbox,
            bbox_space=getattr(chunk, "bbox_space", None),
            element_id=getattr(chunk, "element_id", None),
            parser_metadata=getattr(chunk, "parser_metadata", {}),
            chunk_context=chunk.chunk_context,
            heading_path=chunk.heading_path,
            score=round(float(score), 4) if score is not None else None,
            confidence=(
                round(float(confidence), 4) if confidence is not None else None
            ),
            score_backend=score_backend,
            dense_score=dense_score,
            sparse_score=sparse_score,
            fusion_score=fusion_score,
            lexical_anchor_score=round(float(lexical_anchor_score), 4),
            snippet=chunk.content[:snippet_chars],
            quality_rank={
                "element": 4,
                "page": 3,
                "section": 2,
            }.get(getattr(chunk, "granularity", ""), 1),
        )


@dataclass
class ToolResult:
    """Tool output with citation placeholders such as ``{{cite:0}}``."""

    text: str
    sources: list[EvidenceSource] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
