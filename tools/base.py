from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolPolicy:
    """Execution policy enforced by the harness, not by the model."""

    timeout_seconds: float | None = None
    max_retries: int | None = None
    side_effects: str = "read"  # read | network | write
    idempotent: bool = True
    isolate_process: bool = False


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
    chunk_context: str | None = None
    heading_path: str | None = None
    score: float | None = None
    confidence: float | None = None
    score_backend: str | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    snippet: str | None = None
    image_mime_type: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    image_base64: str | None = None
    published_at: str | None = None
    support_status: str | None = None
    quality_rank: int = 0
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


@runtime_checkable
class Tool(Protocol):
    name: str
    policy: ToolPolicy

    def schema(self) -> dict[str, Any]: ...

    def run(self, **kwargs: Any) -> ToolResult: ...


@dataclass(frozen=True)
class ToolSpec:
    name: str
    schema: dict[str, Any]
    policy: ToolPolicy
    handler: Any

    @classmethod
    def from_tool(cls, tool: Any) -> "ToolSpec":
        return cls(tool.name, tool.schema(), tool.policy, tool)


class ToolRegistry:
    """Single source of truth for tool instances, schemas and execution policies."""

    def __init__(self, tools: list[Any] | None = None) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Any) -> None:
        spec = ToolSpec.from_tool(tool)
        if spec.name in self._specs:
            raise ValueError(f"duplicate tool: {spec.name}")
        if spec.schema.get("name") != spec.name:
            raise ValueError(f"tool/schema name mismatch: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def schemas(self, allowed: set[str] | None = None) -> list[dict[str, Any]]:
        return [
            spec.schema
            for name, spec in self._specs.items()
            if allowed is None or name in allowed
        ]

    def names(self) -> set[str]:
        return set(self._specs)

    def profile_schemas(self, profile: str) -> list[dict[str, Any]]:
        """Progressively expose schemas without turning tools into opaque skills."""

        profiles = {
            "local": {"search_local_papers", "read_paper_section"},
            "pdf": {"read_pdf_page", "read_pdf_region", "search_pdf_images"},
            "arxiv": {"search_arxiv", "ingest_arxiv_papers"},
            "all": self.names(),
        }
        if profile not in profiles:
            raise ValueError(f"unknown tool profile: {profile}")
        return self.schemas(profiles[profile])
