from __future__ import annotations

from config.settings import Settings
from services.arxiv_service import ArxivSearchService
from tools.base import EvidenceSource, ToolResult


class ArxivTool:
    """Read-only arXiv metadata and abstract search."""

    name = "search_arxiv"
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.service = ArxivSearchService(settings)

    def run(self, query: str, max_results: int | None = None) -> ToolResult:
        configured = self.settings.arxiv.max_results
        limit = min(max_results or configured, configured)
        papers = self.service.search(query, max_results=limit)
        if not papers:
            return ToolResult(text="arXiv returned no matching papers.")
        blocks: list[str] = [
            "Read-only arXiv results. Creating an ingest proposal requires an "
            "explicit user action outside this agent run.",
        ]
        sources: list[EvidenceSource] = []
        for index, paper in enumerate(papers):
            published = paper.published.isoformat() if paper.published else None
            blocks.append(
                f"{{{{cite:{index}}}}} Untrusted arXiv abstract.\n"
                f"arxiv_id={paper.arxiv_id} | title={paper.title} | "
                f"authors={', '.join(paper.authors)} | published={published}\n"
                f"{paper.summary}"
            )
            sources.append(
                EvidenceSource(
                    paper_id=paper.arxiv_id,
                    paper_title=paper.title,
                    section="Abstract",
                    source=paper.entry_url,
                    published_at=published,
                    snippet=paper.summary[: self.settings.agent.source_snippet_chars],
                    element_type="abstract",
                    modality="text",
                    quality_rank=1,
                )
            )
        return ToolResult(
            text="\n\n".join(blocks),
            sources=sources,
            metadata={
                "query": query,
                "candidate_arxiv_ids": [paper.arxiv_id for paper in papers],
            },
        )
