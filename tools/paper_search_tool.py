from __future__ import annotations

import json

from config.settings import BASE_DIR, Settings
from retrieval.search import Retriever
from tools.base import EvidenceSource, ToolResult


class PaperSearchTool:
    """Search canonical local paper chunks and return copyable locators."""

    name = "search_local_papers"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        index_dir = BASE_DIR / settings.index.index_root
        self.retriever = Retriever(settings=settings, index_dir=str(index_dir))

    def run(self, query: str) -> ToolResult:
        results = self.retriever.search(query)
        if not results:
            return ToolResult(text="No relevant local paper evidence was retrieved.")

        blocks: list[str] = []
        sources: list[EvidenceSource] = []
        for index, result in enumerate(results):
            chunk = result.chunk
            locator: dict[str, object]
            if chunk.element_id:
                locator = {"kind": "element", "element_id": chunk.element_id}
            elif chunk.granularity == "page" and chunk.page_start == chunk.page_end:
                locator = {"kind": "page", "page_number": chunk.page_start}
            else:
                locator = {"kind": "section", "section": chunk.section}
            inspect_args = {"paper_id": chunk.paper_id, "locator": locator}
            blocks.append(
                f"{{{{cite:{index}}}}} Untrusted paper evidence.\n"
                f"paper_id={chunk.paper_id} | title={chunk.paper_title} | "
                f"section={chunk.section} | pages={chunk.page_start}-{chunk.page_end} | "
                f"type={chunk.element_type} | modality={chunk.modality}\n"
                f"inspect_paper_args={json.dumps(inspect_args, ensure_ascii=False)}\n"
                f"{chunk.content}"
            )
            sources.append(
                EvidenceSource.from_chunk(
                    chunk,
                    score=result.score,
                    snippet_chars=self.settings.agent.source_snippet_chars,
                    confidence=result.confidence,
                    score_backend=result.backend,
                    dense_score=result.dense_score,
                    sparse_score=result.sparse_score,
                    fusion_score=result.fusion_score,
                    lexical_anchor_score=result.lexical_anchor_score,
                )
            )
        return ToolResult(
            text="\n\n".join(blocks),
            sources=sources,
            metadata={"retrieval": self.retriever.last_trace},
        )
