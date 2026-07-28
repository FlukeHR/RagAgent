from __future__ import annotations

from dataclasses import replace

from config.settings import BASE_DIR, Settings
from retrieval.retriever import Retriever
from retrieval.repository import normalize_arxiv_id
from services.library_service import PaperLibraryService
from tools.base import EvidenceSource, ToolPolicy, ToolResult


class ArxivIngestTool:
    """Ingest selected arXiv PDFs and retrieve only from those papers."""

    name = "ingest_arxiv_papers"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.timeout_seconds = settings.arxiv.ingest_timeout_seconds
        self.policy = ToolPolicy(
            timeout_seconds=self.timeout_seconds,
            max_retries=0,
            side_effects="write",
            idempotent=True,
            isolate_process=True,
        )
        self.library = PaperLibraryService(settings)
        self.index_dir = BASE_DIR / settings.index.index_root

    def schema(self) -> dict:
        return {
            "name": "ingest_arxiv_papers",
            "description": (
                "高延迟全文工具：下载 search_arxiv 选定的最相关一篇论文，增量入库后"
                "只在该论文中检索。仅在摘要不足以回答正文细节时使用。"
            ),
            "input_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "arxiv_ids": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "minLength": 5,
                            "maxLength": 80,
                            "pattern": (
                                r"^(?:\d{4}\.\d{4,5}|"
                                r"[A-Za-z][A-Za-z0-9._-]*/\d{7})(?:v\d+)?$"
                            ),
                        },
                        "minItems": 1,
                        "maxItems": self.settings.arxiv.max_ingest_papers,
                        "uniqueItems": True,
                    },
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1000,
                    },
                },
                "required": ["arxiv_ids", "query"],
            },
        }

    def run(self, arxiv_ids: list[str], query: str) -> ToolResult:
        normalized = [normalize_arxiv_id(value) for value in arxiv_ids]
        report = self.library.ingest_arxiv(normalized)
        results = []
        if report.storage_ids and report.indexed_chunks:
            retrieval_settings = self.settings
            if not self.settings.arxiv.use_cross_encoder_after_ingest:
                retrieval_settings = replace(
                    self.settings,
                    rerank=replace(
                        self.settings.rerank,
                        use_cross_encoder=False,
                    ),
                )
            retriever = Retriever(
                retrieval_settings,
                str(self.index_dir),
                embedder=self.library.last_embedder,
            )
            results = retriever.search(query, paper_ids=set(report.storage_ids))

        note_parts = []
        if report.downloaded:
            note_parts.append(f"新入库 {len(report.downloaded)} 篇")
        if report.reused:
            note_parts.append(f"复用 {len(report.reused)} 篇")
        if report.failed:
            note_parts.append(f"下载失败 {len(report.failed)} 篇（{', '.join(report.failed)}）")
        note = "；".join(note_parts) or "无可入库论文"
        if not results:
            return ToolResult(text=f"[入库情况] {note}。未在目标论文全文中检索到相关片段。")

        blocks: list[str] = []
        sources: list[EvidenceSource] = []
        for index, result in enumerate(results):
            chunk = result.chunk
            blocks.append(
                f"{{{{cite:{index}}}}}《{chunk.paper_title}》｜章节 {chunk.section}｜"
                f"论文ID {chunk.paper_id}\n上下文："
                f"{chunk.chunk_context or chunk.section}\n{chunk.content}"
            )
            sources.append(
                EvidenceSource.from_chunk(
                    chunk,
                    score=result.score,
                    snippet_chars=self.settings.harness.source_snippet_chars,
                    confidence=result.confidence,
                    score_backend=result.backend,
                    dense_score=result.dense_score,
                    sparse_score=result.sparse_score,
                    fusion_score=result.fusion_score,
                    lexical_anchor_score=result.lexical_anchor_score,
                )
            )
        return ToolResult(
            text=f"[入库情况] {note}。检索到以下目标论文全文片段：\n\n"
            + "\n\n".join(blocks),
            sources=sources,
            metadata={"ingest_report": report.__dict__},
        )
