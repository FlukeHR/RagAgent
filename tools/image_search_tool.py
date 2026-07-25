from __future__ import annotations

import base64

from config.settings import BASE_DIR, Settings
from retrieval.image_index import PageImageIndex
from retrieval.pdf_service import PDFService
from retrieval.repository import PaperRepository
from tools.base import EvidenceSource, ToolPolicy, ToolResult


class ImageSearchTool:
    """Search a precomputed local page-image index."""

    name = "search_pdf_images"
    policy = ToolPolicy(side_effects="read", idempotent=True)

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repository = PaperRepository(BASE_DIR / settings.project.data_root)
        self.index = PageImageIndex(BASE_DIR / settings.index.index_root)
        self.pdf = PDFService()

    def schema(self) -> dict:
        return {
            "name": "search_pdf_images",
            "description": (
                "用图片或本地 PDF 页/区域查询预计算的页面图像索引。"
            ),
            "input_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "image_base64": {
                        "type": "string",
                        "minLength": 4,
                        "maxLength": self.settings.image_search.max_query_base64_chars,
                    },
                    "paper_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 160,
                    },
                    "page_number": {"type": "integer", "minimum": 1},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "anyOf": [
                    {"required": ["image_base64"]},
                    {"required": ["paper_id", "page_number"]},
                ],
            },
        }

    def run(
        self,
        image_base64: str | None = None,
        paper_id: str | None = None,
        page_number: int | None = None,
        bbox: list[float] | None = None,
        top_k: int = 5,
    ) -> ToolResult:
        if not self.settings.image_search.enabled:
            return ToolResult(text="图像检索未启用。")
        query = self._query_bytes(image_base64, paper_id, page_number, bbox)
        if not query:
            return ToolResult(text="未提供有效查询图片或 paper_id/page_number。")
        try:
            ranked = self.index.search(query, top_k)
        except FileNotFoundError:
            return ToolResult(text="页面图像索引不存在，请先运行 indexing/build_index.py。")
        if not ranked:
            return ToolResult(text="页面图像索引为空。")

        blocks: list[str] = []
        sources: list[EvidenceSource] = []
        for index, (metadata, score) in enumerate(ranked):
            source_pdf = self.repository.resolve(metadata["paper_id"], (".pdf",))
            if source_pdf is None:
                continue
            page = int(metadata["page_number"])
            image = self.pdf.render_page(
                source_pdf,
                page,
                max_side=self.settings.image_search.max_side,
            )
            image_b64 = base64.b64encode(image.data).decode("ascii")
            blocks.append(
                f"{{{{cite:{index}}}}} PDF image match｜paper_id={metadata['paper_id']}｜"
                f"page={page}｜score={score:.4f}"
            )
            sources.append(
                EvidenceSource(
                    chunk_id=f"{metadata['paper_id']}::page_image::{page}",
                    paper_id=metadata["paper_id"],
                    paper_title=metadata["paper_id"],
                    section=f"Page {page} image",
                    source=str(source_pdf),
                    page_start=page,
                    page_end=page,
                    element_type="page_image",
                    modality="image",
                    chunk_context=(
                        f"Page image for 《{metadata['paper_id']}》 page {page}"
                    ),
                    heading_path=f"Page {page} image",
                    score=round(score, 4),
                    snippet=f"Image similarity match on page {page}.",
                    image_base64=image_b64,
                    image_mime_type=image.mime_type,
                    image_width=image.width,
                    image_height=image.height,
                    quality_rank=1,
                )
            )
        return ToolResult(text="\n".join(blocks), sources=sources)

    def _query_bytes(
        self,
        image_base64: str | None,
        paper_id: str | None,
        page_number: int | None,
        bbox: list[float] | None,
    ) -> bytes:
        if image_base64:
            if len(image_base64) > self.settings.image_search.max_query_base64_chars:
                return b""
            try:
                return base64.b64decode(image_base64, validate=True)
            except ValueError:
                return b""
        if not paper_id or not page_number:
            return b""
        pdf_path = self.repository.resolve(paper_id, (".pdf",))
        if pdf_path is None:
            return b""
        box = (
            (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            )
            if bbox
            else None
        )
        return self.pdf.render_page(
            pdf_path,
            page_number,
            max_side=self.settings.image_search.max_side,
            bbox=box,
        ).data
