from __future__ import annotations

import base64

from config.settings import BASE_DIR, Settings
from retrieval.pdf_service import PDFService
from retrieval.repository import PaperRepository
from tools.base import EvidenceSource, ToolPolicy, ToolResult


class PDFPageTool:
    """Read one PDF page and optionally return a bounded rendered image."""

    name = "read_pdf_page"
    policy = ToolPolicy(side_effects="read", idempotent=True)

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.data_dir = BASE_DIR / settings.project.data_root
        self.repository = PaperRepository(self.data_dir)
        self.pdf = PDFService()

    def schema(self) -> dict:
        return {
            "name": "read_pdf_page",
            "description": (
                "按需读取本地 PDF 的一个指定页；必要时返回尺寸受限的页图。"
                "不要用它一次性读取整篇 PDF。"
            ),
            "input_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 160,
                    },
                    "page_number": {"type": "integer", "minimum": 1},
                    "include_image": {"type": "boolean"},
                    "max_side": {
                        "type": "integer",
                        "minimum": 256,
                        "maximum": self.settings.harness.pdf_page_max_side,
                    },
                    "max_image_base64_chars": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 50000,
                    },
                },
                "required": ["paper_id", "page_number"],
            },
        }

    def run(
        self,
        paper_id: str,
        page_number: int,
        include_image: bool = False,
        max_side: int = 1200,
        max_image_base64_chars: int = 12000,
    ) -> ToolResult:
        pdf_path = self.repository.resolve(paper_id, (".pdf",))
        if pdf_path is None:
            return ToolResult(text=f"未找到 paper_id={paper_id} 对应的本地 PDF。")

        total_pages = self.pdf.page_count(pdf_path)
        if page_number > total_pages:
            return ToolResult(text=f"页码越界：{paper_id} 共有 {total_pages} 页。")
        page = self.pdf.read_page(pdf_path, page_number)
        image_note = ""
        image_meta: dict = {}
        if include_image:
            max_side = min(max_side, self.settings.harness.pdf_page_max_side)
            image = self.pdf.render_page(pdf_path, page_number, max_side=max_side)
            image_b64 = base64.b64encode(image.data).decode("ascii")
            image_meta = {
                "image_mime_type": image.mime_type,
                "image_width": image.width,
                "image_height": image.height,
                "image_base64": image_b64,
            }
            limit = min(
                max_image_base64_chars,
                self.settings.harness.image_base64_chars,
            )
            if limit:
                shown = image_b64[:limit]
                suffix = "\n[image_base64 truncated]" if len(shown) < len(image_b64) else ""
                image_note = (
                    f"\n\nimage_mime_type: {image.mime_type}\nimage_base64:\n{shown}{suffix}"
                )
            else:
                image_note = (
                    f"\n\n已渲染压缩页图：{image.mime_type}, "
                    f"{image.width}x{image.height}；base64 仅保存在 source metadata。"
                )

        text = page.indexed_text or (
            "（该页未抽取到文本，可能是扫描页；可结合 include_image=true 使用页图。）"
        )
        modality = (
            "image"
            if include_image
            else "ocr"
            if page.ocr_text and not page.text.strip()
            else "vlm"
            if page.vlm_summary and not page.text.strip()
            else "text"
        )
        body_limit = self.settings.harness.tool_result_max_chars
        source = EvidenceSource(
            chunk_id=f"{paper_id}::page::{page_number}",
            paper_id=paper_id,
            paper_title=paper_id,
            section=f"Page {page_number}",
            source=str(pdf_path),
            page_start=page_number,
            page_end=page_number,
            element_type="page_image" if include_image else "page",
            modality=modality,
            chunk_context=f"PDF page 《{paper_id}》 page {page_number}",
            heading_path=f"Page {page_number}",
            snippet=text[: self.settings.harness.source_snippet_chars],
            quality_rank=3,
            **image_meta,
        )
        return ToolResult(
            text=(
                f"{{{{cite:0}}}} PDF page｜paper_id={paper_id}｜"
                f"page={page_number}/{total_pages}｜scanned_like={page.is_scanned_like}｜"
                f"modality={modality}\n{text[:body_limit]}{image_note}"
            ),
            sources=[source],
        )
