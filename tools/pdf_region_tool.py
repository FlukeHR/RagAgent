from __future__ import annotations

import base64

from config.settings import BASE_DIR, Settings
from retrieval.pdf_service import PDFService
from retrieval.repository import PaperRepository
from tools.base import EvidenceSource, ToolPolicy, ToolResult


class PDFRegionTool:
    """Read or render one bounded PDF region."""

    name = "read_pdf_region"
    policy = ToolPolicy(side_effects="read", idempotent=True)

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repository = PaperRepository(BASE_DIR / settings.project.data_root)
        self.pdf = PDFService()

    def schema(self) -> dict:
        return {
            "name": "read_pdf_region",
            "description": (
                "按 bbox 读取或渲染一个 PDF 页内区域，用于核对表格、图表和公式。"
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
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "include_image": {"type": "boolean"},
                    "max_side": {
                        "type": "integer",
                        "minimum": 128,
                        "maximum": self.settings.harness.pdf_region_max_side,
                    },
                    "max_image_base64_chars": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 30000,
                    },
                },
                "required": ["paper_id", "page_number", "bbox"],
            },
        }

    def run(
        self,
        paper_id: str,
        page_number: int,
        bbox: list[float],
        include_image: bool = True,
        max_side: int = 800,
        max_image_base64_chars: int = 8000,
    ) -> ToolResult:
        pdf_path = self.repository.resolve(paper_id, (".pdf",))
        if pdf_path is None:
            return ToolResult(text=f"未找到 paper_id={paper_id} 对应的本地 PDF。")
        box = (
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3]),
        )
        text = self.pdf.read_region(pdf_path, page_number, box)
        image_note = ""
        image_meta: dict = {}
        if include_image:
            max_side = min(max_side, self.settings.harness.pdf_region_max_side)
            image = self.pdf.render_page(
                pdf_path,
                page_number,
                max_side=max_side,
                bbox=box,
            )
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
            shown = image_b64[:limit] if limit else ""
            if shown:
                suffix = "\n[image_base64 truncated]" if len(shown) < len(image_b64) else ""
                image_note = (
                    f"\n\nimage_mime_type: {image.mime_type}\nimage_base64:\n{shown}{suffix}"
                )
            else:
                image_note = (
                    f"\n\n已渲染区域图片：{image.mime_type}, "
                    f"{image.width}x{image.height}。"
                )

        body = text or "（该区域未抽取到文本，可查看区域图片或 OCR/VLM sidecar。）"
        source = EvidenceSource(
            chunk_id=f"{paper_id}::region::{page_number}:{','.join(str(x) for x in box)}",
            paper_id=paper_id,
            paper_title=paper_id,
            section=f"Page {page_number} region",
            source=str(pdf_path),
            page_start=page_number,
            page_end=page_number,
            element_type="region_image" if include_image else "region",
            modality="image" if include_image else "text",
            bbox=box,
            chunk_context=f"PDF region 《{paper_id}》 page {page_number} bbox {box}",
            heading_path=f"Page {page_number} region",
            snippet=body[: self.settings.harness.source_snippet_chars],
            quality_rank=4,
            **image_meta,
        )
        return ToolResult(
            text=(
                f"{{{{cite:0}}}} PDF region｜paper_id={paper_id}｜"
                f"page={page_number}｜bbox={box}\n"
                f"{body[: self.settings.harness.tool_result_max_chars]}{image_note}"
            ),
            sources=[source],
        )
