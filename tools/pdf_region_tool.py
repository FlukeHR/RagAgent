from __future__ import annotations

import base64
from pathlib import Path

from config.settings import BASE_DIR, Settings
from retrieval.loader import PaperLoader
from tools.base import ToolResult


class PDFRegionTool:
    """Read/render a bounded region from one local PDF page."""

    name = "read_pdf_region"

    def __init__(self, settings: Settings, collection: str) -> None:
        self.collection_dir = BASE_DIR / settings.project.data_root / collection

    @staticmethod
    def schema() -> dict:
        return {
            "name": "read_pdf_region",
            "description": (
                "按 bbox 读取/渲染本地 PDF 的页内区域。用于核对表格、图表、公式或 bbox 高亮。"
                "bbox 为 PDF 坐标 [x0,y0,x1,y1]，一次只读一个小区域。"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "论文标识"},
                    "page_number": {"type": "integer", "minimum": 1, "description": "1-based PDF 页码"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "PDF 坐标 [x0,y0,x1,y1]",
                    },
                    "include_image": {"type": "boolean", "description": "是否返回区域图片 base64"},
                    "max_side": {"type": "integer", "minimum": 128, "maximum": 1600, "description": "最长边像素上限"},
                    "max_image_base64_chars": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 30000,
                        "description": "tool_result 文本中的 base64 字符上限",
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
        _id_base: int = 0,
    ) -> ToolResult:
        pdf = self._pdf_path(paper_id)
        if pdf is None:
            return ToolResult(text=f"未找到 paper_id={paper_id} 对应的本地 PDF。", sources=[])
        box = tuple(float(x) for x in bbox)
        text = self._region_text(pdf, page_number, box)
        sid = f"S{_id_base + 1}"

        image_note = ""
        image_meta: dict = {}
        if include_image:
            image = PaperLoader.render_pdf_page(pdf, page_number, max_side=max_side, bbox=box)
            image_b64 = base64.b64encode(image.data).decode("ascii")
            image_meta = {
                "image_mime_type": image.mime_type,
                "image_width": image.width,
                "image_height": image.height,
                "image_base64": image_b64,
            }
            shown = image_b64[:max_image_base64_chars] if max_image_base64_chars > 0 else ""
            if shown:
                suffix = "\n[image_base64 truncated]" if len(image_b64) > len(shown) else ""
                image_note = f"\n\nimage_mime_type: {image.mime_type}\nimage_base64:\n{shown}{suffix}"
            else:
                image_note = f"\n\n已渲染区域图片：{image.mime_type}, {image.width}x{image.height}。"

        body = text or "(该 bbox 区域未抽取到文本，可查看区域图片或 OCR/VLM sidecar。)"
        source = {
            "id": sid,
            "chunk_id": f"{paper_id}::region::{page_number}:{','.join(str(x) for x in box)}",
            "paper_id": paper_id,
            "paper_title": paper_id,
            "section": f"Page {page_number} region",
            "source": str(pdf),
            "page_start": page_number,
            "page_end": page_number,
            "element_type": "region_image" if include_image else "region",
            "modality": "image" if include_image else "text",
            "bbox": box,
            "chunk_context": f"PDF region 《{paper_id}》 page {page_number} bbox {box}",
            "heading_path": f"Page {page_number} region",
            "score": None,
            "snippet": body[:600],
            **image_meta,
        }
        return ToolResult(
            text=f"[{sid}] PDF region｜paper_id={paper_id}｜page={page_number}｜bbox={box}\n{body[:3000]}{image_note}",
            sources=[source],
        )

    def _pdf_path(self, paper_id: str) -> Path | None:
        candidate = self.collection_dir / f"{paper_id}.pdf"
        return candidate if candidate.exists() else None

    @staticmethod
    def _region_text(pdf: Path, page_number: int, bbox: tuple[float, float, float, float]) -> str:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("解析 PDF 需要 PyMuPDF，请先安装：pip install pymupdf") from exc
        with fitz.open(str(pdf)) as doc:
            if page_number < 1 or page_number > len(doc):
                raise ValueError(f"page_number out of range: {page_number}")
            page = doc[page_number - 1]
            rect = fitz.Rect(*bbox) & page.rect
            if rect.is_empty:
                raise ValueError(f"bbox outside page: {bbox}")
            return page.get_text("text", clip=rect).strip()
