from __future__ import annotations

import base64
from pathlib import Path

from config.settings import BASE_DIR, Settings
from retrieval.loader import PaperLoader
from tools.base import ToolResult


class PDFPageTool:
    """按需读取本地 PDF 的单页文本，并可渲染压缩页图供 OCR/VLM 后续使用。"""

    name = "read_pdf_page"

    def __init__(self, settings: Settings, collection: str) -> None:
        self.collection_dir = BASE_DIR / settings.project.data_root / collection

    @staticmethod
    def schema() -> dict:
        return {
            "name": "read_pdf_page",
            "description": (
                "按需读取本地 PDF 的指定页。返回该页文本、页码、扫描页检测结果；"
                "include_image=true 时会把该页渲染为尺寸受限图片，供扫描件/OCR/VLM 路径使用。"
                "不要用它一次性读取整篇 PDF。"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "论文标识（search_local_papers 返回的 paper_id）",
                    },
                    "page_number": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "1-based PDF 页码",
                    },
                    "include_image": {
                        "type": "boolean",
                        "description": "是否返回该页压缩图片的 base64（默认 false）",
                    },
                    "max_side": {
                        "type": "integer",
                        "minimum": 256,
                        "maximum": 2400,
                        "description": "渲染图片最长边像素上限，默认 1200",
                    },
                    "max_image_base64_chars": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 50000,
                        "description": "返回到 tool_result 文本里的 base64 字符上限，默认 12000；0 表示只放 metadata",
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
        _id_base: int = 0,
    ) -> ToolResult:
        pdf = self._pdf_path(paper_id)
        if pdf is None:
            return ToolResult(text=f"未找到 paper_id={paper_id} 对应的本地 PDF。", sources=[])

        pages = PaperLoader._read_pdf_pages(pdf)
        if page_number < 1 or page_number > len(pages):
            return ToolResult(
                text=f"页码越界：{paper_id} 共有 {len(pages)} 页，请请求 1 到 {len(pages)} 之间的页码。",
                sources=[],
            )

        page = pages[page_number - 1]
        sid = f"S{_id_base + 1}"
        image_note = ""
        image_meta: dict = {}
        image_b64 = ""
        if include_image:
            image = PaperLoader.render_pdf_page(pdf, page_number=page_number, max_side=max_side)
            image_b64 = base64.b64encode(image.data).decode("ascii")
            image_meta = {
                "image_mime_type": image.mime_type,
                "image_width": image.width,
                "image_height": image.height,
                "image_base64": image_b64,
            }
            if max_image_base64_chars > 0:
                shown = image_b64[:max_image_base64_chars]
                suffix = "\n[image_base64 truncated]" if len(image_b64) > len(shown) else ""
                image_note = f"\n\nimage_mime_type: {image.mime_type}\nimage_base64:\n{shown}{suffix}"
            else:
                image_note = (
                    f"\n\n已渲染压缩页图：{image.mime_type}, "
                    f"{image.width}x{image.height}；base64 仅保存在 source metadata。"
                )

        text_parts: list[str] = []
        if page.text.strip():
            text_parts.append(page.text.strip())
        if page.ocr_text and page.ocr_text.strip():
            text_parts.append(f"[OCR]\n{page.ocr_text.strip()}")
        if page.vlm_summary and page.vlm_summary.strip():
            text_parts.append(f"[VLM]\n{page.vlm_summary.strip()}")
        text = "\n\n".join(text_parts) or "(该页未抽取到文本，可能是扫描页；可结合 include_image=true 走 OCR/VLM。)"
        modality = "image" if include_image else (
            "ocr" if page.ocr_text and not page.text.strip() else
            "vlm" if page.vlm_summary and not page.text.strip() else
            "text"
        )
        result_text = (
            f"[{sid}] PDF page｜paper_id={paper_id}｜page={page_number}/{len(pages)}｜"
            f"scanned_like={page.is_scanned_like}｜modality={modality}\n{text[:4000]}{image_note}"
        )
        source = {
            "id": sid,
            "chunk_id": f"{paper_id}::page::{page_number}",
            "paper_id": paper_id,
            "paper_title": paper_id,
            "section": f"Page {page_number}",
            "source": str(pdf),
            "page_start": page_number,
            "page_end": page_number,
            "element_type": "page_image" if include_image else "page",
            "modality": modality,
            "bbox": None,
            "chunk_context": f"PDF page 《{paper_id}》 page {page_number}",
            "heading_path": f"Page {page_number}",
            "score": None,
            "snippet": text[:600],
            **image_meta,
        }
        return ToolResult(text=result_text, sources=[source])

    def _pdf_path(self, paper_id: str) -> Path | None:
        candidate = self.collection_dir / f"{paper_id}.pdf"
        return candidate if candidate.exists() else None
