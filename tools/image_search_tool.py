from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

from config.settings import BASE_DIR, Settings
from retrieval.loader import PaperLoader
from tools.base import ToolResult


class ImageSearchTool:
    """Local page-image similarity search with a deterministic offline fallback."""

    name = "search_pdf_images"

    def __init__(self, settings: Settings, collection: str) -> None:
        self.settings = settings
        self.collection_dir = BASE_DIR / settings.project.data_root / collection

    @staticmethod
    def schema() -> dict:
        return {
            "name": "search_pdf_images",
            "description": (
                "用图片或本地 PDF 页/区域作为 query，在本地论文库中召回相似页面图像。"
                "当前默认使用离线图像签名；后续可替换为 CLIP/SigLIP 向量。"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "image_base64": {"type": "string", "description": "查询图片 base64"},
                    "paper_id": {"type": "string", "description": "可选：用某篇论文的一页/区域作为查询"},
                    "page_number": {"type": "integer", "minimum": 1, "description": "与 paper_id 配套的页码"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "可选：PDF 坐标 [x0,y0,x1,y1]，作为查询区域",
                    },
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "description": "返回数量"},
                },
            },
        }

    def run(
        self,
        image_base64: str | None = None,
        paper_id: str | None = None,
        page_number: int | None = None,
        bbox: list[float] | None = None,
        top_k: int = 5,
        _id_base: int = 0,
    ) -> ToolResult:
        if not self.settings.image_search.enabled:
            return ToolResult(text="图像检索未启用。", sources=[])
        query_bytes = self._query_bytes(image_base64, paper_id, page_number, bbox)
        if not query_bytes:
            return ToolResult(text="未提供有效的查询图片或 paper_id/page_number。", sources=[])
        q = image_signature(query_bytes)

        candidates: list[tuple[Path, int, bytes, float]] = []
        scanned_pages = 0
        for pdf in sorted(self.collection_dir.glob("*.pdf")):
            pages = PaperLoader._read_pdf_pages(pdf)
            for page in pages:
                if scanned_pages >= self.settings.image_search.max_pages:
                    break
                rendered = PaperLoader.render_pdf_page(
                    pdf,
                    page.page_number,
                    max_side=self.settings.image_search.max_side,
                )
                score = cosine(q, image_signature(rendered.data))
                candidates.append((pdf, page.page_number, rendered.data, score))
                scanned_pages += 1
            if scanned_pages >= self.settings.image_search.max_pages:
                break

        ranked = sorted(candidates, key=lambda x: x[3], reverse=True)[:top_k]
        if not ranked:
            return ToolResult(text="未检索到可比较的 PDF 页面图像。", sources=[])

        blocks: list[str] = []
        sources: list[dict] = []
        for i, (pdf, page, image_bytes, score) in enumerate(ranked, start=1):
            sid = f"S{_id_base + i}"
            paper = pdf.stem
            image_b64 = base64.b64encode(image_bytes).decode("ascii")
            blocks.append(f"[{sid}] PDF image match｜paper_id={paper}｜page={page}｜score={score:.4f}")
            sources.append(
                {
                    "id": sid,
                    "chunk_id": f"{paper}::page_image::{page}",
                    "paper_id": paper,
                    "paper_title": paper,
                    "section": f"Page {page} image",
                    "source": str(pdf),
                    "page_start": page,
                    "page_end": page,
                    "element_type": "page_image",
                    "modality": "image",
                    "bbox": None,
                    "chunk_context": f"Page image for 《{paper}》 page {page}",
                    "heading_path": f"Page {page} image",
                    "score": round(score, 4),
                    "snippet": f"Image similarity match on page {page}.",
                    "image_base64": image_b64,
                    "image_mime_type": "image/jpeg",
                }
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
            try:
                return base64.b64decode(image_base64, validate=True)
            except Exception:
                return b""
        if not paper_id or not page_number:
            return b""
        pdf = self.collection_dir / f"{paper_id}.pdf"
        if not pdf.exists():
            return b""
        box = tuple(float(x) for x in bbox) if bbox else None
        return PaperLoader.render_pdf_page(
            pdf,
            page_number,
            max_side=self.settings.image_search.max_side,
            bbox=box,
        ).data


def image_signature(data: bytes) -> np.ndarray:
    """Small deterministic image signature.

    This is not a replacement for CLIP/SigLIP, but it gives an offline testable
    vector path and stable exact/near-exact page-image matching.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size == 0:
        return np.zeros(64, dtype=np.float32)
    hist, _ = np.histogram(arr, bins=64, range=(0, 256))
    vec = hist.astype(np.float32)
    norm = np.linalg.norm(vec) + 1e-8
    return vec / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b))
