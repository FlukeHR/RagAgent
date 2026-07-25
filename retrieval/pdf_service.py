from __future__ import annotations

from pathlib import Path

from retrieval.models import PDFPageImage, PaperPage, TextBlock
from retrieval.pdf_parse import read_page_sidecar


class PDFService:
    """Bounded single-page PDF reads and rendering without parsing the whole file."""

    def read_page(self, file: Path, page_number: int, min_scan_text_chars: int = 50) -> PaperPage:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("解析 PDF 需要 PyMuPDF，请先安装：pip install pymupdf") from exc

        with fitz.open(str(file)) as doc:
            if page_number < 1 or page_number > len(doc):
                raise ValueError(
                    f"page_number out of range: {page_number}; total pages: {len(doc)}"
                )
            page = doc[page_number - 1]
            text = page.get_text("text")
            blocks: list[TextBlock] = []
            for block in page.get_text("blocks"):
                if len(block) >= 5 and str(block[4]).strip():
                    blocks.append(
                        TextBlock(
                            page_number=page_number,
                            text=str(block[4]).strip(),
                            bbox=(
                                float(block[0]),
                                float(block[1]),
                                float(block[2]),
                                float(block[3]),
                            ),
                        )
                    )

        ocr = read_page_sidecar(file, ".ocr.json", ("ocr_text", "text", "content"))
        vlm = read_page_sidecar(file, ".vlm.json", ("vlm_summary", "summary", "text", "content"))
        return PaperPage(
            page_number=page_number,
            text=text,
            is_scanned_like=len("".join(text.split())) < min_scan_text_chars,
            ocr_text=ocr.get(page_number),
            vlm_summary=vlm.get(page_number),
            blocks=blocks,
        )

    @staticmethod
    def page_count(file: Path) -> int:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("解析 PDF 需要 PyMuPDF，请先安装：pip install pymupdf") from exc
        with fitz.open(str(file)) as doc:
            return len(doc)

    @staticmethod
    def read_region(
        file: Path,
        page_number: int,
        bbox: tuple[float, float, float, float],
    ) -> str:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("解析 PDF 需要 PyMuPDF，请先安装：pip install pymupdf") from exc
        with fitz.open(str(file)) as doc:
            if page_number < 1 or page_number > len(doc):
                raise ValueError(f"page_number out of range: {page_number}")
            page = doc[page_number - 1]
            rect = fitz.Rect(*bbox) & page.rect
            if rect.is_empty:
                raise ValueError(f"bbox outside page: {bbox}")
            return page.get_text("text", clip=rect).strip()

    @staticmethod
    def render_page(
        file: Path,
        page_number: int,
        max_side: int = 1600,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> PDFPageImage:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("解析 PDF 需要 PyMuPDF，请先安装：pip install pymupdf") from exc
        with fitz.open(str(file)) as doc:
            if page_number < 1 or page_number > len(doc):
                raise ValueError(f"page_number out of range: {page_number}")
            if max_side <= 0:
                raise ValueError("max_side must be positive")
            page = doc[page_number - 1]
            clip = None
            if bbox is not None:
                clip = fitz.Rect(*bbox) & page.rect
                if clip.is_empty:
                    raise ValueError(f"bbox outside page: {bbox}")
            rect = clip or page.rect
            side = max(float(rect.width), float(rect.height), 1.0)
            zoom = min(2.0, max_side / side)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False, clip=clip)
            try:
                data = pix.tobytes("jpg")
                mime = "image/jpeg"
            except Exception:
                data = pix.tobytes("png")
                mime = "image/png"
            return PDFPageImage(page_number, mime, data, pix.width, pix.height)
