from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from retrieval.models import (
    BBox,
    PaperElement as ParsedElement,
    PaperPage as ParsedPage,
    ParsedPDF,
    TextBlock as ParsedBlock,
)


class PDFParseProvider:
    """Provider interface for PDF text, OCR/VLM sidecars, layout and elements."""

    name = "base"

    def parse(self, file: Path) -> ParsedPDF:
        raise NotImplementedError


class TesseractOCRRuntime:
    """Optional local OCR runtime.

    It is deliberately opt-in and best-effort: if the executable is not available
    or one page times out, parsing falls back to ordinary PDF text.
    """

    def __init__(self, executable: str = "tesseract", lang: str = "eng", timeout_seconds: float = 30.0) -> None:
        self.executable = executable
        self.lang = lang
        self.timeout_seconds = timeout_seconds

    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    def ocr_page(self, file: Path, page_number: int, max_side: int = 1800) -> str:
        if not self.available():
            return ""
        try:
            import fitz
        except ImportError:
            return ""

        with tempfile.TemporaryDirectory() as td:
            image_path = Path(td) / "page.png"
            with fitz.open(str(file)) as doc:
                page = doc[page_number - 1]
                side = max(float(page.rect.width), float(page.rect.height), 1.0)
                zoom = min(3.0, max_side / side)
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                image_path.write_bytes(pix.tobytes("png"))
            try:
                proc = subprocess.run(
                    [self.executable, str(image_path), "stdout", "-l", self.lang],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except (OSError, subprocess.TimeoutExpired):
                return ""
            if proc.returncode != 0:
                return ""
            return proc.stdout.strip()


class PyMuPDFParseProvider(PDFParseProvider):
    """Default local PDF parser: PyMuPDF text + sidecars + optional Tesseract OCR."""

    name = "pymupdf"

    def __init__(
        self,
        min_scan_text_chars: int = 50,
        ocr_runtime: TesseractOCRRuntime | None = None,
        persist_generated_sidecars: bool = False,
    ) -> None:
        self.min_scan_text_chars = min_scan_text_chars
        self.ocr_runtime = ocr_runtime
        self.persist_generated_sidecars = persist_generated_sidecars

    def parse(self, file: Path) -> ParsedPDF:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("解析 PDF 需要 PyMuPDF，请先安装：pip install pymupdf") from exc

        ocr_by_page = read_page_sidecar(file, ".ocr.json", ("ocr_text", "text", "content"))
        vlm_by_page = read_page_sidecar(file, ".vlm.json", ("vlm_summary", "summary", "text", "content"))
        generated_ocr: dict[int, str] = {}
        pages: list[ParsedPage] = []
        trace: list[dict] = []

        with fitz.open(str(file)) as doc:
            for page_number, page in enumerate(doc, start=1):
                text = page.get_text("text")
                visible = "".join(text.split())
                scanned = len(visible) < self.min_scan_text_chars
                ocr_text = ocr_by_page.get(page_number)
                if scanned and not ocr_text and self.ocr_runtime is not None:
                    ocr_text = self.ocr_runtime.ocr_page(file, page_number)
                    if ocr_text:
                        generated_ocr[page_number] = ocr_text
                    trace.append(
                        {
                            "provider": self.name,
                            "runtime": "tesseract",
                            "page": page_number,
                            "generated": bool(ocr_text),
                        }
                    )
                pages.append(
                    ParsedPage(
                        page_number=page_number,
                        text=text,
                        is_scanned_like=scanned,
                        ocr_text=ocr_text,
                        vlm_summary=vlm_by_page.get(page_number),
                        blocks=_blocks_from_page(page, page_number),
                    )
                )

        if generated_ocr and self.persist_generated_sidecars:
            _write_ocr_sidecar(file, {**ocr_by_page, **generated_ocr})

        return ParsedPDF(pages=pages, elements=read_element_sidecars(file), trace=trace)


def provider_from_config(provider: str = "pymupdf", auto_ocr: bool = False, timeout_seconds: float = 30.0) -> PDFParseProvider:
    """Create a configured provider, falling back to PyMuPDF for unknown optional providers."""
    name = (provider or "pymupdf").lower()
    ocr_runtime = TesseractOCRRuntime(timeout_seconds=timeout_seconds) if auto_ocr or name == "tesseract" else None
    return PyMuPDFParseProvider(
        ocr_runtime=ocr_runtime,
        persist_generated_sidecars=bool(auto_ocr or name == "tesseract"),
    )


def read_page_sidecar(pdf: Path, suffix: str, text_keys: tuple[str, ...]) -> dict[int, str]:
    path = pdf.with_suffix(suffix)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    out: dict[int, str] = {}

    def add(page_value: Any, payload: Any) -> None:
        try:
            page_number = int(page_value)
        except (TypeError, ValueError):
            return
        text = ""
        if isinstance(payload, str):
            text = payload
        elif isinstance(payload, dict):
            for key in text_keys:
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    text = value
                    break
        if text.strip():
            out[page_number] = text.strip()

    if isinstance(raw, dict) and isinstance(raw.get("pages"), list):
        for item in raw["pages"]:
            if isinstance(item, dict):
                add(item.get("page") or item.get("page_number"), item)
    elif isinstance(raw, dict):
        for page_value, payload in raw.items():
            add(page_value, payload)
    return out


def read_element_sidecars(pdf: Path) -> list[ParsedElement]:
    elements: list[ParsedElement] = []
    for suffix, default_type in (
        (".elements.json", None),
        (".layout.json", "text_block"),
        (".tables.json", "table"),
        (".figures.json", "figure"),
        (".formulas.json", "formula"),
        (".bboxes.json", "text_block"),
    ):
        path = pdf.with_suffix(suffix)
        if path.exists():
            elements.extend(_parse_element_json(path, default_type=default_type))
    return elements


def _parse_element_json(path: Path, default_type: str | None) -> list[ParsedElement]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    items: list[dict] = []
    if isinstance(raw, list):
        items = [x for x in raw if isinstance(x, dict)]
    elif isinstance(raw, dict):
        for key in ("elements", "tables", "figures", "formulas", "blocks"):
            value = raw.get(key)
            if isinstance(value, list):
                items.extend(x for x in value if isinstance(x, dict))
        pages = raw.get("pages")
        if isinstance(pages, list):
            for page in pages:
                if not isinstance(page, dict):
                    continue
                page_number = page.get("page") or page.get("page_number")
                for key in ("elements", "tables", "figures", "formulas", "blocks"):
                    value = page.get(key)
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                merged = {"page": page_number, **item}
                                if default_type is None:
                                    merged.setdefault("element_type", key[:-1] if key.endswith("s") else key)
                                items.append(merged)

    out: list[ParsedElement] = []
    for i, item in enumerate(items, start=1):
        element = _element_from_item(item, default_type, f"{path.stem}::{i}", path.name)
        if element is not None and element.content.strip():
            out.append(element)
    return out


def _element_from_item(item: dict, default_type: str | None, fallback_id: str, source: str) -> ParsedElement | None:
    page = item.get("page") or item.get("page_number") or item.get("page_start")
    page_end = item.get("page_end") or page
    if page is None or page_end is None:
        return None
    try:
        page_start_i = int(page)
        page_end_i = int(page_end)
    except (TypeError, ValueError):
        return None
    element_type = str(item.get("element_type") or item.get("type") or default_type or "element")
    text = _first_text(item, ("text", "content", "markdown", "latex", "html"))
    caption = _first_text(item, ("caption", "title"))
    summary = _first_text(item, ("summary", "description"))
    if not text and isinstance(item.get("cells"), list):
        text = _cells_to_text(item["cells"])
    return ParsedElement(
        element_id=str(item.get("id") or item.get("element_id") or fallback_id),
        element_type=element_type,
        page_start=page_start_i,
        page_end=page_end_i,
        text=text,
        modality=str(item.get("modality") or _default_modality(element_type)),
        bbox=_bbox(item.get("bbox")),
        caption=caption or None,
        summary=summary or None,
        source=source,
    )


def _blocks_from_page(page: Any, page_number: int) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    try:
        raw_blocks = page.get_text("blocks")
    except Exception:
        return blocks
    for block in raw_blocks:
        if len(block) < 5:
            continue
        text = str(block[4]).strip()
        if not text:
            continue
        blocks.append(ParsedBlock(page_number=page_number, text=text, bbox=_bbox(block[:4])))
    return blocks


def _write_ocr_sidecar(pdf: Path, ocr_by_page: dict[int, str]) -> None:
    path = pdf.with_suffix(".ocr.json")
    payload = {
        "provider": "tesseract",
        "pages": [
            {"page": page, "ocr_text": text}
            for page, text in sorted(ocr_by_page.items())
            if text.strip()
        ],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _first_text(item: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _cells_to_text(cells: list) -> str:
    rows: dict[int, list[str]] = {}
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        row = int(cell.get("row") or 0)
        col = int(cell.get("col") or cell.get("column") or 0)
        rows.setdefault(row, [])
        while len(rows[row]) <= col:
            rows[row].append("")
        rows[row][col] = str(cell.get("text") or cell.get("value") or "").strip()
    return "\n".join(" | ".join(v for v in rows[row] if v) for row in sorted(rows))


def _default_modality(element_type: str) -> str:
    if element_type in {"figure", "image", "page_image"}:
        return "image"
    if element_type == "formula":
        return "formula"
    return "text"


def _bbox(value: Any) -> BBox | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(float(x) for x in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None
