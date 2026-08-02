from __future__ import annotations

import json
from typing import Any

from config.settings import BASE_DIR, Settings
from retrieval.mineru import MinerUAdapter, MinerUError, clean_text
from retrieval.documents import PaperElement, PaperPage, PaperRepository, PaperSection
from tools.base import EvidenceSource, ToolResult


class InspectPaperTool:
    """Read one bounded view from a canonical MinerU sidecar."""

    name = "inspect_paper"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repository = PaperRepository(BASE_DIR / settings.project.data_root)
        self.adapter = MinerUAdapter(settings.mineru)

    def run(self, paper_id: str, locator: dict[str, Any]) -> ToolResult:
        pdf = self.repository.resolve(paper_id, (".pdf",))
        if pdf is None:
            return ToolResult(text=f"No local PDF found for paper_id={paper_id}.")
        sidecar = pdf.with_suffix(".mineru.json")
        if not sidecar.exists():
            return ToolResult(
                text=f"Paper {paper_id} has no successful MinerU sidecar; rebuild the index."
            )
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            parsed = self.adapter.load(payload, pdf)
        except (OSError, json.JSONDecodeError, MinerUError) as exc:
            return ToolResult(text=f"MinerU sidecar for {paper_id} is invalid: {exc}")

        kind = str(locator.get("kind") or "")
        if kind == "overview":
            return self._overview(paper_id, pdf, parsed.sections, parsed.elements, payload)
        if kind == "section":
            return self._section(
                paper_id, pdf, parsed.sections, str(locator.get("section") or ""), payload
            )
        if kind == "page":
            return self._page(
                paper_id,
                pdf,
                parsed.pages,
                parsed.elements,
                int(locator["page_number"]),
                payload,
            )
        if kind == "element":
            return self._element(
                paper_id, pdf, parsed.elements, str(locator["element_id"]), payload
            )
        if kind == "region":
            box = self._validate_region(locator["bbox"])
            return self._region(
                paper_id,
                pdf,
                parsed.pages,
                parsed.elements,
                int(locator["page_number"]),
                box,
                payload,
            )
        return ToolResult(text=f"Unsupported inspect locator: {kind}")

    def _overview(
        self,
        paper_id: str,
        pdf: Any,
        sections: list[PaperSection],
        elements: list[PaperElement],
        payload: dict[str, Any],
    ) -> ToolResult:
        toc = "\n".join(f"- {section.title}" for section in sections[:80])
        counts: dict[str, int] = {}
        for element in elements:
            counts[element.element_type] = counts.get(element.element_type, 0) + 1
        element_summary = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        snippet = sections[0].text if sections else ""
        snippet = self._bounded(snippet)
        source = self._source(
            paper_id,
            pdf,
            "Overview",
            snippet,
            payload,
            page_start=1,
            page_end=int(payload.get("page_count") or 1),
        )
        return ToolResult(
            text=(
                f"{{{{cite:0}}}} Untrusted paper evidence.\nSections:\n{toc}\n\n"
                f"Elements: {element_summary or 'none'}\n\nOpening:\n{snippet}"
            ),
            sources=[source],
        )

    def _section(
        self,
        paper_id: str,
        pdf: Any,
        sections: list[PaperSection],
        requested: str,
        payload: dict[str, Any],
    ) -> ToolResult:
        lowered = requested.casefold()
        exact = [section for section in sections if section.title.casefold() == lowered]
        matches = exact or [
            section for section in sections if lowered in section.title.casefold()
        ]
        if len(matches) != 1:
            candidates = [section.title for section in matches or sections]
            return ToolResult(
                text=(
                    f"Section locator {requested!r} is not unique. "
                    f"Candidates: {candidates[:30]}"
                )
            )
        selected = matches[0]
        selected_text = self._bounded(selected.text)
        source = self._source(
            paper_id,
            pdf,
            selected.title,
            selected_text,
            payload,
            page_start=selected.page_start,
            page_end=selected.page_end,
        )
        return ToolResult(
            text=(
                f"{{{{cite:0}}}} Untrusted paper evidence.\n"
                f"Section: {selected.title}\n{selected_text}"
            ),
            sources=[source],
        )

    def _page(
        self,
        paper_id: str,
        pdf: Any,
        pages: list[PaperPage],
        elements: list[PaperElement],
        page_number: int,
        payload: dict[str, Any],
    ) -> ToolResult:
        selected = next((page for page in pages if page.page_number == page_number), None)
        if selected is None:
            return ToolResult(text=f"Page {page_number} is outside the paper.")
        page_elements = [item for item in elements if item.page_start == page_number]
        element_text = "\n\n".join(
            f"[{item.element_id} | {item.element_type} | bbox={item.bbox}]\n{item.content}"
            for item in page_elements
        )
        body = selected.text
        if element_text:
            body += f"\n\nStructured elements:\n{element_text}"
        body = self._bounded(body)
        source = self._source(
            paper_id,
            pdf,
            f"Page {page_number}",
            body,
            payload,
            page_start=page_number,
            page_end=page_number,
        )
        return ToolResult(
            text=f"{{{{cite:0}}}} Untrusted paper evidence.\nPage {page_number}\n{body}",
            sources=[source],
        )

    def _element(
        self,
        paper_id: str,
        pdf: Any,
        elements: list[PaperElement],
        element_id: str,
        payload: dict[str, Any],
    ) -> ToolResult:
        selected = next((item for item in elements if item.element_id == element_id), None)
        if selected is None:
            return ToolResult(text=f"Unknown MinerU element_id={element_id}.")
        source = self._source(
            paper_id,
            pdf,
            selected.heading_path or selected.element_type,
            self._bounded(selected.content),
            payload,
            page_start=selected.page_start,
            page_end=selected.page_end,
            element=selected,
        )
        return ToolResult(
            text=(
                f"{{{{cite:0}}}} Untrusted paper evidence.\n"
                f"Element {selected.element_id} ({selected.element_type})\n"
                f"page={selected.page_start}, bbox={selected.bbox}\n"
                f"{self._bounded(selected.content)}"
            ),
            sources=[source],
        )

    def _region(
        self,
        paper_id: str,
        pdf: Any,
        pages: list[PaperPage],
        elements: list[PaperElement],
        page_number: int,
        bbox: tuple[float, float, float, float],
        payload: dict[str, Any],
    ) -> ToolResult:
        page = next((item for item in pages if item.page_number == page_number), None)
        if page is None:
            return ToolResult(text=f"Page {page_number} is outside the paper.")
        blocks = [block.text for block in page.blocks if block.bbox and _intersects(block.bbox, bbox)]
        selected_elements = [
            item
            for item in elements
            if item.page_start == page_number and item.bbox and _intersects(item.bbox, bbox)
        ]
        parts = blocks + [item.content for item in selected_elements if item.content]
        text = "\n\n".join(
            dict.fromkeys(clean_text(part) for part in parts if clean_text(part))
        )
        text = self._bounded(text)
        if not text:
            return ToolResult(text="No canonical MinerU content intersects that region.")
        source = self._source(
            paper_id,
            pdf,
            f"Page {page_number} region",
            text,
            payload,
            page_start=page_number,
            page_end=page_number,
            bbox=bbox,
        )
        return ToolResult(
            text=(
                f"{{{{cite:0}}}} Untrusted paper evidence.\n"
                f"Page {page_number}, normalized bbox={bbox}\n{text}"
            ),
            sources=[source],
        )

    def _source(
        self,
        paper_id: str,
        pdf: Any,
        section: str,
        snippet: str,
        payload: dict[str, Any],
        *,
        page_start: int | None,
        page_end: int | None,
        element: PaperElement | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> EvidenceSource:
        parser = payload.get("parser") or {}
        return EvidenceSource(
            paper_id=paper_id,
            paper_title=str(payload.get("title") or paper_id),
            section=section,
            source=str(pdf),
            chunk_id=(
                f"{paper_id}::element::{element.element_id}"
                if element
                else f"{paper_id}::{section}"
            ),
            page_start=page_start,
            page_end=page_end,
            element_type=element.element_type if element else "page",
            modality=element.modality if element else "text",
            bbox=element.bbox if element else bbox,
            bbox_space="normalized_1000" if (element and element.bbox) or bbox else None,
            element_id=element.element_id if element else None,
            parser_metadata=dict(parser),
            heading_path=element.heading_path if element else section,
            snippet=snippet[: self.settings.agent.source_snippet_chars],
            chunk_context=(
                f"MinerU {parser.get('version', 'unknown')} | {section} | "
                f"pages {page_start}-{page_end}"
            ),
            quality_rank=4 if element else 3,
        )

    @staticmethod
    def _validate_region(values: list[Any]) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = (float(value) for value in values)
        if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
            raise ValueError("bbox must satisfy 0 <= x0 < x1 <= 1000 and y likewise")
        return x0, y0, x1, y1

    def _bounded(self, text: str) -> str:
        reserve = 1000
        limit = max(1000, self.settings.agent.tool_result_max_chars - reserve)
        return text[:limit]


def _intersects(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return not (
        left[2] <= right[0]
        or right[2] <= left[0]
        or left[3] <= right[1]
        or right[3] <= left[1]
    )
