from __future__ import annotations

import re

from retrieval.models import PaperPage, PaperSection


_SECTION_KEYWORDS = {
    "abstract",
    "introduction",
    "related work",
    "background",
    "preliminaries",
    "method",
    "methodology",
    "approach",
    "model",
    "architecture",
    "experiment",
    "experiments",
    "experimental setup",
    "results",
    "evaluation",
    "analysis",
    "ablation",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "future work",
    "references",
    "acknowledgment",
    "acknowledgments",
    "acknowledgements",
    "appendix",
}
_NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z].{0,70}$")


class DocumentNormalizer:
    """Normalize parsed text/pages into titled, page-aware paper sections."""

    @staticmethod
    def title(text: str, fallback: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if len(stripped) >= 8 and not stripped.lower().startswith("arxiv"):
                return stripped[:200]
        return fallback

    def sections_from_text(self, text: str) -> list[PaperSection]:
        sections: list[PaperSection] = []
        current_title = "Body"
        current_lines: list[str] = []

        def flush() -> None:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append(PaperSection(title=current_title, text=body))

        for line in text.splitlines():
            if self.is_heading(line):
                flush()
                current_title = line.strip()[:80]
                current_lines = []
            else:
                current_lines.append(line)
        flush()
        return sections or [PaperSection(title="Body", text=text.strip())]

    def sections_from_pages(self, pages: list[PaperPage]) -> list[PaperSection]:
        sections: list[PaperSection] = []
        current_title = "Body"
        current_lines: list[str] = []
        current_start = pages[0].page_number if pages else None
        current_end = current_start
        current_modality = pages[0].dominant_modality if pages else "text"

        def flush() -> None:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append(
                    PaperSection(
                        title=current_title,
                        text=body,
                        page_start=current_start,
                        page_end=current_end,
                        modality=current_modality,
                        heading_path=current_title,
                    )
                )

        for page in pages:
            page_modality = page.dominant_modality
            for line in page.primary_text.splitlines():
                if self.is_heading(line):
                    flush()
                    current_title = line.strip()[:80]
                    current_lines = []
                    current_start = page.page_number
                    current_end = page.page_number
                    current_modality = page_modality
                else:
                    if not current_lines:
                        current_modality = page_modality
                    current_lines.append(line)
                    current_end = page.page_number
            if current_lines:
                current_end = page.page_number
        flush()
        if sections:
            return sections
        text = "\n".join(page.primary_text for page in pages).strip()
        return (
            [
                PaperSection(
                    title="Body",
                    text=text,
                    page_start=pages[0].page_number if pages else None,
                    page_end=pages[-1].page_number if pages else None,
                    modality=pages[0].dominant_modality if pages else "text",
                    heading_path="Body",
                )
            ]
            if text
            else []
        )

    @staticmethod
    def is_heading(line: str) -> bool:
        stripped = line.strip()
        if not stripped or len(stripped) > 80:
            return False
        return bool(_NUMBERED_HEADING.match(stripped)) or (
            stripped.lower().rstrip(":.") in _SECTION_KEYWORDS
        )
