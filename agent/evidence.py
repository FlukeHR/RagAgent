from __future__ import annotations

import re
from dataclasses import dataclass, fields

from config.settings import Settings
from retrieval.analyzer import QueryAnalyzer
from tools.base import EvidenceSource, ToolResult


_CITATION = re.compile(r"\[(S\d+)\]")
_NUMBER = re.compile(r"(?<!\w)-?\d+(?:\.\d+)?%?")
_NEGATIONS = (" not ", " no ", "without", "未", "不", "没有", "无法", "并非")


@dataclass
class CitationCheck:
    cleaned: str
    valid: list[str]
    invalid: list[str]
    unsupported_claims: list[str]


class EvidenceRegistry:
    """Deduplicate evidence, assign citation IDs and detect likely conflicts."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.analyzer = QueryAnalyzer(settings.retrieval.cjk_ngram_size)
        self.sources: list[EvidenceSource] = []
        self._by_key: dict[str, EvidenceSource] = {}

    def register(self, result: ToolResult) -> tuple[str, list[EvidenceSource]]:
        local_ids: list[str] = []
        added: list[EvidenceSource] = []
        for source in result.sources:
            existing = (
                self._by_key.get(source.dedup_key)
                if self.settings.retrieval.deduplicate_evidence
                else None
            )
            if existing is not None:
                local_ids.append(existing.citation_id or "")
                continue
            source.citation_id = f"S{len(self.sources) + 1}"
            if (
                source.image_base64
                and len(source.image_base64) > self.settings.harness.image_base64_chars
            ):
                source.image_base64 = None
            self.sources.append(source)
            self._by_key[source.dedup_key] = source
            local_ids.append(source.citation_id)
            added.append(source)

        text = result.text
        for index, citation_id in enumerate(local_ids):
            text = text.replace(f"{{{{cite:{index}}}}}", f"[{citation_id}]")
        text = re.sub(r"\{\{cite:\d+\}\}", "", text)
        return text[: self.settings.harness.tool_result_max_chars], added

    def as_dicts(self) -> list[dict]:
        return [source.to_dict() for source in self.sources]

    def seed_dicts(self, sources: list[dict]) -> None:
        """Load already-numbered fallback sources into the same verifier."""

        allowed = {field.name for field in fields(EvidenceSource)}
        for item in sources:
            data = {key: value for key, value in item.items() if key in allowed}
            data["citation_id"] = item.get("id")
            source = EvidenceSource(**data)
            self.sources.append(source)
            self._by_key[source.dedup_key] = source

    def check_answer(self, answer: str) -> CitationCheck:
        answer = answer or "未能生成答案。"
        valid_ids = {source.citation_id for source in self.sources}
        cited = _CITATION.findall(answer)
        invalid = sorted(
            {citation for citation in cited if citation not in valid_ids},
            key=lambda value: int(value[1:]),
        )
        valid = sorted(
            {citation for citation in cited if citation in valid_ids},
            key=lambda value: int(value[1:]),
        )
        cleaned = answer
        for citation in invalid:
            cleaned = cleaned.replace(f"[{citation}]", "")
        unsupported = self._unsupported_claims(cleaned)
        return CitationCheck(cleaned, valid, invalid, unsupported)

    def _unsupported_claims(self, answer: str) -> list[str]:
        unsupported: list[str] = []
        source_by_id = {source.citation_id: source for source in self.sources}
        for sentence in re.split(r"(?<=[。！？.!?])\s*", answer):
            citations = _CITATION.findall(sentence)
            claim = _CITATION.sub("", sentence).strip()
            if not citations or len(claim) < 4:
                continue
            best = max(
                (
                    self.analyzer.overlap(claim, source_by_id[citation].snippet or "")
                    for citation in citations
                    if citation in source_by_id
                ),
                default=0.0,
            )
            if best < self.settings.harness.claim_support_min_overlap:
                unsupported.append(claim[:240])
        return unsupported

    def detect_conflicts(self) -> list[dict]:
        if not self.settings.retrieval.conflict_detection:
            return []
        conflicts: list[dict] = []
        for left_index, left in enumerate(self.sources):
            for right in self.sources[left_index + 1 :]:
                if left.paper_id == right.paper_id:
                    continue
                left_text = left.snippet or ""
                right_text = right.snippet or ""
                overlap = min(
                    self.analyzer.overlap(left_text, right_text),
                    self.analyzer.overlap(right_text, left_text),
                )
                if overlap < 0.18:
                    continue
                left_numbers = set(_NUMBER.findall(left_text))
                right_numbers = set(_NUMBER.findall(right_text))
                numeric = bool(left_numbers and right_numbers and left_numbers != right_numbers)
                negation = self._has_negation(left_text) != self._has_negation(right_text)
                if numeric or negation:
                    left.support_status = "conflict"
                    right.support_status = "conflict"
                    preferred = self._preferred(left, right)
                    conflicts.append(
                        {
                            "left": left.citation_id,
                            "right": right.citation_id,
                            "reason": "numeric" if numeric else "negation",
                            "overlap": round(overlap, 3),
                            "preferred": preferred,
                            "left_published_at": left.published_at,
                            "right_published_at": right.published_at,
                            "left_quality_rank": left.quality_rank,
                            "right_quality_rank": right.quality_rank,
                        }
                    )
        return conflicts

    @staticmethod
    def _has_negation(text: str) -> bool:
        lowered = f" {text.lower()} "
        return any(token in lowered for token in _NEGATIONS)

    @staticmethod
    def _preferred(left: EvidenceSource, right: EvidenceSource) -> str | None:
        left_key = (left.quality_rank, left.published_at or "")
        right_key = (right.quality_rank, right.published_at or "")
        if left_key == right_key:
            return None
        return (
            left.citation_id
            if left_key > right_key
            else right.citation_id
        )
