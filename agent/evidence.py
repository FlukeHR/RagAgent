from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, fields
from typing import Any

from config.settings import Settings
from retrieval.search import QueryAnalyzer
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


@dataclass
class ExecutionResult:
    """Unverified output produced by an Agent or local RAG executor."""

    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentAnswer:
    """Verified public answer returned by the API-facing Agent."""

    answer: str
    status: str = "answered"
    steps: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)


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
                existing.origin_tools = list(
                    dict.fromkeys([*existing.origin_tools, *source.origin_tools])
                )
                local_ids.append(existing.citation_id or "")
                continue
            source.citation_id = f"S{len(self.sources) + 1}"
            self.sources.append(source)
            self._by_key[source.dedup_key] = source
            local_ids.append(source.citation_id)
            added.append(source)

        text = result.text
        for index, citation_id in enumerate(local_ids):
            text = text.replace(f"{{{{cite:{index}}}}}", f"[{citation_id}]")
        text = re.sub(r"\{\{cite:\d+\}\}", "", text)
        return text[: self.settings.agent.tool_result_max_chars], added

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
            if best < self.settings.retrieval.claim_support_min_overlap:
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


@dataclass(frozen=True)
class EvidenceAssessment:
    """Whether retrieved sources can support an answer or need another hop."""

    sufficient: bool
    low_confidence: bool
    reason: str


class EvidenceSelector:
    """Choose a small, relevant and diverse evidence set for final synthesis."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.analyzer = QueryAnalyzer(settings.retrieval.cjk_ngram_size)

    def select(
        self,
        question: str,
        sources: list[dict[str, Any]],
        *,
        required_tools: tuple[str, ...] = (),
        preferred_ids: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        """Rank, deduplicate and cap sources while preserving required tools."""

        candidates = [
            source
            for source in sources
            if source.get("id") and str(source.get("snippet") or "").strip()
        ]
        if not candidates:
            return []

        preferred = set(preferred_ids)
        ranked = sorted(
            enumerate(candidates),
            key=lambda item: self._rank(question, item[1], preferred, item[0]),
            reverse=True,
        )
        ordered = [source for _, source in ranked]
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        paper_counts: dict[str, int] = {}

        # Required capabilities get one reserved slot so final composition cannot
        # accidentally discard the only external/local evidence it must cite.
        for tool in required_tools:
            source = next(
                (
                    item
                    for item in ordered
                    if tool in item.get("origin_tools", [])
                    and str(item.get("id")) not in selected_ids
                ),
                None,
            )
            if source is not None:
                self._append(source, selected, selected_ids, paper_counts)

        for source in ordered:
            if len(selected) >= self.settings.agent.final_max_sources:
                break
            source_id = str(source.get("id"))
            if source_id in selected_ids:
                continue
            paper_key = str(source.get("paper_id") or source.get("source") or source_id)
            if (
                paper_counts.get(paper_key, 0)
                >= self.settings.agent.final_max_sources_per_paper
            ):
                continue
            self._append(source, selected, selected_ids, paper_counts)
        return selected[: self.settings.agent.final_max_sources]

    def _rank(
        self,
        question: str,
        source: dict[str, Any],
        preferred: set[str],
        original_index: int,
    ) -> tuple[float, ...]:
        evidence = " ".join(
            str(source.get(name) or "")
            for name in ("paper_title", "section", "snippet")
        )
        relevance = max(
            self.analyzer.overlap(question, evidence),
            self.analyzer.identifier_overlap(question, evidence),
        )
        confidence = self._number(source.get("confidence"), default=0.5)
        lexical = self._number(source.get("lexical_anchor_score"), default=0.0)
        quality = self._number(source.get("quality_rank"), default=0.0)
        return (
            1.0 if str(source.get("id")) in preferred else 0.0,
            relevance,
            confidence,
            lexical,
            quality,
            -float(original_index),
        )

    @staticmethod
    def _number(value: Any, *, default: float) -> float:
        try:
            return float(value) if value is not None else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _append(
        source: dict[str, Any],
        selected: list[dict[str, Any]],
        selected_ids: set[str],
        paper_counts: dict[str, int],
    ) -> None:
        source_id = str(source.get("id"))
        paper_key = str(source.get("paper_id") or source.get("source") or source_id)
        selected.append(source)
        selected_ids.add(source_id)
        paper_counts[paper_key] = paper_counts.get(paper_key, 0) + 1


@dataclass(frozen=True)
class VerificationResult:
    """Citation and answerability result for one draft answer."""

    checked: CitationCheck
    sources: list[dict[str, Any]]
    conflicts: list[dict[str, Any]]
    answerable: bool
    reason: str
    retry_reasons: tuple[str, ...]

    @property
    def needs_retry(self) -> bool:
        return bool(self.retry_reasons)


class AnswerVerifier:
    """Single policy for evidence sufficiency, citations, conflicts, and refusal."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def assess_sources(self, sources: list[dict[str, Any]]) -> EvidenceAssessment:
        """Check hard answerability gates and the softer corrective-search signal."""

        effective = [
            source
            for source in sources
            if source.get("snippet") and self._meets_minimum_score(source)
        ]
        minimum = self.settings.retrieval.answerability_min_sources
        if len(effective) < minimum:
            return EvidenceAssessment(False, True, f"有效来源不足 {minimum} 条")
        return EvidenceAssessment(True, self._low_confidence(effective), "证据充分")

    def verify(
        self,
        answer: str,
        sources: list[dict[str, Any]],
    ) -> VerificationResult:
        """Verify a draft against already numbered source dictionaries."""

        evidence = EvidenceRegistry(self.settings)
        evidence.seed_dicts(sources)
        return self.verify_registry(answer, evidence)

    def verify_registry(
        self,
        answer: str,
        evidence: EvidenceRegistry,
    ) -> VerificationResult:
        """Verify a draft against request-scoped evidence from model tools."""

        checked = evidence.check_answer(answer)
        conflicts = evidence.detect_conflicts()
        sources = evidence.as_dicts()
        assessment = self.assess_sources(sources)
        citation_missing = bool(
            self.settings.retrieval.answerability_require_citation
            and not checked.valid
        )
        answerable = assessment.sufficient and not citation_missing
        if not assessment.sufficient:
            reason = assessment.reason
        elif citation_missing:
            reason = "答案没有可回查引用"
        else:
            reason = assessment.reason

        retry_reasons: list[str] = []
        if not assessment.sufficient:
            retry_reasons.append(assessment.reason)
        elif assessment.low_confidence:
            retry_reasons.append("来源相关性偏低")
        if citation_missing:
            retry_reasons.append("答案缺少真实引用")
        if checked.invalid:
            retry_reasons.append("答案含无效引用")
        if checked.unsupported_claims:
            retry_reasons.append("部分陈述缺少词面支持")
        return VerificationResult(
            checked,
            sources,
            conflicts,
            answerable,
            reason,
            tuple(dict.fromkeys(retry_reasons)),
        )

    def finalize(self, execution: ExecutionResult) -> AgentAnswer:
        """Convert unverified executor output into the public answer contract."""

        result = self.verify(execution.answer, execution.sources)
        steps = list(execution.steps)
        trace = list(execution.trace)
        if not result.answerable:
            steps.append(f"Verify: {result.reason}，拒绝生成实质答案")
            trace.append(
                {"type": "verify", "result": "reject", "reason": result.reason}
            )
            return AgentAnswer(
                answer=self.insufficient_answer(result.reason, result.sources),
                status="insufficient_evidence",
                steps=steps,
                sources=result.sources,
                trace=trace,
            )

        answer = result.checked.cleaned
        if result.checked.invalid:
            steps.append(f"Verify: 无效引用 {result.checked.invalid} 已移除")
            answer += (
                f"\n\n> 引用核查：无效引用 {', '.join(result.checked.invalid)} 已移除。"
            )
        if result.checked.unsupported_claims:
            steps.append(
                f"Verify: {len(result.checked.unsupported_claims)} 条陈述需要人工复核"
            )
            trace.append(
                {
                    "type": "claim_verify",
                    "unsupported": result.checked.unsupported_claims,
                }
            )
        if result.conflicts:
            answer += (
                "\n\n> 检测到来源间可能存在数值或否定关系冲突；"
                "请分别核对实验条件，不将其视为一致结论。"
            )
            trace.append({"type": "evidence_conflict", "items": result.conflicts})
        steps.append(f"Verify: {len(result.checked.valid)} 条引用可回查")
        trace.append(
            {
                "type": "verify",
                "result": "final",
                "valid": result.checked.valid,
                "hallucinated": result.checked.invalid,
            }
        )
        if self.settings.retrieval.answerability_require_citation:
            valid_ids = set(result.checked.valid)
            answer_sources = [
                source
                for source in result.sources
                if str(source.get("id")) in valid_ids
            ]
        else:
            # Keep retrieved sources visible when inline citations are optional.
            answer_sources = result.sources
        return AgentAnswer(answer, "answered", steps, answer_sources, trace)

    @staticmethod
    def insufficient_answer(reason: str, sources: list[dict[str, Any]]) -> str:
        """Render the product-wide evidence refusal message."""

        if not sources:
            return f"未检索到充分依据，无法可靠回答这个问题。（原因：{reason}）"
        identifiers = ", ".join(
            str(source.get("id")) for source in sources[:3] if source.get("id")
        )
        suffix = (
            f" 已检索到的候选来源（{identifiers}）不足以支撑实质结论。"
            if identifiers
            else ""
        )
        return f"未检索到充分依据，无法可靠回答这个问题。（原因：{reason}）{suffix}"

    def _meets_minimum_score(self, source: dict[str, Any]) -> bool:
        confidence = source.get("confidence")
        if confidence is not None:
            try:
                return (
                    float(confidence)
                    >= self.settings.retrieval.answerability_min_confidence
                )
            except (TypeError, ValueError):
                return False
        score = source.get("score")
        if score is None:
            return True
        try:
            return float(score) >= self.settings.retrieval.answerability_min_score
        except (TypeError, ValueError):
            return False

    def _low_confidence(self, sources: list[dict[str, Any]]) -> bool:
        values: list[float] = []
        for source in sources:
            confidence = source.get("confidence")
            score = source.get("score")
            try:
                if confidence is not None:
                    values.append(float(confidence))
                elif score is not None:
                    bounded = max(-30.0, min(30.0, float(score)))
                    values.append(1.0 / (1.0 + math.exp(-bounded)))
            except (TypeError, ValueError):
                continue
        if not values:
            return False
        retrieval = self.settings.retrieval
        return not (
            max(values) >= retrieval.low_confidence_threshold
            and sum(value >= retrieval.weak_confidence_threshold for value in values)
            >= retrieval.min_confident_sources
        )
