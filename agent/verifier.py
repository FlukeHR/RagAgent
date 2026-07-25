from __future__ import annotations

from agent.evidence import CitationCheck, EvidenceRegistry


class AnswerVerifier:
    """Claim/citation verification facade kept separate from orchestration."""

    def __init__(self, evidence: EvidenceRegistry) -> None:
        self.evidence = evidence

    def verify(self, answer: str) -> CitationCheck:
        return self.evidence.check_answer(answer)

    def conflicts(self) -> list[dict]:
        return self.evidence.detect_conflicts()
