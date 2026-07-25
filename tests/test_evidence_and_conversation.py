from __future__ import annotations

import unittest
from dataclasses import replace

from agent.conversation import ConversationManager, ConversationState
from agent.evidence import EvidenceRegistry
from config.settings import load_settings
from tools.base import EvidenceSource, ToolResult


def source(paper: str, snippet: str) -> EvidenceSource:
    return EvidenceSource(
        paper_id=paper,
        paper_title=paper,
        section="Results",
        source=f"{paper}.pdf",
        chunk_id=f"{paper}-1",
        snippet=snippet,
    )


class EvidenceAndConversationTests(unittest.TestCase):
    def test_evidence_ids_dedup_and_invalid_citation(self) -> None:
        settings = load_settings()
        registry = EvidenceRegistry(settings)
        text, added = registry.register(
            ToolResult(
                text="{{cite:0}} first {{cite:1}} duplicate",
                sources=[
                    source("p", "the same evidence"),
                    source("p", "the same evidence"),
                ],
            )
        )
        self.assertEqual(len(added), 1)
        self.assertEqual(text.count("[S1]"), 2)
        check = registry.check_answer("A claim [S1]. Another [S99].")
        self.assertEqual(check.invalid, ["S99"])
        self.assertNotIn("[S99]", check.cleaned)

    def test_numeric_conflict_detection(self) -> None:
        settings = load_settings()
        registry = EvidenceRegistry(settings)
        registry.register(
            ToolResult(
                text="{{cite:0}} a {{cite:1}} b",
                sources=[
                    source("a", "model accuracy is 90 percent on the same dataset"),
                    source("b", "model accuracy is 80 percent on the same dataset"),
                ],
            )
        )
        conflicts = registry.detect_conflicts()
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["reason"], "numeric")

    def test_ambiguity_drift_and_compaction(self) -> None:
        settings = load_settings()
        settings = replace(
            settings,
            harness=replace(
                settings.harness,
                history_max_messages=4,
                recent_history_messages=2,
                history_max_chars=500,
            ),
        )
        manager = ConversationManager(settings)
        state = ConversationState()
        self.assertIsNotNone(manager.clarification_question("哪个更好？", state))
        state.goal = "比较 RAG 和长上下文模型"
        self.assertIsNotNone(manager.drift_question("继续说这个", state))
        prepared = manager.prepare(
            [
                {"role": "user", "content": f"question {index}"}
                for index in range(10)
            ]
        )
        self.assertGreater(prepared.dropped_messages, 0)
        self.assertLessEqual(
            sum(len(item["content"]) for item in prepared.history),
            settings.harness.history_max_chars
            + settings.harness.history_summary_max_chars,
        )


if __name__ == "__main__":
    unittest.main()
