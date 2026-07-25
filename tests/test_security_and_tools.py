from __future__ import annotations

import time
import unittest
from tempfile import TemporaryDirectory

from agent.evidence import EvidenceRegistry
from agent.harness import ToolHarness
from config.settings import load_settings
from llm.model import ToolCall
from retrieval.repository import (
    InvalidPaperId,
    PaperRepository,
    arxiv_storage_id,
    normalize_arxiv_id,
)
from tools.base import ToolPolicy, ToolRegistry, ToolResult


class SlowWriteTool:
    name = "slow_write"
    policy = ToolPolicy(
        timeout_seconds=0.15,
        max_retries=0,
        side_effects="write",
        isolate_process=True,
    )

    @staticmethod
    def schema() -> dict:
        return {
            "name": "slow_write",
            "description": "test",
            "input_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            },
        }

    @staticmethod
    def run() -> ToolResult:
        time.sleep(5)
        return ToolResult(text="late")


class SecurityAndToolTests(unittest.TestCase):
    def test_repository_rejects_path_escape(self) -> None:
        with TemporaryDirectory() as directory:
            repository = PaperRepository(directory)
            for value in ("../secret", "..\\secret", "/absolute", "a/b", ""):
                with self.subTest(value=value), self.assertRaises(InvalidPaperId):
                    repository.resolve(value)

    def test_arxiv_id_validation_and_flat_storage(self) -> None:
        self.assertEqual(normalize_arxiv_id("2401.12345v2"), "2401.12345v2")
        self.assertEqual(arxiv_storage_id("hep-th/9901001"), "hep-th__9901001")
        for value in ("../../x", "https://arxiv.org/abs/2401.1", "bad"):
            with self.subTest(value=value), self.assertRaises(InvalidPaperId):
                normalize_arxiv_id(value)

    def test_write_tool_is_hard_terminated_on_timeout(self) -> None:
        settings = load_settings()
        registry = ToolRegistry([SlowWriteTool()])
        evidence = EvidenceRegistry(settings)
        harness = ToolHarness(settings, registry, evidence)
        trace: list[dict] = []
        started = time.perf_counter()
        result = harness.execute(
            ToolCall(id="1", name="slow_write", input={}),
            [],
            trace,
        )
        elapsed = time.perf_counter() - started
        self.assertTrue(result.is_error)
        self.assertLess(elapsed, 2.0)
        self.assertEqual(trace[0]["attempts"], 1)

    def test_unknown_schema_fields_are_rejected(self) -> None:
        settings = load_settings()
        registry = ToolRegistry([SlowWriteTool()])
        harness = ToolHarness(settings, registry, EvidenceRegistry(settings))
        result = harness.execute(
            ToolCall(id="1", name="slow_write", input={"unexpected": True}),
            [],
            [],
        )
        self.assertTrue(result.is_error)
        self.assertIn("入参不合法", result.content)


if __name__ == "__main__":
    unittest.main()
