from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from langchain.messages import AIMessage, HumanMessage
from langchain.tools import ToolRuntime
from pydantic import ValidationError

from agent.evidence import EvidenceRegistry
from agent.runtime import (
    AgentExecutionError,
    AgentRunContext,
    AnswerVerificationMiddleware,
    InspectPaperArgs,
    LangChainAgentRuntime,
    RequestBudgetMiddleware,
    PlannerContractMiddleware,
    SearchLocalPapersArgs,
    ToolResultAdapter,
)
from config.settings import BASE_DIR, load_settings
from llm.model import LLMClient
from services.arxiv_service import ArxivPaper, ArxivSearchService
from tools.arxiv_tool import ArxivTool
from tools.base import EvidenceSource, ToolResult


class _StubTool:
    def run(self, **arguments: object) -> ToolResult:
        del arguments
        return ToolResult(text="safe")


class _ArxivSearchStub:
    @staticmethod
    def search(query: str, max_results: int | None = None) -> list[ArxivPaper]:
        del query, max_results
        return [
            ArxivPaper(
                arxiv_id="2601.00001",
                title="Paper",
                authors=("Author",),
                summary="Abstract",
                published=datetime(2026, 1, 1, tzinfo=timezone.utc),
                entry_url="https://arxiv.org/abs/2601.00001",
            )
        ]


class AgentRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = load_settings()
        self.adapter = ToolResultAdapter(self.settings)

    def context(self) -> AgentRunContext:
        return AgentRunContext(
            settings=self.settings,
            evidence=EvidenceRegistry(self.settings),
        )

    def test_only_three_read_tools_are_registered(self) -> None:
        local_llm = replace(
            self.settings.llm,
            openai_api_base="http://127.0.0.1:9",
            openai_api_key="",
        )
        settings = replace(self.settings, llm=local_llm)
        stub = _StubTool()
        runtime = LangChainAgentRuntime(
            settings,
            LLMClient(local_llm),
            cast(Any, stub),
            cast(Any, stub),
            cast(Any, stub),
        )
        self.assertEqual(
            runtime.tool_names,
            {"search_local_papers", "inspect_paper", "search_arxiv"},
        )
        self.assertTrue(all("runtime" not in tool.args for tool in runtime.tools))
        self.assertNotIn("ingest_arxiv_papers", runtime.tool_names)

    def test_arxiv_search_is_read_only_and_returns_candidate_ids(self) -> None:
        search = ArxivTool(self.settings)
        search.service = cast(Any, _ArxivSearchStub())
        result = search.run("paper")
        self.assertEqual(result.metadata["candidate_arxiv_ids"], ["2601.00001"])
        self.assertNotIn("proposal_id", result.metadata)

    def test_arxiv_fetches_relevant_pool_then_sorts_it_by_date(self) -> None:
        payload = b"""<feed xmlns="http://www.w3.org/2005/Atom">
          <entry><id>https://arxiv.org/abs/2601.00001</id><title>Older</title>
            <published>2026-01-01T00:00:00Z</published><summary>old</summary></entry>
          <entry><id>https://arxiv.org/abs/2602.00001</id><title>Newer</title>
            <published>2026-02-01T00:00:00Z</published><summary>new</summary></entry>
        </feed>"""
        response = SimpleNamespace(
            content=payload,
            raise_for_status=lambda: None,
        )
        with patch("requests.get", return_value=response) as get:
            papers = ArxivSearchService(self.settings).search("agentic rag")
        url = str(get.call_args.args[0])
        self.assertIn("sortBy=relevance", url)
        self.assertIn("max_results=20", url)
        self.assertEqual([paper.title for paper in papers], ["Newer", "Older"])

    def test_planner_contract_does_not_force_provider_tool_choice(self) -> None:
        context = self.context()
        context.required_tools = ("inspect_paper", "search_arxiv")
        context.answer_language = "Traditional Chinese"
        request = SimpleNamespace(
            runtime=SimpleNamespace(context=context),
            tool_choice=None,
            system_prompt="base prompt",
        )

        def override(**changes: object) -> Any:
            return SimpleNamespace(
                runtime=request.runtime,
                tool_choice=changes.get("tool_choice"),
                system_message=changes.get("system_message"),
            )

        request.override = override
        received: list[Any] = []

        def handler(value: Any) -> Any:
            received.append(value)
            return value

        PlannerContractMiddleware().wrap_model_call(cast(Any, request), handler)
        self.assertIsNone(received[0].tool_choice)
        self.assertIn("inspect_paper", received[0].system_message.text)
        self.assertIn("search_arxiv", received[0].system_message.text)
        self.assertIn("Traditional Chinese", received[0].system_message.text)

    def test_runtime_uses_graph_step_budget_and_preserves_partial_error(self) -> None:
        local_llm = replace(
            self.settings.llm,
            openai_api_base="http://127.0.0.1:9",
            openai_api_key="",
        )
        settings = replace(self.settings, llm=local_llm)
        stub = _StubTool()
        runtime = LangChainAgentRuntime(
            settings,
            LLMClient(local_llm),
            cast(Any, stub),
            cast(Any, stub),
            cast(Any, stub),
        )

        class FailingGraph:
            config: dict[str, Any] = {}

            @classmethod
            def invoke(cls, *args: object, **kwargs: Any) -> dict[str, Any]:
                del args
                cls.config = kwargs["config"]
                kwargs["context"].steps.append("before failure")
                raise RuntimeError("boom")

        runtime.graph = cast(Any, FailingGraph())
        with self.assertRaises(AgentExecutionError) as raised:
            runtime.invoke("question", [], required_tools=("search_arxiv",))
        self.assertEqual(
            FailingGraph.config["recursion_limit"], settings.agent.max_graph_steps
        )
        self.assertIn("before failure", raised.exception.partial.steps)
        self.assertEqual(
            raised.exception.partial.trace[-1]["type"], "agent_runtime_error"
        )

    def test_pydantic_schemas_reject_extra_fields_and_invalid_locators(self) -> None:
        injected = ToolRuntime(
            state={"messages": []},
            context=self.context(),
            config={},
            stream_writer=lambda _: None,
            tool_call_id="test",
            store=None,
        )
        parsed = SearchLocalPapersArgs.model_validate(
            {"query": "x", "runtime": injected}
        )
        self.assertIs(parsed.runtime, injected)
        with self.assertRaises(ValidationError):
            SearchLocalPapersArgs.model_validate({"query": "x", "write": True})
        for locator in (
            {"kind": "page", "page_number": 0},
            {"kind": "region", "page_number": 1, "bbox": [0, 0, 1001, 1]},
            {"kind": "region", "page_number": 1, "bbox": [10, 10, 5, 20]},
            {"kind": "section", "section": "A", "page_number": 1},
        ):
            with self.subTest(locator=locator), self.assertRaises(ValidationError):
                InspectPaperArgs.model_validate({"paper_id": "paper", "locator": locator})

    def test_result_character_and_source_budgets_are_enforced(self) -> None:
        tight_agent = replace(
            self.settings.agent,
            tool_result_max_chars=3,
            max_total_tool_result_chars=3,
            max_total_sources=1,
        )
        self.settings = replace(self.settings, agent=tight_agent)
        adapter = ToolResultAdapter(self.settings)
        context = self.context()
        with self.assertRaises(ValueError):
            adapter.execute("search_local_papers", {}, lambda: ToolResult("long"), context)

        source_path = str(BASE_DIR / "data" / "papers" / "paper.pdf")
        sources = [
            EvidenceSource("p1", "P1", "S", source_path),
            EvidenceSource("p2", "P2", "S", source_path),
        ]
        with self.assertRaises(ValueError):
            adapter.execute(
                "search_local_papers",
                {},
                lambda: ToolResult("ok", sources=sources),
                self.context(),
            )

    def test_parallel_duplicate_search_is_rejected_before_handler(self) -> None:
        context = self.context()
        context.active_searches.add("search_local_papers")

        with self.assertRaisesRegex(ValueError, "parallel duplicate search"):
            self.adapter.execute(
                "search_local_papers",
                {"query": "BERT"},
                lambda **_: (_ for _ in ()).throw(
                    AssertionError("duplicate handler must not run")
                ),
                context,
            )

        self.assertIn("search_local_papers", context.active_searches)
        self.assertEqual(context.trace[-1]["tool"], "search_local_papers")

    def test_malicious_path_base64_and_forged_placeholder_are_rejected(self) -> None:
        local_path = str(BASE_DIR / "data" / "papers" / "paper.pdf")
        bad_results = (
            ToolResult("{{cite:1}} forged"),
            ToolResult(
                "bad",
                sources=[EvidenceSource("p", "P", "S", str(BASE_DIR.parent / "x.pdf"))],
            ),
            ToolResult(
                "bad",
                sources=[
                    EvidenceSource(
                        "p", "P", "S", local_path, parser_metadata={"base64": "secret"}
                    )
                ],
            ),
        )
        for result in bad_results:
            with self.subTest(result=result), self.assertRaises(ValueError):
                self.adapter.execute(
                    "search_local_papers", {}, lambda value=result: value, self.context()
                )

    def test_trace_redacts_sensitive_values(self) -> None:
        context = self.context()
        self.adapter.execute(
            "search_local_papers",
            {"query": "safe", "api_token": "secret", "image_base64": "payload"},
            lambda **_: ToolResult("safe"),
            context,
        )
        event = context.trace[0]
        self.assertEqual(event["input"]["api_token"], "[REDACTED]")
        self.assertEqual(event["input"]["image_base64"], "[REDACTED]")

    def test_evidence_records_and_merges_origin_tools(self) -> None:
        context = self.context()
        source_path = str(BASE_DIR / "data" / "papers" / "paper.pdf")

        def result() -> ToolResult:
            return ToolResult(
                "{{cite:0}} evidence",
                sources=[
                    EvidenceSource(
                        "paper",
                        "Paper",
                        "Methods",
                        source_path,
                        snippet="same evidence",
                    )
                ],
            )

        self.adapter.execute("search_local_papers", {}, result, context)
        self.adapter.execute("inspect_paper", {}, result, context)
        source = context.evidence.as_dicts()[0]
        self.assertEqual(
            source["origin_tools"], ["search_local_papers", "inspect_paper"]
        )

    def test_token_budget_and_citation_correction_are_bounded(self) -> None:
        verifier = AnswerVerificationMiddleware()
        context = self.context()
        runtime = SimpleNamespace(context=context)
        first = verifier.after_model(
            {"messages": [AIMessage(content="unsupported")]}, runtime
        )
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(first["jump_to"], "model")
        second = verifier.after_model(
            {"messages": [AIMessage(content="still unsupported")]}, runtime
        )
        self.assertIsNone(second)
        self.assertEqual(context.corrections, 1)

        tiny = replace(self.settings, agent=replace(self.settings.agent, token_budget=1))
        tiny_context = AgentRunContext(tiny, EvidenceRegistry(tiny))
        stopped = RequestBudgetMiddleware().before_model(
            {"messages": [HumanMessage(content="question")]},
            SimpleNamespace(context=tiny_context),
        )
        self.assertIsNotNone(stopped)
        assert stopped is not None
        self.assertEqual(stopped["jump_to"], "end")

    def test_lexical_support_warning_alone_does_not_trigger_more_tools(self) -> None:
        context = self.context()
        context.required_tools = ("search_arxiv",)
        source = EvidenceSource(
            "2601.00001",
            "Paper",
            "Abstract",
            "https://arxiv.org/abs/2601.00001",
            snippet="An English abstract about retrieval.",
        )
        self.adapter.execute(
            "search_arxiv",
            {},
            lambda: ToolResult("{{cite:0}} evidence", sources=[source]),
            context,
        )
        result = AnswerVerificationMiddleware().after_model(
            {"messages": [AIMessage(content="完全不同语言的陈述 [S1]。")]},
            SimpleNamespace(context=context),
        )
        self.assertIsNone(result)
        self.assertEqual(context.corrections, 0)


if __name__ == "__main__":
    unittest.main()
