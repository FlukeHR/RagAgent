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
    FreshToolRequirementMiddleware,
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
        self.assertIn(
            f"max_results={self.settings.arxiv.max_results * 4}",
            url,
        )
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

    def test_fresh_tool_requirement_forces_only_arxiv_after_empty_prefetch(self) -> None:
        context = self.context()
        context.require_fresh_tool = True
        context.force_tool_next = True
        context.forced_tool_names = ("search_arxiv",)
        request = SimpleNamespace(
            runtime=SimpleNamespace(context=context),
            tools=[
                SimpleNamespace(name="search_local_papers"),
                SimpleNamespace(name="inspect_paper"),
                SimpleNamespace(name="search_arxiv"),
            ],
            tool_choice=None,
            system_prompt="base prompt",
        )

        def override(**changes: object) -> Any:
            return SimpleNamespace(
                runtime=request.runtime,
                tools=changes.get("tools", request.tools),
                tool_choice=changes.get("tool_choice"),
                system_message=changes.get("system_message"),
            )

        request.override = override
        received: list[Any] = []

        def handler(value: Any) -> Any:
            received.append(value)
            return value

        FreshToolRequirementMiddleware().wrap_model_call(
            cast(Any, request), handler
        )

        self.assertEqual(received[0].tool_choice, "required")
        self.assertEqual(
            [tool.name for tool in received[0].tools], ["search_arxiv"]
        )
        self.assertIn("must call", received[0].system_message.text)

    def test_empty_prefetch_exposes_only_arxiv_before_forcing(self) -> None:
        context = self.context()
        context.require_fresh_tool = True
        context.forced_tool_names = ("search_arxiv",)
        request = SimpleNamespace(
            runtime=SimpleNamespace(context=context),
            tools=[
                SimpleNamespace(name="search_local_papers"),
                SimpleNamespace(name="inspect_paper"),
                SimpleNamespace(name="search_arxiv"),
            ],
        )
        request.override = lambda **changes: SimpleNamespace(
            runtime=request.runtime,
            tools=changes.get("tools", request.tools),
        )
        received: list[Any] = []

        def handler(value: Any) -> Any:
            received.append(value)
            return value

        FreshToolRequirementMiddleware().wrap_model_call(
            cast(Any, request), handler
        )

        self.assertEqual(
            [tool.name for tool in received[0].tools], ["search_arxiv"]
        )

    def test_fresh_tool_requirement_retries_resets_and_exempts_social(self) -> None:
        middleware = FreshToolRequirementMiddleware()
        context = self.context()
        context.require_fresh_tool = True
        resets: list[str] = []
        context.reset_sink = lambda: resets.append("reset")
        runtime = SimpleNamespace(context=context)

        retry = middleware.after_model(
            {"messages": [AIMessage(content="我不知道。")]}, runtime
        )

        self.assertIsNotNone(retry)
        assert retry is not None
        self.assertEqual(retry["jump_to"], "model")
        self.assertTrue(context.force_tool_next)
        self.assertEqual(context.forced_tool_escalations, 1)
        self.assertEqual(resets, ["reset"])

        context.model_completed_tools.add("search_arxiv")
        accepted = middleware.after_model(
            {"messages": [AIMessage(content="GRPO 使用组内相对优势。 [S1]")]},
            runtime,
        )
        self.assertIsNone(accepted)

        social = self.context()
        social.require_fresh_tool = True
        direct = middleware.after_model(
            {"messages": [AIMessage(content="[[DIRECT_NO_EVIDENCE]]你好")]},
            SimpleNamespace(context=social),
        )
        self.assertIsNone(direct)
        self.assertEqual(social.trace[-1]["result"], "direct_exemption")

    def test_prefetch_does_not_count_as_model_completed_tool(self) -> None:
        context = self.context()
        self.adapter.execute(
            "search_local_papers",
            {"query": "GRPO", "prefetched": True},
            lambda **_: ToolResult("no local evidence"),
            context,
        )
        self.assertEqual(context.model_completed_tools, set())

        self.adapter.execute(
            "search_arxiv",
            {"query": "GRPO"},
            lambda **_: ToolResult("no arxiv evidence"),
            context,
        )
        self.assertEqual(context.model_completed_tools, {"search_arxiv"})

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

    def test_runtime_registers_prefetched_local_evidence_before_model(self) -> None:
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
        source = EvidenceSource(
            "paper",
            "Paper",
            "Methods",
            str(BASE_DIR / "data" / "papers" / "paper.pdf"),
            snippet="prefetched evidence",
        )

        class SuccessfulGraph:
            @staticmethod
            def invoke(payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
                self.assertIn("[S1]", payload["messages"][-1]["content"])
                self.assertEqual(len(kwargs["context"].evidence.sources), 1)
                return {"messages": [AIMessage(content="answer [S1]")]}

        runtime.graph = cast(Any, SuccessfulGraph())
        result = runtime.invoke(
            "question",
            [],
            prefetched_local=ToolResult(
                "{{cite:0}} prefetched evidence", sources=[source]
            ),
        )

        self.assertEqual(result.answer, "answer [S1]")
        self.assertEqual(result.sources[0]["id"], "S1")

    def test_grpo_without_local_evidence_records_fresh_arxiv_completion(self) -> None:
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
        source = EvidenceSource(
            "2402.03300",
            "DeepSeekMath",
            "Abstract",
            "https://arxiv.org/abs/2402.03300",
            snippet="GRPO estimates relative advantages from groups of sampled outputs.",
        )

        class ArxivGraph:
            @staticmethod
            def invoke(payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
                context = kwargs["context"]
                self.assertIn("did not meet the fast-answer threshold", payload["messages"][-1]["content"])
                self.assertEqual(context.forced_tool_names, ("search_arxiv",))
                runtime.adapter.execute(
                    "search_arxiv",
                    {"query": "GRPO"},
                    lambda **_: ToolResult("{{cite:0}} GRPO evidence", sources=[source]),
                    context,
                )
                return {
                    "messages": [
                        AIMessage(content="GRPO 使用组内相对优势估计。 [S1]")
                    ]
                }

        runtime.graph = cast(Any, ArxivGraph())
        result = runtime.invoke(
            "GRPO 是怎么做的？",
            [],
            prefetched_local=ToolResult("no local evidence"),
            require_fresh_tool=True,
        )

        self.assertIn("组内相对优势", result.answer)
        self.assertEqual(result.sources[0]["id"], "S1")
        self.assertTrue(
            any(
                item.get("type") == "fresh_tool_requirement"
                and item.get("result") == "completed"
                for item in result.trace
            )
        )

    def test_runtime_streams_final_model_text(self) -> None:
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

        class StreamingGraph:
            @staticmethod
            def stream(*args: object, **kwargs: Any) -> Any:
                del args, kwargs
                yield "messages", (AIMessage(content="streamed "), {})
                yield "messages", (
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "search_arxiv",
                                "args": {"query": "q"},
                                "id": "call-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    {},
                )
                yield "messages", (AIMessage(content="answer"), {})
                yield "values", {"messages": [AIMessage(content="streamed answer")]}

        runtime.graph = cast(Any, StreamingGraph())
        chunks: list[str] = []
        result = runtime.invoke("question", [], token_sink=chunks.append)

        self.assertEqual(chunks, ["streamed ", "answer"])
        self.assertEqual(result.answer, "streamed answer")

    def test_runtime_hides_direct_answer_marker_from_stream(self) -> None:
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

        class DirectGraph:
            @staticmethod
            def stream(*args: object, **kwargs: Any) -> Any:
                del args, kwargs
                yield "messages", (AIMessage(content="[[DIRECT_"), {})
                yield "messages", (AIMessage(content="NO_EVIDENCE]]你好"), {})
                yield "values", {
                    "messages": [
                        AIMessage(content="[[DIRECT_NO_EVIDENCE]]你好")
                    ]
                }

        runtime.graph = cast(Any, DirectGraph())
        chunks: list[str] = []
        result = runtime.invoke("hello", [], token_sink=chunks.append)

        self.assertEqual(chunks, ["你好"])
        self.assertEqual(result.answer, "你好")
        self.assertEqual(result.trace[-1]["type"], "direct")

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
        context.settings = replace(
            context.settings,
            retrieval=replace(
                context.settings.retrieval,
                answerability_require_citation=True,
                max_corrections=1,
            ),
        )
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
