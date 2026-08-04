from __future__ import annotations

import unittest
import json
from dataclasses import replace
from typing import Any, cast

from agent.evidence import AnswerVerifier, EvidenceSelector, ExecutionResult
from agent.graph import FinalComposer, PaperRAGAgent, PlanRoute, QueryPlan, RequestPlanner
from agent.memory import ConversationState
from agent.runtime import AgentExecutionError
from config.settings import BASE_DIR, load_settings
from llm.model import LLMClient
from tools.base import EvidenceSource, ToolResult


class AgentFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = load_settings()
        self.verifier = AnswerVerifier(self.settings)

    @staticmethod
    def planner(decision: dict[str, Any]) -> RequestPlanner:
        class PlannerLLM:
            @staticmethod
            def supports_agentic() -> bool:
                return True

            @staticmethod
            def generate(prompt: str, system: str = "") -> str:
                del prompt, system
                return json.dumps(decision, ensure_ascii=False)

        return RequestPlanner(cast(Any, PlannerLLM()))

    def test_planner_routes_direct_and_substantive_questions(self) -> None:
        direct = self.planner(
            {
                "route": "direct",
                "query": "",
                "answer": "你好，我可以帮助研究论文。",
                "required_tools": [],
                "goal": "",
                "answer_language": "Simplified Chinese",
            }
        ).plan(
            "你好",
            [],
            ConversationState(),
            tools_available=True,
        )
        self.assertEqual(direct.route, PlanRoute.DIRECT)
        self.assertEqual(direct.answer, "你好，我可以帮助研究论文。")

        tools_decision = {
            "route": "tools",
            "query": "Transformer 与 BERT 的预训练目标差异",
            "answer": "",
            "required_tools": ["search_local_papers"],
            "goal": "比较 Transformer 与 BERT",
            "answer_language": "Simplified Chinese",
        }
        tool_plan = self.planner(tools_decision).plan(
            "请比较 Transformer 与 BERT 的预训练目标差异",
            [],
            ConversationState(),
            tools_available=True,
        )
        self.assertEqual(tool_plan.route, PlanRoute.TOOLS)

        local_plan = self.planner(tools_decision).plan(
            "请比较 Transformer 与 BERT 的预训练目标差异",
            [],
            ConversationState(),
            tools_available=False,
        )
        self.assertEqual(local_plan.route, PlanRoute.LOCAL_RAG)

        clarify = self.planner(
            {
                "route": "clarify",
                "query": "",
                "answer": "你希望比较哪两篇论文？",
                "required_tools": [],
                "goal": "",
                "answer_language": "Simplified Chinese",
            }
        ).plan("比较一下", [], ConversationState(), tools_available=True)
        self.assertEqual(clarify.route, PlanRoute.CLARIFY)
        self.assertEqual(clarify.answer, "你希望比较哪两篇论文？")

    def test_model_decision_selects_required_tool_without_keyword_rules(self) -> None:
        planner = self.planner(
            {
                "route": "tools",
                "query": "Agentic RAG papers after the model knowledge cutoff",
                "answer": "",
                "required_tools": ["search_arxiv"],
                "goal": "追踪 Agentic RAG 新论文",
                "answer_language": "Simplified Chinese",
            }
        )
        plan = planner.plan(
            "知识截止后这个方向又出现了哪些工作？",
            [],
            ConversationState(),
            tools_available=True,
        )
        self.assertEqual(plan.required_tools, ("search_arxiv",))
        self.assertEqual(plan.trace[0]["required_tools"], ["search_arxiv"])

    def test_required_tool_failure_does_not_fallback_to_local_results(self) -> None:
        class FailingRuntime:
            @staticmethod
            def invoke(
                query: str,
                history: list[dict[str, Any]],
                *,
                required_tools: tuple[str, ...] = (),
                answer_language: str = "",
            ) -> ExecutionResult:
                del query, history, required_tools, answer_language
                raise AgentExecutionError(
                    RuntimeError("failed"),
                    ExecutionResult("", [], ["partial"], []),
                )

        class ForbiddenFallback:
            @staticmethod
            def execute(question: str, query: str) -> ExecutionResult:
                del question, query
                raise AssertionError("local fallback must not run")

        agent = object.__new__(PaperRAGAgent)
        agent.runtime = cast(Any, FailingRuntime())
        agent.fallback = cast(Any, ForbiddenFallback())
        result = agent._execute(
            QueryPlan(
                PlanRoute.TOOLS,
                "最新论文",
                "latest papers",
                required_tools=("search_arxiv",),
            ),
            [],
        )
        self.assertEqual(result.answer, "")
        self.assertTrue(
            any(item.get("type") == "required_tools" for item in result.trace)
        )

    def test_verifier_requires_evidence_before_citations(self) -> None:
        answer = self.verifier.finalize(ExecutionResult("没有依据的结论"))
        self.assertEqual(answer.status, "insufficient_evidence")
        self.assertIn("有效来源不足", answer.answer)

    def test_verifier_accepts_only_real_numbered_sources(self) -> None:
        source = EvidenceSource(
            paper_id="paper",
            paper_title="Paper",
            section="Methods",
            source=str(BASE_DIR / "data" / "papers" / "paper.pdf"),
            snippet="BERT uses masked language modeling during pretraining.",
            confidence=0.9,
            citation_id="S1",
        ).to_dict()
        accepted = self.verifier.finalize(
            ExecutionResult("BERT 使用掩码语言模型作为预训练任务 [S1]。", [source])
        )
        self.assertEqual(accepted.status, "answered")
        self.assertEqual(accepted.sources[0]["id"], "S1")

        missing = self.verifier.finalize(
            ExecutionResult("BERT 使用掩码语言模型作为预训练任务。", [source])
        )
        self.assertEqual(missing.status, "answered")
        self.assertEqual([item["id"] for item in missing.sources], ["S1"])

        hallucinated = self.verifier.finalize(
            ExecutionResult("BERT 使用掩码语言模型 [S99]。", [source])
        )
        self.assertEqual(hallucinated.status, "answered")
        self.assertNotIn("[S99]", hallucinated.answer)
        self.assertEqual([item["id"] for item in hallucinated.sources], ["S1"])

    def test_verifier_accepts_zero_confidence_source_in_lax_mode(self) -> None:
        source = EvidenceSource(
            paper_id="paper",
            paper_title="Paper",
            section="Methods",
            source=str(BASE_DIR / "data" / "papers" / "paper.pdf"),
            snippet="Retrieved evidence remains available to the answer.",
            confidence=0.0,
            citation_id="S1",
        ).to_dict()

        answer = self.verifier.finalize(ExecutionResult("直接回答。", [source]))

        self.assertEqual(answer.status, "answered")
        self.assertEqual([item["id"] for item in answer.sources], ["S1"])

    def test_lax_verifier_accepts_uncited_required_tool_source(self) -> None:
        agent = object.__new__(PaperRAGAgent)
        agent.settings = self.settings
        source = EvidenceSource(
            paper_id="paper",
            paper_title="Paper",
            section="Methods",
            source=str(BASE_DIR / "data" / "papers" / "paper.pdf"),
            snippet="Retrieved evidence.",
            origin_tools=["search_local_papers"],
            citation_id="S1",
        ).to_dict()

        missing = agent._uncited_required_tools(
            ExecutionResult("Answer without an inline citation.", [source]),
            ("search_local_papers",),
        )

        self.assertEqual(missing, ())

    def test_fast_local_path_streams_one_model_answer(self) -> None:
        class StreamingLLM:
            @staticmethod
            def stream(prompt: str, system: str = "") -> Any:
                self.assertIn("[S1]", prompt)
                self.assertIn("untrusted", system)
                yield "快速"
                yield "回答 [S1]"

        agent = object.__new__(PaperRAGAgent)
        agent.settings = self.settings
        agent.llm = cast(Any, StreamingLLM())
        agent.verifier = self.verifier
        agent.selector = EvidenceSelector(self.settings)
        agent.composer = FinalComposer(
            self.settings,
            agent.llm,
            self.verifier,
            agent.selector,
        )
        prefetched = ToolResult(
            "{{cite:0}} evidence",
            sources=[
                EvidenceSource(
                    paper_id="paper",
                    paper_title="Paper",
                    section="Methods",
                    source=str(BASE_DIR / "data" / "papers" / "paper.pdf"),
                    snippet="Fast local evidence.",
                    confidence=0.9,
                )
            ],
            metadata={"retrieval": {"total_ms": 1.0}},
        )
        chunks: list[str] = []

        result = agent._execute_fast_local(
            "question", [], prefetched, chunks.append
        )

        self.assertTrue(agent._use_fast_local(prefetched))
        self.assertEqual(result.answer, "快速回答 [S1]")
        self.assertEqual(chunks, ["快速", "回答 [S1]"])
        self.assertEqual(result.trace[0]["type"], "fast_local")

    def test_local_generation_preserves_source_ids(self) -> None:
        prompt = (
            "用户问题:\nBERT 的目标是什么？\n\n"
            "[S1] 《BERT》· Methods\nContext: masked language modeling"
        )
        output = LLMClient._generate_local(prompt)
        self.assertIn("[S1]", output)

    def test_evidence_selector_caps_and_diversifies_sources(self) -> None:
        self.settings.agent = replace(
            self.settings.agent,
            final_max_sources=3,
            final_max_sources_per_paper=1,
        )
        selector = EvidenceSelector(self.settings)
        sources = [
            EvidenceSource(
                paper_id="paper-a",
                paper_title="BERT",
                section=f"Section {index}",
                source=str(BASE_DIR / "data" / "papers" / "paper-a.pdf"),
                snippet=f"BERT masked language modeling evidence {index}",
                confidence=0.9 - index * 0.1,
                origin_tools=["search_local_papers"],
                citation_id=f"S{index}",
            ).to_dict()
            for index in (1, 2)
        ]
        sources.extend(
            [
                EvidenceSource(
                    paper_id="paper-b",
                    paper_title="Agentic RAG",
                    section="Abstract",
                    source="https://arxiv.org/abs/2601.00001",
                    snippet="Recent Agentic RAG work uses iterative retrieval.",
                    origin_tools=["search_arxiv"],
                    citation_id="S3",
                ).to_dict(),
                EvidenceSource(
                    paper_id="paper-c",
                    paper_title="Retrieval",
                    section="Methods",
                    source=str(BASE_DIR / "data" / "papers" / "paper-c.pdf"),
                    snippet="Retrieval combines dense and sparse evidence.",
                    origin_tools=["search_local_papers"],
                    citation_id="S4",
                ).to_dict(),
            ]
        )

        selected = selector.select(
            "比较 BERT 与最新 Agentic RAG",
            sources,
            required_tools=("search_arxiv",),
        )

        self.assertLessEqual(len(selected), 3)
        self.assertTrue(
            any("search_arxiv" in item["origin_tools"] for item in selected)
        )
        paper_ids = [str(item["paper_id"]) for item in selected]
        self.assertEqual(len(paper_ids), len(set(paper_ids)))

    def test_final_composer_repairs_refusal_when_evidence_is_sufficient(self) -> None:
        class ComposerLLM:
            @staticmethod
            def supports_agentic() -> bool:
                return True

            @staticmethod
            def generate(prompt: str, system: str = "") -> str:
                self.assertIn("[S1]", prompt)
                del system
                return "BERT 的预训练包含掩码语言模型任务 [S1]。"

        source = EvidenceSource(
            paper_id="paper",
            paper_title="BERT",
            section="Pre-training",
            source=str(BASE_DIR / "data" / "papers" / "paper.pdf"),
            snippet="BERT is pretrained with a masked language model objective.",
            confidence=0.9,
            origin_tools=["search_local_papers"],
            citation_id="S1",
        ).to_dict()
        composer = FinalComposer(
            self.settings,
            cast(Any, ComposerLLM()),
            self.verifier,
            EvidenceSelector(self.settings),
        )

        result = composer.compose(
            "BERT 的预训练目标是什么？",
            "Simplified Chinese",
            ("search_local_papers",),
            ExecutionResult("证据不足，无法回答。", [source]),
        )

        self.assertIn("[S1]", result.answer)
        self.assertEqual([item["id"] for item in result.sources], ["S1"])
        self.assertTrue(result.trace[-1]["repaired_draft"])

    def test_final_composer_reuses_small_valid_draft_without_model_call(self) -> None:
        class NoCallLLM:
            @staticmethod
            def supports_agentic() -> bool:
                return True

            @staticmethod
            def generate(prompt: str, system: str = "") -> str:
                del prompt, system
                raise AssertionError("a small valid draft must not add an LLM call")

        source = EvidenceSource(
            paper_id="paper",
            paper_title="BERT",
            section="Pre-training",
            source=str(BASE_DIR / "data" / "papers" / "paper.pdf"),
            snippet="BERT is pretrained with a masked language model objective.",
            confidence=0.9,
            origin_tools=["search_local_papers"],
            citation_id="S1",
        ).to_dict()
        composer = FinalComposer(
            self.settings,
            cast(Any, NoCallLLM()),
            self.verifier,
            EvidenceSelector(self.settings),
        )

        result = composer.compose(
            "BERT 的预训练目标是什么？",
            "Simplified Chinese",
            ("search_local_papers",),
            ExecutionResult("BERT 使用掩码语言模型目标 [S1]。", [source]),
        )

        self.assertEqual(result.trace[-1]["result"], "reused_agent_draft")
        self.assertEqual([item["id"] for item in result.sources], ["S1"])

    def test_final_composer_rewrites_overlong_valid_draft(self) -> None:
        class RewriteLLM:
            @staticmethod
            def supports_agentic() -> bool:
                return True

            @staticmethod
            def generate(prompt: str, system: str = "") -> str:
                del prompt, system
                return "精简结论 [S1]。"

        source = EvidenceSource(
            paper_id="paper",
            paper_title="BERT",
            section="Pre-training",
            source=str(BASE_DIR / "data" / "papers" / "paper.pdf"),
            snippet="BERT uses masked language modeling.",
            confidence=0.9,
            origin_tools=["search_local_papers"],
            citation_id="S1",
        ).to_dict()
        composer = FinalComposer(
            self.settings,
            cast(Any, RewriteLLM()),
            self.verifier,
            EvidenceSelector(self.settings),
        )

        result = composer.compose(
            "请简要回答 BERT 的目标。",
            "Simplified Chinese",
            ("search_local_papers",),
            ExecutionResult(
                "冗长说明" * self.settings.agent.final_reuse_max_chars + " [S1]。",
                [source],
            ),
        )

        self.assertEqual(result.answer, "精简结论 [S1]。")
        self.assertEqual(result.trace[-1]["result"], "composed")


if __name__ == "__main__":
    unittest.main()
