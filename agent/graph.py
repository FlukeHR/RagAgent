from __future__ import annotations

import math
import re
import time
from agent.conversation import ConversationManager, ConversationState
from agent.prompts import SYSTEM_PROMPT
from agent.evidence import EvidenceRegistry
from agent.fallback import FallbackRAG
from agent.harness import ToolHarness
from agent.memory import ConversationMemory
from agent.verifier import AnswerVerifier
from agent.state import AgentAnswer
from config.settings import Settings
from llm.model import LLMClient
from llm.prompt_builder import build_generation_prompt
from tools import (
    ArxivIngestTool,
    ArxivTool,
    ImageSearchTool,
    PDFPageTool,
    PDFRegionTool,
    PaperReaderTool,
    PaperSearchTool,
    ToolRegistry,
)


def _sigmoid(x: float) -> float:
    """把 reranker 分数（CE logit 等）归一化为 0~1 相关概率；做数值钳制避免溢出。"""
    x = max(-30.0, min(30.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


class PaperRAGAgent:
    """论文问答 Agent。

    - OpenAI-compatible 后端支持 function calling 时，走 agentic 循环
      （多源检索 + 自我纠错 + 引用溯源）。
    - 否则：降级为传统单跳 RAG（本地检索 + 拼接生成）。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = LLMClient(settings.llm)
        self.conversation = ConversationManager(settings)
        self.memory = ConversationMemory(settings)
        self.search_tool = PaperSearchTool(settings)
        self.fallback = FallbackRAG(settings, self.search_tool.retriever)
        self.arxiv_tool = ArxivTool(settings)
        self.arxiv_ingest_tool = ArxivIngestTool(settings)
        self.reader_tool = PaperReaderTool(settings)
        self.pdf_page_tool = PDFPageTool(settings)
        self.pdf_region_tool = PDFRegionTool(settings)
        self.image_search_tool = ImageSearchTool(settings)
        self.tool_registry = ToolRegistry(
            [
                self.search_tool,
                self.arxiv_tool,
                self.arxiv_ingest_tool,
                self.reader_tool,
                self.pdf_page_tool,
                self.pdf_region_tool,
                self.image_search_tool,
            ]
        )

    def ask(
        self,
        question: str,
        history: list[dict] | None = None,
        session_id: str | None = None,
    ) -> AgentAnswer:
        """回答一个问题。history 为本会话之前的对话轮次（{role, content} 纯文本）。

        有历史时先做查询改写 / 指代消解，得到独立可检索的问题；两条路径都用它检索，
        agentic 路径还会把历史注入工作上下文，使多轮对话连贯。
        """
        started = time.perf_counter()
        self.llm.consume_usage_events()
        state = ConversationState()
        stored_history: list[dict] = []
        if session_id:
            state, stored_history = self.memory.load(session_id)
        raw_history = history if history is not None else stored_history
        if not state.goal:
            state.goal = next(
                (
                    str(item.get("content") or "")[:500]
                    for item in raw_history or []
                    if item.get("role") == "user" and item.get("content")
                ),
                "",
            )
        prepared = self.conversation.prepare(raw_history or [])
        bounded_history = prepared.history
        smalltalk = self.conversation.smalltalk_response(question)
        if smalltalk:
            result = AgentAnswer(
                answer=smalltalk,
                steps=["Router: 日常对话直接响应，未启动论文检索"],
                trace=[{"type": "route", "intent": "smalltalk"}],
            )
            return self._finish_request(
                result,
                question,
                raw_history or [],
                state,
                prepared.summary,
                session_id,
                started,
            )
        standalone = self._rewrite_query(question, bounded_history)
        clarification = self.conversation.clarification_question(standalone, state)
        drift = self.conversation.drift_question(question, state)
        if clarification or drift:
            result = AgentAnswer(
                answer=drift or clarification or "",
                status="needs_clarification",
                steps=["Clarifier: 问题信息不足或多轮目标可能偏移，暂停检索"],
                trace=[
                    {
                        "type": "clarification",
                        "reason": "conversation_drift" if drift else "ambiguous_question",
                    }
                ],
            )
            return self._finish_request(
                result,
                question,
                raw_history or [],
                state,
                prepared.summary,
                session_id,
                started,
            )
        if self.llm.supports_agentic():
            try:
                result = self._ask_agentic(question, bounded_history, standalone)
            except Exception as exc:  # noqa: BLE001 - 后端不可用时降级，不让请求直接失败
                result = self._ask_fallback(question, standalone)
                result.steps.insert(0, f"⚠️ agentic 后端调用失败，已降级为本地检索: {exc}")
        else:
            result = self._ask_fallback(question, standalone)
            issue = self.llm.configuration_issue()
            if issue:
                result.steps.insert(0, f"LLM: {issue}，使用本地检索降级路径")
                result.trace.insert(
                    0,
                    {"type": "llm_unavailable", "reason": issue},
                )
        if prepared.dropped_messages:
            result.trace.insert(
                0,
                {
                    "type": "context_compaction",
                    "dropped_messages": prepared.dropped_messages,
                    "summary_chars": len(prepared.summary),
                },
            )
        return self._finish_request(
            result,
            question,
            raw_history or [],
            state,
            prepared.summary,
            session_id,
            started,
        )

    def _finish_request(
        self,
        result: AgentAnswer,
        question: str,
        history: list[dict],
        state: ConversationState,
        summary: str,
        session_id: str | None,
        started: float,
    ) -> AgentAnswer:
        state = self.conversation.update_state(
            state,
            question,
            result.sources,
            summary,
        )
        if session_id:
            stored = [
                *history,
                {"role": "user", "content": question},
                {"role": "assistant", "content": result.answer},
            ]
            compacted = self.conversation.prepare(stored).history
            self.memory.save(session_id, state, compacted)
        result.trace.append(
            {
                "type": "request",
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                "status": result.status,
            }
        )
        auxiliary = self.llm.consume_usage_events()
        for event in auxiliary:
            result.trace.append({"type": "llm_aux", **event})
        input_tokens = sum(
            int(event.get("tokens_in", event.get("input_tokens", 0)) or 0)
            for event in result.trace
            if event.get("type") in {"llm", "llm_aux"}
        )
        output_tokens = sum(
            int(event.get("tokens_out", event.get("output_tokens", 0)) or 0)
            for event in result.trace
            if event.get("type") in {"llm", "llm_aux"}
        )
        evaluation = self.settings.evaluation
        estimated_cost = (
            input_tokens / 1_000_000 * evaluation.input_price_per_million
            + output_tokens / 1_000_000 * evaluation.output_price_per_million
        )
        result.trace.append(
            {
                "type": "usage",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost": round(estimated_cost, 8),
            }
        )
        return result

    def _rewrite_query(self, question: str, history: list[dict]) -> str:
        """指代消解 / 查询改写（检索链路 §1）：把依赖上下文的口语问题改写成独立问题。

        仅在有历史且有真实 LLM 后端时触发；无后端 / 失败 / 结果异常时优雅降级回原问题。
        """
        if not history or not self.llm.supports_agentic():
            return question
        convo = "\n".join(
            f"{'用户' if t.get('role') == 'user' else '助手'}：{(t.get('content') or '')[:500]}"
            for t in history[-6:]
        )
        prompt = (
            "下面是对话历史和用户的最新问题。请把最新问题改写成一个【不依赖上下文、可独立检索】"
            "的完整问题：消解其中的指代（它/该方法/这个模型/上文等），补全省略的主语。"
            "只输出改写后的问题本身，不要解释、不要加引号。若问题本身已经独立完整，原样输出。\n\n"
            f"对话历史：\n{convo}\n\n最新问题：{question}\n\n改写后的独立问题："
        )
        try:
            out = self.llm.generate(prompt, system="你是检索查询改写助手，只输出改写后的问题。")
        except Exception:  # noqa: BLE001 - 改写失败不阻断主流程，回退原问题
            return question
        out = (out or "").strip().strip("\"'「」“”")
        # 结果为空 / 过长 / 命中本地降级文案，视为不可用，回退原问题
        if not out or len(out) > 300 or "降级模式" in out or "调用失败" in out:
            return question
        return out

    # ---------- agentic 主路径 ----------
    def _tool_schemas_for(self, question: str, trace: list[dict]) -> list[dict]:
        """Progressively expose declarative tool profiles while retaining harness control."""

        allowed = {"search_local_papers", "read_paper_section"}
        lowered = question.lower()
        explicit_ingest = bool(
            re.search(
                r"(?:\d{4}\.\d{4,5}|[a-z][a-z0-9._-]*/\d{7})(?:v\d+)?",
                lowered,
            )
            and any(term in lowered for term in ("下载", "入库", "全文", "ingest"))
        )
        if any(
            term in lowered
            for term in ("最新", "近期", "arxiv", "state of the art", "sota")
        ):
            allowed.add("search_arxiv")
        if explicit_ingest:
            allowed.add("ingest_arxiv_papers")
        if any(
            term in lowered
            for term in ("第", "页", "图", "表", "公式", "bbox", "image", "figure")
        ):
            allowed.update(
                {"read_pdf_page", "read_pdf_region", "search_pdf_images"}
            )
        # After one bounded tool observation, allow strategy expansion. Atomic tools
        # remain visible and every call still passes through ToolHarness.
        if any(event.get("type") == "tool" for event in trace):
            allowed = self.tool_registry.names()
        searched_arxiv = any(
            event.get("type") == "tool"
            and event.get("tool") == "search_arxiv"
            and event.get("ok")
            for event in trace
        )
        if not searched_arxiv and not explicit_ingest:
            allowed.discard("ingest_arxiv_papers")
        return self.tool_registry.schemas(allowed)

    def _ask_agentic(
        self,
        question: str,
        prior: list[dict] | None = None,
        standalone: str | None = None,
    ) -> AgentAnswer:
        prior = prior or []
        standalone = standalone or question
        steps: list[str] = ["Planner: 启动 agentic 检索循环(openai-compatible)"]
        sources: list[dict] = []
        evidence = EvidenceRegistry(self.settings)
        verifier = AnswerVerifier(evidence)
        harness = ToolHarness(self.settings, self.tool_registry, evidence)
        trace: list[dict] = []  # 结构化可观测事件（护栏 #6）
        if prior:
            steps.append(f"Context: 注入 {len(prior)} 条历史轮次")
            trace.append({"type": "context", "history_turns": len(prior)})
        if standalone != question:
            steps.append(f"Rewriter: 指代消解 → {standalone}")
            trace.append({"type": "rewrite", "original": question, "standalone": standalone})
        # 历史注入工作上下文 + 用改写后的独立问题作为当前轮检索种子
        history = self.llm.init_history(standalone, prior=prior)
        answer = ""
        corrections = 0
        max_corrections = self.settings.retrieval.max_corrections
        total_tokens = 0
        budget = self.settings.harness.token_budget
        budget_hit = False

        for hop in range(self.settings.llm.max_tool_iters):
            self._bound_work_history(history, trace)
            estimated_input = max(
                1,
                sum(len(str(message.get("content") or "")) for message in history)
                // 4,
            )
            if (
                total_tokens
                + estimated_input
                + self.settings.llm.max_tokens
                >= budget
            ):
                budget_hit = True
                steps.append(
                    f"Budget: 调用前预估将超过 token 预算 {budget}，停止多跳"
                )
                trace.append(
                    {
                        "type": "budget",
                        "phase": "preflight",
                        "estimated_input_tokens": estimated_input,
                        "total_tokens": total_tokens,
                        "budget": budget,
                    }
                )
                break
            turn = self.llm.create_turn(
                SYSTEM_PROMPT,
                history,
                self._tool_schemas_for(standalone, trace),
            )
            tin = int(turn.usage.get("input_tokens", 0) or 0)
            tout = int(turn.usage.get("output_tokens", 0) or 0)
            total_tokens += tin + tout
            trace.append(
                {"type": "llm", "step": hop + 1, "tokens_in": tin,
                 "tokens_out": tout, "total_tokens": total_tokens, "stop": turn.stop,
                 "duration_ms": turn.usage.get("duration_ms", 0)}
            )
            self.llm.append_assistant(history, turn)

            if not turn.stop and turn.tool_calls:
                # token 预算用尽（护栏 #2）：停止多跳，转总结，避免无限/超预算
                if total_tokens >= budget:
                    budget_hit = True
                    steps.append(f"Budget: token 预算 {budget} 已用尽（累计 {total_tokens}），停止多跳")
                    trace.append({"type": "budget", "total_tokens": total_tokens, "budget": budget})
                    break
                outcomes = [harness.execute(tc, steps, trace) for tc in turn.tool_calls]
                conflicts = verifier.conflicts()
                sources = evidence.as_dicts()
                # OpenAI-compatible APIs require every assistant tool_call to be
                # answered by its tool message immediately. Conflict feedback is
                # a user message, so it must be appended only after all outcomes.
                self.llm.append_tool_results(history, outcomes)
                if conflicts:
                    trace.append({"type": "evidence_conflict", "items": conflicts})
                    steps.append(f"Evidence: 检测到 {len(conflicts)} 组潜在冲突，要求显式说明分歧")
                    history.append(
                        {
                            "role": "user",
                            "content": (
                                "系统检测到检索来源间可能存在数值或否定关系冲突。"
                                "请分别陈述各来源结论和适用条件；无法消解时降低结论强度，"
                                "不得强行合并为单一结论。冲突记录："
                                f"{conflicts}"
                            ),
                        }
                    )
                self._bound_work_history(history, trace)
                continue

            # 模型给出最终答案 -> 生成后回查（护栏 #3）+ 低置信判断（护栏 #4）
            citation_check = verifier.verify(turn.text)
            cleaned = citation_check.cleaned
            valid_cited = citation_check.valid
            hallucinated = citation_check.invalid
            unsupported = citation_check.unsupported_claims
            low_conf = self._is_low_confidence(sources)
            # 核查失败或低置信，且仍有纠错预算时：反馈给模型，触发有界二次检索
            if (hallucinated or unsupported or low_conf) and corrections < max_corrections:
                corrections += 1
                reason_parts = []
                if hallucinated:
                    reason_parts.append(f"编造引用 {hallucinated}")
                if low_conf:
                    reason_parts.append("召回置信偏低")
                if unsupported:
                    reason_parts.append(
                        f"引用与陈述词面支持不足 {len(unsupported)} 条"
                    )
                steps.append(
                    f"Verify: 触发第 {corrections} 次二次检索"
                    f"（{'，'.join(reason_parts)}）"
                )
                trace.append(
                    {"type": "verify", "result": "recheck", "correction": corrections,
                     "hallucinated": hallucinated, "unsupported": unsupported,
                     "low_conf": low_conf}
                )
                history.append(
                    {"role": "user", "content": self._correction_feedback(hallucinated, low_conf)}
                )
                continue

            answerable, reason = self._answerability_status(sources, valid_cited)
            if not answerable:
                steps.append(f"Answerability: {reason}，拒绝生成实质答案")
                trace.append({"type": "answerability", "result": "reject", "reason": reason})
                answer = self._insufficient_evidence_answer(reason, sources)
                break

            steps.append(f"Generator: 第 {hop + 1} 轮生成最终答案")
            answer = self._finalize(cleaned, valid_cited, hallucinated, sources, steps)
            trace.append(
                {"type": "verify", "result": "final", "valid": valid_cited, "hallucinated": hallucinated}
            )
            break
        else:
            steps.append("Generator: 达到最大轮次，基于已检索内容生成总结")
            answer = self._summarize(question, sources, steps, trace)

        if budget_hit:
            steps.append("Generator: 预算用尽，基于已检索内容生成总结")
            answer = self._summarize(question, sources, steps, trace)

        return AgentAnswer(
            answer=answer or "未能生成答案。",
            steps=steps,
            sources=sources,  # 保留每条带唯一 id 的来源，前端按 [S编号] 映射
            trace=trace,
        )

    def _bound_work_history(
        self,
        history: list[dict],
        trace: list[dict],
    ) -> None:
        """Bound message content while preserving tool-call/result ordering."""

        limit = self.settings.harness.context_max_chars
        total = sum(len(str(message.get("content") or "")) for message in history)
        if total <= limit:
            return
        original = total
        per_tool = max(1000, self.settings.harness.tool_result_max_chars // 2)
        for message in history:
            content = str(message.get("content") or "")
            if message.get("role") == "tool" and len(content) > per_tool:
                message["content"] = content[:per_tool] + "\n[tool result truncated]"
        total = sum(len(str(message.get("content") or "")) for message in history)
        for message in history:
            if total <= limit:
                break
            content = str(message.get("content") or "")
            if not content:
                continue
            keep = max(200, len(content) - (total - limit))
            message["content"] = content[-keep:]
            total = sum(len(str(item.get("content") or "")) for item in history)
        trace.append(
            {
                "type": "context_budget",
                "before_chars": original,
                "after_chars": total,
                "limit": limit,
            }
        )

    def _summarize(
        self, question: str, sources: list[dict], steps: list[str], trace: list[dict]
    ) -> str:
        """轮次/预算用尽时的兜底总结，仍走引用回查。"""
        summary = self.llm.generate(self._summary_prompt(question, sources))
        cleaned, valid_cited, hallucinated = self._check_citations(summary, sources)
        ans = self._finalize(cleaned, valid_cited, hallucinated, sources, steps)
        trace.append(
            {"type": "verify", "result": "final", "valid": valid_cited, "hallucinated": hallucinated}
        )
        return ans

    @staticmethod
    def _check_citations(
        answer: str, sources: list[dict]
    ) -> tuple[str, list[str], list[str]]:
        """提取并回查答案里的 [S编号]（护栏 #3，核心不变量 #2）。

        返回 (剔除编造引用后的答案, 有效引用编号, 编造引用编号)。
        编造引用 = 未对应任何真实召回来源的编号；从正文剔除，绝不放行。
        """
        answer = answer or "未能生成答案。"
        valid_ids = {s.get("id") for s in sources}
        cited = re.findall(r"\[(S\d+)\]", answer)
        order = lambda sid: int(sid[1:])  # noqa: E731
        hallucinated = sorted({c for c in cited if c not in valid_ids}, key=order)
        valid_cited = sorted({c for c in cited if c in valid_ids}, key=order)

        cleaned = answer
        for bad in hallucinated:
            cleaned = cleaned.replace(f"[{bad}]", "")
        return cleaned, valid_cited, hallucinated

    def _is_low_confidence(self, sources: list[dict]) -> bool:
        """低置信判据（护栏 #4）：分数经 sigmoid 归一化为相关概率后做强度+数量双判据。

        判为低置信（→ 触发二次检索）当：最高分概率 < 强阈值，或 概率≥弱阈值的来源不足
        min_confident_sources 条。比单点 max<阈值 更稳，且与 reranker 量纲解耦。
        仅 arxiv 摘要 / 章节精读这类无可比分数（score=None）的来源不据此判定。
        """
        confidences = [
            float(s["confidence"])
            for s in sources
            if s.get("confidence") is not None
        ]
        if not confidences:
            scores = [s["score"] for s in sources if s.get("score") is not None]
            confidences = [_sigmoid(score) for score in scores]
        if not confidences:
            return False
        rcfg = self.settings.retrieval
        top_ok = max(confidences) >= rcfg.low_confidence_threshold
        count_ok = (
            sum(
                confidence >= rcfg.weak_confidence_threshold
                for confidence in confidences
            )
            >= rcfg.min_confident_sources
        )
        return not (top_ok and count_ok)

    def _answerability_status(
        self, sources: list[dict], valid_cited: list[str] | None = None
    ) -> tuple[bool, str]:
        """Hard no-answer gate after bounded retrieval/correction.

        Vector search always returns nearest neighbors; this gate decides whether
        those neighbors are sufficient evidence for a substantive answer.
        """
        rcfg = self.settings.retrieval
        effective = [
            s
            for s in sources
            if s.get("snippet") and self._source_meets_answerability_score(s)
        ]
        if len(effective) < rcfg.answerability_min_sources:
            return False, f"有效来源不足 {rcfg.answerability_min_sources} 条"
        if valid_cited is not None and rcfg.answerability_require_citation and not valid_cited:
            return False, "答案没有可回查引用"
        return True, "证据充分"

    def _source_meets_answerability_score(self, source: dict) -> bool:
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

    @staticmethod
    def _insufficient_evidence_answer(reason: str, sources: list[dict]) -> str:
        if not sources:
            return f"未检索到充分依据，无法可靠回答这个问题。（原因：{reason}）"
        ids = ", ".join(str(s.get("id")) for s in sources[:3] if s.get("id"))
        suffix = f" 已检索到的候选来源（{ids}）不足以支撑实质结论。" if ids else ""
        return f"未检索到充分依据，无法可靠回答这个问题。（原因：{reason}）{suffix}"

    def _correction_feedback(self, hallucinated: list[str], low_conf: bool) -> str:
        """二次检索反馈：引导模型改写查询或转 arXiv，仅用真实来源重写答案。"""
        parts = ["【系统核查反馈，请据此修正后重新作答】"]
        if hallucinated:
            parts.append(
                f"- 你引用了 {', '.join(hallucinated)}，但这些编号不在任何检索结果中，"
                "属于编造引用，严禁如此。"
            )
        if low_conf:
            parts.append(
                f"- 本地召回最高相关度低于阈值 "
                f"{self.settings.retrieval.low_confidence_threshold}，证据可能不足。"
            )
        parts.append(
            "- 请改写查询用 search_local_papers 重新检索，或改用 search_arxiv 获取更多/更新论文补足证据；"
            "随后仅使用工具真实返回过的 [S编号] 重写答案。若确实检索不到充分依据，请如实说明而非编造。"
        )
        return "\n".join(parts)

    @staticmethod
    def _finalize(
        cleaned: str,
        valid_cited: list[str],
        hallucinated: list[str],
        sources: list[dict],
        steps: list[str],
    ) -> str:
        """二次检索预算用尽后定稿：编造引用已剔除，按情况追加核查提示并下调结论强度。"""
        if hallucinated:
            steps.append(f"引用核查: 仍有编造引用 {hallucinated}，已剔除并下调结论强度")
            cleaned += (
                f"\n\n> ⚠️ 引用核查：检测到无效引用 {', '.join(hallucinated)}"
                "（未对应真实检索来源，已从正文移除）。相关结论缺乏可溯源依据，请谨慎对待。"
            )
        elif not valid_cited and sources:
            steps.append("引用核查: 答案未标注任何可溯源引用")
            cleaned += "\n\n> ⚠️ 引用核查：本回答未标注可溯源的引用，请谨慎参考。"
        else:
            steps.append(f"引用核查: {len(valid_cited)} 条引用均可溯源 {valid_cited}")
        return cleaned

    @staticmethod
    def _summary_prompt(question: str, sources: list[dict]) -> str:
        refs = "\n".join(
            f"[{s.get('id', '?')}]《{s['paper_title']}》｜章节 {s['section']}"
            f"{PaperRAGAgent._page_label(s)}｜模态 {s.get('modality') or 'text'}\n{(s.get('snippet') or '')[:300]}"
            for s in sources
        )
        return (
            "请基于以下已检索到的论文来源，对用户问题给出尽可能完整的回答。"
            "每条来源以 [S编号] 标识，在关键结论后紧跟对应的 [S编号] 标注引用"
            "（如 [S1]，可连写 [S1][S3]）；只能引用下面出现过的编号，不要使用其他格式。\n\n"
            f"用户问题:\n{question}\n\n已检索来源:\n{refs or '(无)'}\n"
        )

    @staticmethod
    def _page_label(source: dict) -> str:
        start = source.get("page_start")
        end = source.get("page_end")
        if start is None or end is None:
            return ""
        if start == end:
            return f"｜页码 {start}"
        return f"｜页码 {start}-{end}"

    # ---------- 降级 RAG ----------
    @staticmethod
    def _top_score(results) -> float:
        return max((float(r.score) for r in results), default=float("-inf"))

    def _reformulate(self, query: str) -> str | None:
        """降级二次检索用的查询重写；无可用后端 / 失败 / 结果异常时返回 None（不重试）。"""
        if not self.llm.supports_agentic():
            return None
        prompt = (
            "下面的检索查询召回效果不佳，请换一种表述改写它以提升论文检索命中率："
            "可替换近义术语、补全全称/缩写、聚焦核心概念。只输出改写后的查询本身。\n\n"
            f"原查询：{query}\n改写："
        )
        try:
            out = self.llm.generate(prompt, system="你是检索查询改写助手，只输出改写后的查询。")
        except Exception:  # noqa: BLE001 - 重写失败不阻断主流程
            return None
        out = (out or "").strip().strip("\"'「」“”")
        if not out or len(out) > 300 or out == query or "降级模式" in out or "调用失败" in out:
            return None
        return out

    def _ask_fallback(self, question: str, standalone: str | None = None) -> AgentAnswer:
        """传统单跳 RAG，同样接入引用回查（护栏 #3）与有界低置信二次检索（护栏 #4）。"""
        query = standalone or question
        steps: list[str] = []
        trace: list[dict] = []
        if standalone and standalone != question:
            steps.append(f"Rewriter: 指代消解 → {standalone}")

        results = self.fallback.retrieve(query)
        steps.append(f"Retriever: 召回 {len(results)} 个片段（降级 RAG 模式）")
        sources = self.fallback.sources(results)

        # 低置信 → 有界一次纠错：改写查询再检索一次，取置信更高者（受 max_corrections 约束）
        if self._is_low_confidence(sources) and self.settings.retrieval.max_corrections > 0:
            new_q = self._reformulate(query)
            if new_q:
                retry = self.fallback.retrieve(new_q)
                if retry and self._top_score(retry) > self._top_score(results):
                    results, sources = retry, self.fallback.sources(retry)
                    steps.append(f"Corrective: 低置信，改写为「{new_q}」二次检索（采用更优结果）")
                    trace.append({"type": "verify", "result": "recheck", "mode": "fallback"})
                else:
                    steps.append("Corrective: 低置信，二次检索未带来更优结果")

        answerable, reason = self._answerability_status(sources)
        if not answerable:
            steps.append(f"Answerability: {reason}，拒绝生成实质答案")
            trace.append({"type": "answerability", "result": "reject", "reason": reason})
            return AgentAnswer(
                answer=self._insufficient_evidence_answer(reason, sources),
                steps=steps,
                sources=sources,
                trace=trace,
            )

        prompt = build_generation_prompt(question, results)
        answer = self.llm.generate(prompt)
        steps.append("Generator: 生成答案")

        # 引用回查 + claim 支持度核查（与 agentic 主路径同口径）
        fallback_evidence = EvidenceRegistry(self.settings)
        fallback_evidence.seed_dicts(sources)
        fallback_verifier = AnswerVerifier(fallback_evidence)
        checked = fallback_verifier.verify(answer)
        conflicts = fallback_verifier.conflicts()
        cleaned, valid_cited, hallucinated = (
            checked.cleaned,
            checked.valid,
            checked.invalid,
        )
        if checked.unsupported_claims:
            steps.append(
                f"ClaimVerify: {len(checked.unsupported_claims)} 条陈述需要人工复核"
            )
            trace.append(
                {
                    "type": "claim_verify",
                    "unsupported": checked.unsupported_claims,
                }
            )
        answerable, reason = self._answerability_status(sources, valid_cited)
        if not answerable:
            steps.append(f"Answerability: {reason}，拒绝生成实质答案")
            trace.append({"type": "answerability", "result": "reject", "reason": reason})
            return AgentAnswer(
                answer=self._insufficient_evidence_answer(reason, sources),
                steps=steps,
                sources=sources,
                trace=trace,
            )
        answer = self._finalize(cleaned, valid_cited, hallucinated, sources, steps)
        if conflicts:
            answer += (
                "\n\n> 检测到来源之间可能存在数值或否定关系冲突；"
                "请分别核对相关论文的实验条件，不将其视为一致结论。"
            )
            trace.append({"type": "evidence_conflict", "items": conflicts})
        trace.append(
            {"type": "verify", "result": "final", "valid": valid_cited, "hallucinated": hallucinated}
        )
        return AgentAnswer(answer=answer, steps=steps, sources=sources, trace=trace)
