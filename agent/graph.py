from __future__ import annotations

from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentAnswer
from config.settings import Settings
from llm.model import LLMClient, ToolCall, ToolOutcome
from llm.prompt_builder import build_generation_prompt
from tools import ArxivTool, PaperReaderTool, PaperSearchTool


class PaperRAGAgent:
    """论文问答 Agent。

    - 后端支持工具调用时（Claude 原生 tool use，或 DeepSeek/Qwen/Ollama 等 OpenAI
      兼容的 function calling）：走 agentic 循环（多源检索 + 自我纠错 + 引用溯源）。
    - 否则：降级为传统单跳 RAG（本地检索 + 拼接生成）。
    """

    def __init__(self, settings: Settings, collection: str) -> None:
        self.settings = settings
        self.collection = collection
        self.llm = LLMClient(settings.llm)
        self.search_tool = PaperSearchTool(settings, collection)
        self.arxiv_tool = ArxivTool(settings)
        self.reader_tool = PaperReaderTool(settings, collection)
        self._tools = {
            self.search_tool.name: self.search_tool,
            self.arxiv_tool.name: self.arxiv_tool,
            self.reader_tool.name: self.reader_tool,
        }

    def ask(self, question: str) -> AgentAnswer:
        if self.llm.supports_agentic():
            try:
                return self._ask_agentic(question)
            except Exception as exc:  # noqa: BLE001 - 后端不可用时降级，不让请求直接失败
                fb = self._ask_fallback(question)
                fb.steps.insert(0, f"⚠️ agentic 后端调用失败，已降级为本地检索: {exc}")
                return fb
        return self._ask_fallback(question)

    # ---------- agentic 主路径（provider 无关） ----------
    def _tool_schemas(self) -> list[dict]:
        return [PaperSearchTool.schema(), ArxivTool.schema(), PaperReaderTool.schema()]

    def _ask_agentic(self, question: str) -> AgentAnswer:
        steps: list[str] = [f"Planner: 启动 agentic 检索循环（provider={self.llm.provider}）"]
        sources: list[dict] = []
        history = self.llm.init_history(question)
        answer = ""

        for hop in range(self.settings.llm.max_tool_iters):
            turn = self.llm.create_turn(SYSTEM_PROMPT, history, self._tool_schemas())
            self.llm.append_assistant(history, turn)

            if turn.stop or not turn.tool_calls:
                answer = turn.text
                steps.append(f"Generator: 第 {hop + 1} 轮生成最终答案")
                break

            outcomes = [self._run_tool(tc, sources, steps) for tc in turn.tool_calls]
            self.llm.append_tool_results(history, outcomes)
        else:
            steps.append("Generator: 达到最大轮次，基于已检索内容生成总结")
            answer = self.llm.generate(self._summary_prompt(question, sources))

        return AgentAnswer(
            answer=answer or "未能生成答案。",
            steps=steps,
            sources=sources,  # 保留每条带唯一 id 的来源，前端按 [S编号] 映射
        )

    def _run_tool(
        self, tool_call: ToolCall, sources: list[dict], steps: list[str]
    ) -> ToolOutcome:
        name = tool_call.name
        tool_input = tool_call.input
        if name not in self._tools:
            steps.append(f"Tool[{name}] 未知工具")
            return ToolOutcome(tool_call, f"未知工具: {name}", is_error=True)
        try:
            # _id_base 让本轮检索到的 chunk 获得跨多次调用全局唯一的 [S编号]
            result = self._tools[name].run(**tool_input, _id_base=len(sources))
            sources.extend(result.sources)
            steps.append(f"Tool[{name}] {tool_input} -> {len(result.sources)} 来源")
            return ToolOutcome(tool_call, result.text)
        except Exception as exc:  # noqa: BLE001 - 工具失败回传给模型，由其调整策略
            steps.append(f"Tool[{name}] 失败: {exc}")
            return ToolOutcome(tool_call, f"工具执行失败: {exc}", is_error=True)

    @staticmethod
    def _summary_prompt(question: str, sources: list[dict]) -> str:
        refs = "\n".join(
            f"[{s.get('id', '?')}]《{s['paper_title']}》｜章节 {s['section']}\n{(s.get('snippet') or '')[:300]}"
            for s in sources
        )
        return (
            "请基于以下已检索到的论文来源，对用户问题给出尽可能完整的回答。"
            "每条来源以 [S编号] 标识，在关键结论后紧跟对应的 [S编号] 标注引用"
            "（如 [S1]，可连写 [S1][S3]）；只能引用下面出现过的编号，不要使用其他格式。\n\n"
            f"用户问题:\n{question}\n\n已检索来源:\n{refs or '(无)'}\n"
        )

    # ---------- 降级 RAG ----------
    def _ask_fallback(self, question: str) -> AgentAnswer:
        results = self.search_tool.retriever.search(question)
        steps = [f"Retriever: 召回 {len(results)} 个片段（降级 RAG 模式）"]
        prompt = build_generation_prompt(question, results)
        answer = self.llm.generate(prompt)
        steps.append("Generator: 生成答案")

        sources = [
            {
                "id": f"S{i}",
                "chunk_id": r.chunk.chunk_id,
                "paper_id": r.chunk.paper_id,
                "paper_title": r.chunk.paper_title,
                "section": r.chunk.section,
                "source": r.chunk.source,
                "score": round(float(r.score), 4),
                "snippet": r.chunk.content[:600],
            }
            for i, r in enumerate(results, start=1)
        ]
        return AgentAnswer(answer=answer, steps=steps, sources=sources)
