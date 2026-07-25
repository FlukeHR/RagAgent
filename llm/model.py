from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from config.settings import LLMConfig

DEFAULT_SYSTEM = "你是一个专业、严谨的学术论文研究助手，回答需基于证据并标注引用。"


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class LLMTurn:
    """一次 OpenAI-compatible 模型回复。"""

    text: str
    tool_calls: list[ToolCall]
    stop: bool
    raw: Any = None  # OpenAI assistant message，用于回填历史
    usage: dict = field(default_factory=dict)  # {input_tokens, output_tokens}，供预算核算


@dataclass
class ToolOutcome:
    tool_call: ToolCall
    content: str
    is_error: bool = False


class LLMClient:
    """OpenAI-compatible LLM 封装。

    支持 OpenAI function calling，以及 DeepSeek、Qwen、Ollama、vLLM 等兼容接口。
    未配置接口或调用失败时降级为本地检索摘要。
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    # ---------- 可用性 ----------
    def _openai_configured(self) -> bool:
        return bool(
            self.config.openai_api_base
            or self.config.openai_api_key
            or os.getenv("OPENAI_API_KEY")
        )

    def supports_agentic(self) -> bool:
        """是否能走带工具调用的 agentic 循环。"""
        return self._openai_configured()

    # ---------- agentic 原语 ----------
    def init_history(
        self, question: str, prior: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """构造工作历史：把之前的对话轮次（纯文本）注入到当前问题前面。

        prior 中每项为 {"role": "user"|"assistant", "content": str}。
        非法 role / 空内容会被跳过。
        """
        msgs: list[dict[str, Any]] = []
        for t in prior or []:
            role = t.get("role")
            content = (t.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})
        msgs.append({"role": "user", "content": question})
        return msgs

    def create_turn(
        self, system: str, history: list[dict], tools: list[dict]
    ) -> LLMTurn:
        return self._openai_turn(system, history, tools)

    def append_assistant(self, history: list[dict], turn: LLMTurn) -> None:
        history.append(turn.raw)

    def append_tool_results(
        self, history: list[dict], outcomes: list[ToolOutcome]
    ) -> None:
        for o in outcomes:
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": o.tool_call.id,
                    "content": o.content,
                }
            )

    # ---------- OpenAI 兼容实现 ----------
    def _openai_client(self):
        import httpx
        from openai import OpenAI

        return OpenAI(
            api_key=self.config.openai_api_key or os.getenv("OPENAI_API_KEY") or "not-needed",
            base_url=self.config.openai_api_base or None,
            timeout=httpx.Timeout(120.0, connect=30.0),
            max_retries=2,
        )

    @staticmethod
    def _to_openai_tools(tools: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

    def _openai_turn(
        self, system: str, history: list[dict], tools: list[dict]
    ) -> LLMTurn:
        client = self._openai_client()
        messages = [{"role": "system", "content": system}, *history]
        resp = client.chat.completions.create(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            messages=messages,
            tools=self._to_openai_tools(tools),
        )
        msg = resp.choices[0].message
        raw_tool_calls = msg.tool_calls or []

        tool_calls = []
        for tc in raw_tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))

        raw = {"role": "assistant", "content": msg.content}
        if raw_tool_calls:
            raw["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in raw_tool_calls
            ]
        usage = getattr(resp, "usage", None)
        return LLMTurn(
            text=msg.content or "",
            tool_calls=tool_calls,
            stop=not raw_tool_calls,
            raw=raw,
            usage={
                "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            },
        )

    # ---------- 降级：纯文本生成 ----------
    def generate(self, prompt: str, system: str = DEFAULT_SYSTEM) -> str:
        try:
            if self._openai_configured():
                return self._generate_openai(prompt, system)
        except Exception as exc:  # noqa: BLE001 - 调用失败兜底为本地检索摘要
            return f"【大模型调用失败，降级为本地检索摘要：{exc}】\n\n" + self._generate_local(prompt)
        return self._generate_local(prompt)

    def _generate_openai(self, prompt: str, system: str) -> str:
        client = self._openai_client()
        response = client.chat.completions.create(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _generate_local(prompt: str) -> str:
        lines = prompt.splitlines()
        question = "未找到"
        for i, line in enumerate(lines):
            if line.strip() == "用户问题:" and i + 1 < len(lines):
                question = lines[i + 1].strip()
                break
        contexts = [line for line in lines if line.startswith("[来源")]

        out = [
            "【本地降级模式回答（未配置大模型 API）】",
            f"问题: {question}",
            "",
            "基于本地检索到的论文片段，相关来源如下：",
        ]
        if contexts:
            out.extend(f"- {c}" for c in contexts[:5])
            out.append("")
            out.append("如需高质量的综合回答与引用整合，请配置 OpenAI-compatible API。")
        else:
            out.append("- 未检索到相关论文片段。")
        return "\n".join(out)
