from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
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
    """一次模型回复的 provider 无关表示。"""

    text: str
    tool_calls: list[ToolCall]
    stop: bool
    raw: Any = None  # provider 原生 assistant 表示，用于回填历史
    usage: dict = field(default_factory=dict)  # {input_tokens, output_tokens}，供预算核算


@dataclass
class ToolOutcome:
    tool_call: ToolCall
    content: str
    is_error: bool = False


class LLMClient:
    """统一 LLM 封装，支持两类 agentic 后端：

    - Anthropic：Claude 原生 tool use。
    - OpenAI 兼容：OpenAI function calling（DeepSeek / Qwen / Ollama / vLLM 等）。

    两者通过 init_history / create_turn / append_* 这组 provider 无关的原语被
    agent 的同一个循环驱动。无可用后端时降级为纯文本生成。
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._anthropic = None

    @property
    def provider(self) -> str:
        return self.config.provider.lower()

    # ---------- 可用性 ----------
    def anthropic_available(self) -> bool:
        if self.provider != "anthropic":
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        # 支持三种凭据：API key / OAuth AUTH_TOKEN / `ant auth login` 写入的 profile
        if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
            return True
        config_dir = os.getenv("ANTHROPIC_CONFIG_DIR") or os.path.expanduser(
            "~/.config/anthropic"
        )
        creds = Path(config_dir) / "credentials"
        return creds.is_dir() and any(creds.glob("*.json"))

    def _openai_configured(self) -> bool:
        return bool(
            self.config.openai_api_base
            or self.config.openai_api_key
            or os.getenv("OPENAI_API_KEY")
        )

    def supports_agentic(self) -> bool:
        """是否能走带工具调用的 agentic 循环。"""
        if self.anthropic_available():
            return True
        return self.provider == "openai" and self._openai_configured()

    # ---------- 统一 agentic 原语 ----------
    def init_history(
        self, question: str, prior: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """构造工作历史：把之前的对话轮次（纯文本）注入到当前问题前面。

        prior 中每项为 {"role": "user"|"assistant", "content": str}；两类 provider
        都接受字符串 content，故可直接复用。非法 role / 空内容会被跳过。
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
        if self.provider == "openai":
            return self._openai_turn(system, history, tools)
        return self._anthropic_turn(system, history, tools)

    def append_assistant(self, history: list[dict], turn: LLMTurn) -> None:
        if self.provider == "openai":
            history.append(turn.raw)
        else:
            history.append({"role": "assistant", "content": turn.raw})

    def append_tool_results(
        self, history: list[dict], outcomes: list[ToolOutcome]
    ) -> None:
        if self.provider == "openai":
            for o in outcomes:
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": o.tool_call.id,
                        "content": o.content,
                    }
                )
        else:
            blocks = []
            for o in outcomes:
                block = {
                    "type": "tool_result",
                    "tool_use_id": o.tool_call.id,
                    "content": o.content,
                }
                if o.is_error:
                    block["is_error"] = True
                blocks.append(block)
            history.append({"role": "user", "content": blocks})

    # ---------- Anthropic 实现 ----------
    def _client(self):
        import anthropic

        if self._anthropic is None:
            self._anthropic = anthropic.Anthropic(timeout=120.0)
        return self._anthropic

    def _anthropic_turn(
        self, system: str, history: list[dict], tools: list[dict]
    ) -> LLMTurn:
        client = self._client()
        kwargs: dict[str, Any] = dict(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            system=system,
            messages=history,
            tools=tools,
            thinking={"type": "adaptive"},
        )
        try:
            resp = client.messages.create(
                extra_body={"output_config": {"effort": self.config.effort}}, **kwargs
            )
        except TypeError:
            resp = client.messages.create(**kwargs)

        text = "".join(b.text for b in resp.content if b.type == "text")
        tool_calls = [
            ToolCall(id=b.id, name=b.name, input=dict(b.input))
            for b in resp.content
            if b.type == "tool_use"
        ]
        usage = getattr(resp, "usage", None)
        return LLMTurn(
            text=text,
            tool_calls=tool_calls,
            stop=resp.stop_reason != "tool_use",
            raw=resp.content,
            usage={
                "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
                "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
            },
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
            if self.provider == "anthropic" and self.anthropic_available():
                return self._generate_anthropic(prompt, system)
            if self.provider == "openai":
                return self._generate_openai(prompt, system)
        except Exception as exc:  # noqa: BLE001 - 调用失败兜底为本地检索摘要
            return f"【大模型调用失败，降级为本地检索摘要：{exc}】\n\n" + self._generate_local(prompt)
        return self._generate_local(prompt)

    def _generate_anthropic(self, prompt: str, system: str) -> str:
        client = self._client()
        resp = client.messages.create(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

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
            out.append("如需高质量的综合回答与引用整合，请配置大模型后端（Anthropic 或 OpenAI 兼容）。")
        else:
            out.append("- 未检索到相关论文片段。")
        return "\n".join(out)
