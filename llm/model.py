from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any, cast
from urllib.parse import urlparse

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from config.settings import LLMConfig


DEFAULT_SYSTEM = "你是一个专业、严谨的学术论文研究助手，回答必须基于证据并标注引用。"


def message_text(message: AIMessage) -> str:
    """Return provider-neutral text from a LangChain AI message."""

    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
    return "".join(parts)


class LLMClient:
    """OpenAI-compatible LangChain model factory plus bounded text generation."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._usage_events: list[dict[str, Any]] = []
        self._model: BaseChatModel | None = None

    def consume_usage_events(self) -> list[dict[str, Any]]:
        events, self._usage_events = self._usage_events, []
        return events

    def _api_key(self) -> str:
        return self.config.openai_api_key or os.getenv("OPENAI_API_KEY") or ""

    def _is_local_endpoint(self) -> bool:
        if not self.config.openai_api_base:
            return False
        hostname = (urlparse(self.config.openai_api_base).hostname or "").lower()
        return hostname in {"localhost", "127.0.0.1", "::1"}

    def supports_agentic(self) -> bool:
        return bool(self._api_key() or self._is_local_endpoint())

    def configuration_issue(self) -> str | None:
        if self.supports_agentic():
            return None
        if self.config.openai_api_base:
            return "远程 OpenAI-compatible 后端缺少 OPENAI_API_KEY"
        return "未配置 OpenAI-compatible 后端"

    def chat_model(self) -> BaseChatModel:
        """Build one reusable ChatOpenAI client without enabling external tracing."""

        if not self.supports_agentic():
            raise RuntimeError(self.configuration_issue() or "LLM 后端不可用")
        if self._model is None:
            self._model = ChatOpenAI(
                model=self.config.model_name,
                api_key=SecretStr(self._api_key() or "not-needed"),
                base_url=self.config.openai_api_base or None,
                timeout=httpx.Timeout(
                    self.config.request_timeout_seconds,
                    connect=self.config.connect_timeout_seconds,
                ),
                max_retries=self.config.max_retries,
                max_completion_tokens=self.config.max_tokens,
                stream_usage=False,
            )
        return self._model

    def generate(self, prompt: str, system: str = DEFAULT_SYSTEM) -> str:
        """Generate plain text or return the deterministic local fallback."""

        if not self.supports_agentic():
            return self._generate_local(prompt)
        started = time.perf_counter()
        message = self.chat_model().invoke(
            [SystemMessage(content=system), HumanMessage(content=prompt)]
        )
        if not isinstance(message, AIMessage):
            raise TypeError("chat model returned a non-AI message")
        usage: dict[str, Any] = message.usage_metadata or {}
        self._usage_events.append(
            {
                "operation": "generate",
                "input_tokens": int(usage.get("input_tokens", 0) or 0),
                "output_tokens": int(usage.get("output_tokens", 0) or 0),
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )
        return message_text(message)

    def stream(self, prompt: str, system: str = DEFAULT_SYSTEM) -> Iterator[str]:
        """Yield provider text chunks and record first-token/full latency."""

        if not self.supports_agentic():
            yield self._generate_local(prompt)
            return
        started = time.perf_counter()
        first_token_ms: float | None = None
        usage: dict[str, Any] = {}
        for chunk in self.chat_model().stream(
            [SystemMessage(content=system), HumanMessage(content=prompt)]
        ):
            usage = cast(dict[str, Any], getattr(chunk, "usage_metadata", None) or usage)
            text = message_text(cast(AIMessage, chunk))
            if not text:
                continue
            if first_token_ms is None:
                first_token_ms = round((time.perf_counter() - started) * 1000, 1)
            yield text
        self._usage_events.append(
            {
                "operation": "stream",
                "input_tokens": int(usage.get("input_tokens", 0) or 0),
                "output_tokens": int(usage.get("output_tokens", 0) or 0),
                "first_token_ms": first_token_ms,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )

    @staticmethod
    def _generate_local(prompt: str) -> str:
        lines = prompt.splitlines()
        question = "未找到"
        for index, line in enumerate(lines):
            if line.strip() == "用户问题:" and index + 1 < len(lines):
                question = lines[index + 1].strip()
                break
        contexts = [line for line in lines if line.startswith("[S")]
        output = [
            "【本地降级模式回答（未配置大模型 API）】",
            f"问题: {question}",
            "",
            "基于本地检索到的论文片段，相关来源如下：",
        ]
        if contexts:
            output.extend(f"- {item}" for item in contexts[:5])
        else:
            output.append("- 未检索到相关论文片段。")
        return "\n".join(output)
