from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from agent.graph import PaperRAGAgent
from config.settings import load_settings


class FakeLLM:
    """provider 无关的假后端：按预设序列吐 turn，绝不打任何 API（红线）。"""

    provider = "fake"

    def __init__(self, turns, summary: str = "兜底总结", gen=None, agentic: bool = True):
        self.turns = list(turns)
        self.summary = summary
        self._gen = gen  # 可选：自定义 generate（如改写/标题），签名 (prompt, system)->str
        self._agentic = agentic

    def supports_agentic(self) -> bool:
        return self._agentic

    def init_history(self, question, prior=None):
        msgs = []
        for t in prior or []:
            role = t.get("role")
            content = (t.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})
        msgs.append({"role": "user", "content": question})
        return msgs

    def create_turn(self, system, history, tools):
        return self.turns.pop(0)

    def append_assistant(self, history, turn):
        history.append({"role": "assistant", "content": turn.text})

    def append_tool_results(self, history, outcomes):
        for o in outcomes:
            history.append({"role": "tool", "content": o.content})

    def generate(self, prompt, system=None):
        if self._gen is not None:
            return self._gen(prompt, system)
        return self.summary


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def fake_llm():
    """返回 FakeLLM 类；测试按 `fake_llm([turn, ...])` 实例化。"""
    return FakeLLM


@pytest.fixture
def make_agent(settings):
    """构造一个绕过 __init__（不加载索引/真实工具）的 PaperRAGAgent。

    可传 llm / tools，并用关键字覆盖 harness 配置（如 tool_timeout_seconds）。
    """

    def _make(llm=None, tools=None, **harness_overrides):
        agent = object.__new__(PaperRAGAgent)
        if harness_overrides:
            agent.settings = dataclasses.replace(
                settings,
                harness=dataclasses.replace(settings.harness, **harness_overrides),
            )
        else:
            agent.settings = settings
        agent.llm = llm
        agent._tools = tools or {}
        return agent

    return _make
