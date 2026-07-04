"""harness 护栏的离线测试：schema 校验 / 超时重试 / token 预算。"""
from __future__ import annotations

import time

from llm.model import LLMTurn, ToolCall
from tools.base import ToolResult


def _schema(required):
    return {
        "name": "t",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": required,
        },
    }


class GoodTool:
    @staticmethod
    def schema():
        return _schema(["query"])

    def run(self, _id_base: int = 0, **kw):
        return ToolResult(text="ok", sources=[{"id": "S1", "score": 1.0, "paper_title": "T", "section": "M"}])


class SlowTool:
    @staticmethod
    def schema():
        return _schema([])

    def run(self, _id_base: int = 0, **kw):
        time.sleep(2)
        return ToolResult(text="late", sources=[])


class BoomTool:
    @staticmethod
    def schema():
        return _schema([])

    def run(self, _id_base: int = 0, **kw):
        raise RuntimeError("boom")


def test_schema_rejects_missing_required(make_agent):
    agent = make_agent(tools={"t": GoodTool()})
    out = agent._run_tool(ToolCall("1", "t", {}), [], [], [])
    assert out.is_error and "required" in out.content


def test_unknown_tool_returns_error(make_agent):
    agent = make_agent(tools={})
    out = agent._run_tool(ToolCall("1", "nope", {}), [], [], [])
    assert out.is_error


def test_timeout_is_bounded_and_retried(make_agent):
    agent = make_agent(tools={"t": SlowTool()}, tool_timeout_seconds=0.3, tool_max_retries=1)
    trace: list[dict] = []
    t0 = time.perf_counter()
    out = agent._run_tool(ToolCall("1", "t", {}), [], [], trace)
    elapsed = time.perf_counter() - t0
    assert out.is_error and "超时" in out.content
    assert trace[-1]["attempts"] == 2  # 首次 + 1 次重试
    assert elapsed < 2.0  # 远小于 2×2s 的真实执行 -> 超时确实生效


def test_tool_exception_becomes_error_outcome(make_agent):
    agent = make_agent(tools={"t": BoomTool()})
    out = agent._run_tool(ToolCall("1", "t", {}), [], [], [])
    assert out.is_error and "boom" in out.content


def test_token_budget_stops_multihop(make_agent, fake_llm):
    turns = [
        LLMTurn(
            text="", tool_calls=[ToolCall("t", "t", {"query": "x"})], stop=False,
            usage={"input_tokens": 80, "output_tokens": 80},
        )
    ]
    agent = make_agent(llm=fake_llm(turns), tools={"t": GoodTool()}, token_budget=100)
    ans = agent._ask_agentic("q")
    assert any(e["type"] == "budget" for e in ans.trace)
    assert any("预算" in s for s in ans.steps)
