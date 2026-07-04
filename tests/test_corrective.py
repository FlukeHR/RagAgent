"""引用回查 + 低置信二次检索（Corrective RAG）的离线测试。"""
from __future__ import annotations

from llm.model import LLMTurn, ToolCall
from tools.base import ToolResult

SEARCH_SCHEMA = {
    "name": "search_local_papers",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}


def _src(sid: str, score: float) -> dict:
    return {
        "id": sid, "score": score, "paper_id": "p",
        "paper_title": "T", "section": "M", "snippet": "x",
    }


class FakeSearchTool:
    def __init__(self, sources):
        self._sources = sources

    @staticmethod
    def schema():
        return SEARCH_SCHEMA

    def run(self, _id_base: int = 0, **kw):
        return ToolResult(text="ctx", sources=self._sources)


def _call(query="x"):
    return ToolCall(id="t", name="search_local_papers", input={"query": query})


def test_hallucinated_citation_triggers_recheck_then_fixed(make_agent, fake_llm):
    turns = [
        LLMTurn(text="", tool_calls=[_call()], stop=False),
        LLMTurn(text="结论 [S9]。", tool_calls=[], stop=True),  # 编造 S9
        LLMTurn(text="结论 [S1]。", tool_calls=[], stop=True),  # 修正为真实 S1
    ]
    agent = make_agent(
        llm=fake_llm(turns), tools={"search_local_papers": FakeSearchTool([_src("S1", 5.0)])}
    )
    ans = agent._ask_agentic("q")
    assert "[S1]" in ans.answer and "[S9]" not in ans.answer
    assert any("二次检索" in s for s in ans.steps)
    assert any(e["type"] == "verify" and e.get("result") == "recheck" for e in ans.trace)


def test_low_confidence_triggers_bounded_recheck(make_agent, fake_llm):
    # score 低于阈值(0.0) -> 低置信；二次检索次数应被 max_corrections 限制
    turns = [
        LLMTurn(text="", tool_calls=[_call()], stop=False),
        LLMTurn(text="结论 [S1]。", tool_calls=[], stop=True),
        LLMTurn(text="结论 [S1]。", tool_calls=[], stop=True),
    ]
    agent = make_agent(
        llm=fake_llm(turns), tools={"search_local_papers": FakeSearchTool([_src("S1", -1.0)])}
    )
    ans = agent._ask_agentic("q")
    rechecks = [e for e in ans.trace if e.get("result") == "recheck"]
    assert len(rechecks) == agent.settings.retrieval.max_corrections


def test_valid_citation_passes_without_recheck(make_agent, fake_llm):
    turns = [
        LLMTurn(text="", tool_calls=[_call()], stop=False),
        LLMTurn(text="结论 [S1]。", tool_calls=[], stop=True),
    ]
    agent = make_agent(
        llm=fake_llm(turns), tools={"search_local_papers": FakeSearchTool([_src("S1", 5.0)])}
    )
    ans = agent._ask_agentic("q")
    assert ans.answer.strip().startswith("结论 [S1]")
    assert not any(e.get("result") == "recheck" for e in ans.trace)


def test_no_sources_does_not_crash(make_agent, fake_llm):
    turns = [LLMTurn(text="一段没有引用的回答。", tool_calls=[], stop=True)]
    agent = make_agent(llm=fake_llm(turns), tools={})
    ans = agent._ask_agentic("q")
    assert "没有引用的回答" in ans.answer


# ---------- 降级单跳 RAG 路径：引用回查 + 有界二次检索 ----------
def _fake_retriever(score_map):
    """按 query 返回不同分数的单条结果；score_map: {query: score}，缺省 default。"""
    from retrieval.chunker import Chunk
    from retrieval.retriever import RetrievalResult

    chunk = Chunk(chunk_id="c0", paper_id="p", paper_title="T", section="M",
                  content="证据内容", source="x.pdf")

    class _R:
        def search(self, q):
            return [RetrievalResult(chunk=chunk, score=score_map.get(q, score_map["__default__"]))]

    return _R()


def _attach_retriever(agent, retriever):
    agent.search_tool = type("ST", (), {"retriever": retriever})()


def test_fallback_grounds_citations(make_agent, fake_llm):
    # 高置信(score=5)，答案含编造 [S9] → 回查剔除，仅留真实 [S1]
    agent = make_agent(llm=fake_llm([], gen=lambda p, s: "结论 [S1]，另称 [S9]。"))
    _attach_retriever(agent, _fake_retriever({"__default__": 5.0}))
    ans = agent._ask_fallback("q")
    assert "[S1]" in ans.answer and "[S9]" not in ans.answer
    assert any(e.get("result") == "final" for e in ans.trace)


def test_fallback_low_confidence_retry(make_agent, fake_llm):
    # 首次低置信(-1)，改写为"更好的查询"后高置信(5) → 采用更优结果
    def gen(p, s):
        return "更好的查询" if "改写" in p else "结论 [S1]。"

    agent = make_agent(llm=fake_llm([], gen=gen))
    _attach_retriever(agent, _fake_retriever({"__default__": -1.0, "更好的查询": 5.0}))
    ans = agent._ask_fallback("q")
    assert any("二次检索（采用更优结果）" in s for s in ans.steps)
    assert any(e.get("result") == "recheck" and e.get("mode") == "fallback" for e in ans.trace)
