"""多轮对话的离线测试：历史注入 / 查询改写（指代消解）/ 标题降级。全程 mock，不打 API。"""
from __future__ import annotations

from llm.model import LLMClient, LLMTurn


def _final(text):
    """一个直接给最终答案、不调工具的 turn。"""
    return LLMTurn(text=text, tool_calls=[], stop=True, raw=text, usage={})


# ---------- init_history 历史注入（LLMClient 纯单元） ----------
def test_init_history_injects_prior_turns(settings):
    client = LLMClient(settings.llm)
    prior = [
        {"role": "user", "content": "Transformer 是什么？"},
        {"role": "assistant", "content": "一种基于自注意力的架构。"},
        {"role": "system", "content": "应被忽略"},  # 非法 role
        {"role": "user", "content": "  "},            # 空内容
    ]
    hist = client.init_history("它的复杂度呢？", prior=prior)
    assert hist == [
        {"role": "user", "content": "Transformer 是什么？"},
        {"role": "assistant", "content": "一种基于自注意力的架构。"},
        {"role": "user", "content": "它的复杂度呢？"},
    ]


def test_init_history_no_prior(settings):
    client = LLMClient(settings.llm)
    assert client.init_history("hi") == [{"role": "user", "content": "hi"}]


# ---------- _ask_agentic 把历史注入工作上下文 ----------
def test_ask_agentic_injects_history(make_agent, fake_llm):
    captured = {}

    class RecordingLLM(fake_llm):
        def create_turn(self, system, history, tools):
            captured["history"] = list(history)
            return self.turns.pop(0)

    llm = RecordingLLM([_final("这是基于上下文的回答。")])
    agent = make_agent(llm=llm)
    prior = [
        {"role": "user", "content": "讲讲 BERT"},
        {"role": "assistant", "content": "双向编码器。"},
    ]
    agent.ask("它的参数量是多少？", history=prior, )
    # 工作历史前缀应是注入的两条，末尾是当前问题
    roles = [m["role"] for m in captured["history"]]
    assert roles == ["user", "assistant", "user"]
    assert captured["history"][0]["content"] == "讲讲 BERT"


# ---------- 查询改写 / 指代消解 ----------
def test_rewrite_resolves_coreference(make_agent, fake_llm):
    llm = fake_llm([], gen=lambda p, s: "BERT 的参数量是多少？")
    agent = make_agent(llm=llm)
    prior = [{"role": "user", "content": "讲讲 BERT"}]
    out = agent._rewrite_query("它的参数量是多少？", prior)
    assert out == "BERT 的参数量是多少？"


def test_rewrite_skipped_without_history(make_agent, fake_llm):
    called = {"n": 0}

    def gen(p, s):
        called["n"] += 1
        return "不该被调用"

    llm = fake_llm([], gen=gen)
    agent = make_agent(llm=llm)
    assert agent._rewrite_query("它是什么？", []) == "它是什么？"
    assert called["n"] == 0


def test_rewrite_skipped_without_backend(make_agent, fake_llm):
    llm = fake_llm([], gen=lambda p, s: "改写结果", agentic=False)
    agent = make_agent(llm=llm)
    prior = [{"role": "user", "content": "上文"}]
    assert agent._rewrite_query("它呢？", prior) == "它呢？"


def test_rewrite_falls_back_on_junk(make_agent, fake_llm):
    # 命中本地降级文案 -> 视为不可用，回退原问题
    llm = fake_llm([], gen=lambda p, s: "【本地降级模式回答（未配置大模型 API）】 ...")
    agent = make_agent(llm=llm)
    prior = [{"role": "user", "content": "上文"}]
    q = "它的准确率？"
    assert agent._rewrite_query(q, prior) == q


# ---------- 标题降级 ----------
def test_title_truncate_helper():
    from api.routes import _truncate_title

    assert _truncate_title("") == "新对话"
    assert _truncate_title("短标题") == "短标题"
    long = "这是一个非常非常非常非常非常非常长的问题用来测试截断逻辑"
    out = _truncate_title(long, limit=10)
    assert out.endswith("…") and len(out) == 11


def test_title_route_degrades_without_backend(monkeypatch):
    """无真实后端时 /title 走截断降级，不调用任何 API（红线）。"""
    from api import routes
    from api.schemas import Turn, TitleRequest

    monkeypatch.setattr(routes.LLMClient, "supports_agentic", lambda self: False)
    req = TitleRequest(messages=[Turn(role="user", content="Transformer 的注意力机制原理")])
    resp = routes.make_title(req)
    assert resp.title.startswith("Transformer")
