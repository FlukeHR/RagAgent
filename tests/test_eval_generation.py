"""生成侧评估脚本的离线测试：只测授权门 + 样本组装纯逻辑，不导入 ragas、不打 API。"""
from __future__ import annotations

from evaluation.eval_generation import authorized, collect_samples


def test_authorized_gate(monkeypatch):
    monkeypatch.delenv("RAG_EVAL_ALLOW_API", raising=False)
    assert authorized(False) is False        # 默认拒绝
    assert authorized(True) is True           # --yes 放行
    monkeypatch.setenv("RAG_EVAL_ALLOW_API", "1")
    assert authorized(False) is True          # 环境变量放行


class _FakeAnswer:
    def __init__(self, answer, sources):
        self.answer = answer
        self.sources = sources


class _FakeAgent:
    def __init__(self, answer, sources):
        self._a, self._s = answer, sources

    def ask(self, q):
        return _FakeAnswer(self._a, self._s)


def test_collect_samples_shapes():
    items = [
        {"question": "q1"},
        {"question": "q2", "reference": "r2", "collection": "x"},
    ]
    agent = _FakeAgent("结论 [S1]", [{"snippet": "ctx1"}, {"snippet": ""}])
    out = collect_samples(items, lambda c: agent, default_collection="demo")

    assert out[0]["user_input"] == "q1"
    assert out[0]["response"] == "结论 [S1]"
    assert out[0]["retrieved_contexts"] == ["ctx1"]   # 空 snippet 被过滤
    assert "reference" not in out[0]                   # 未提供 reference
    assert out[1]["reference"] == "r2"                 # 提供则带上


def test_collect_samples_empty_contexts_fallback():
    agent = _FakeAgent("无依据回答", [])
    out = collect_samples([{"question": "q"}], lambda c: agent, default_collection="demo")
    assert out[0]["retrieved_contexts"] == ["(无检索内容)"]  # 无来源时占位，避免 ragas 报错


def test_collect_samples_limit():
    agent = _FakeAgent("a", [{"snippet": "c"}])
    items = [{"question": f"q{i}"} for i in range(5)]
    out = collect_samples(items, lambda c: agent, default_collection="demo", limit=2)
    assert len(out) == 2


# ---------- 官方 QASPER 生成侧（离线：hash 嵌入 + token 重排 + 假 LLM，不打 API） ----------
def _qasper_paper():
    return {
        "title": "Tiny Paper",
        "full_text": [{"section_name": "Method", "paragraphs": [
            "self attention computes pairwise relevance across all positions in parallel",
            "bert uses masked language modeling as a pretraining objective",
            "the transformer architecture has encoder and decoder stacks",
        ]}],
        "qas": [
            {"question": "How does self attention work?",
             "answers": [{"answer": {"free_form_answer": "It computes attention over all positions in parallel.",
                                      "evidence": ["self attention computes pairwise relevance across all positions in parallel"]}}]},
            {"question": "Is this unanswerable?",
             "answers": [{"answer": {"unanswerable": True}}]},
        ],
    }


def test_qasper_reference_variants():
    from evaluation.eval_generation import qasper_reference
    assert qasper_reference({"answers": [{"answer": {"free_form_answer": "X"}}]}) == "X"
    assert qasper_reference({"answers": [{"answer": {"extractive_spans": ["a", "b"]}}]}) == "a b"
    assert qasper_reference({"answers": [{"answer": {"yes_no": True}}]}) == "Yes"
    assert qasper_reference({"answers": [{"answer": {"yes_no": False}}]}) == "No"
    assert qasper_reference({"answers": [{"answer": {"unanswerable": True}}]}) is None
    assert qasper_reference({"answers": []}) is None


def test_collect_samples_qasper_offline(settings):
    import dataclasses

    from evaluation.eval_generation import collect_samples_qasper

    s = dataclasses.replace(
        settings,
        embedding=dataclasses.replace(settings.embedding, use_sentence_transformers=False),
        rerank=dataclasses.replace(settings.rerank, use_cross_encoder=False),
    )

    class FakeLLM:
        def generate(self, prompt, system=None):
            return "self attention works in parallel [S1]"

    out = collect_samples_qasper({"p1": _qasper_paper()}, s, FakeLLM(), limit=10)
    assert len(out) == 1  # unanswerable 那题被跳过
    sample = out[0]
    assert sample["user_input"].startswith("How does self attention")
    assert sample["response"] == "self attention works in parallel [S1]"
    assert sample["reference"].startswith("It computes attention")
    assert sample["retrieved_contexts"] and all(isinstance(c, str) for c in sample["retrieved_contexts"])
