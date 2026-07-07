"""检索侧离线测试：嵌入/重排降级、BM25 RRF 融合、低置信判定。"""
from __future__ import annotations

import numpy as np

from retrieval.chunker import Chunk
from retrieval.embedder import Embedder
from retrieval.reranker import Reranker
from retrieval.retriever import Retriever


def _chunk(cid: str, content: str) -> Chunk:
    return Chunk(
        chunk_id=cid, paper_id="p", paper_title="T",
        section="S", content=content, source="x",
    )


def test_embedder_hash_fallback_dim_and_norm():
    emb = Embedder(model_name="nonexistent-model-xyz", use_sentence_transformers=False)
    vecs = emb.encode(["hello world", "foo bar baz"])
    assert vecs.shape == (2, 384)  # 与默认 ST 模型维度对齐，避免 faiss 维度断言失败
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-3)


def test_reranker_token_overlap_fallback_orders_by_relevance():
    rr = Reranker(model_name="x", use_cross_encoder=False)
    cands = [
        (_chunk("1", "apple banana orange"), 0.1),
        (_chunk("2", "attention transformer model"), 0.1),
    ]
    out = rr.rerank("transformer attention", cands, top_n=2)
    assert out[0][0].chunk_id == "2"  # 与查询词重叠更多者排前


def test_rrf_fuse_combines_two_lists():
    c1, c2, c3 = _chunk("1", "a"), _chunk("2", "b"), _chunk("3", "c")
    dense = [(c1, 0.9), (c2, 0.8)]
    sparse = [(c2, 5.0), (c3, 4.0)]
    fused = Retriever._rrf_fuse([dense, sparse])
    ids = [c.chunk_id for c, _ in fused]
    assert ids[0] == "2"  # 两路都靠前 -> RRF 最高
    assert set(ids) == {"1", "2", "3"}


def test_is_low_confidence(make_agent):
    # 新判据：分数过 sigmoid 归一化为相关概率，再做强度+数量双判据
    agent = make_agent()  # 默认 强阈值0.5 / 弱阈值0.35 / 至少1条
    # CE logit 5.0 → sigmoid≈0.99：强且够格 → 不低置信
    assert agent._is_low_confidence([{"id": "S1", "score": 5.0}]) is False
    # CE logit -1.0 → sigmoid≈0.27：低于强阈值 → 低置信
    assert agent._is_low_confidence([{"id": "S1", "score": -1.0}]) is True
    assert agent._is_low_confidence([{"id": "S1", "score": None}]) is False  # 无可比分数


def test_low_confidence_count_gate(make_agent):
    import dataclasses

    agent = make_agent()
    # 要求至少 2 条够格证据
    agent.settings = dataclasses.replace(
        agent.settings,
        retrieval=dataclasses.replace(agent.settings.retrieval, min_confident_sources=2),
    )
    # 只有 1 条强分 → 数量不足 → 低置信
    assert agent._is_low_confidence([{"id": "S1", "score": 5.0}]) is True
    # 两条都够格（sigmoid(5)=0.99, sigmoid(2)=0.88 均 ≥0.35）→ 通过
    assert agent._is_low_confidence(
        [{"id": "S1", "score": 5.0}, {"id": "S2", "score": 2.0}]
    ) is False


def test_answerability_gate_rejects_no_sources(make_agent):
    agent = make_agent()
    ok, reason = agent._answerability_status([])
    assert ok is False
    assert "有效来源不足" in reason
    assert "未检索到充分依据" in agent._insufficient_evidence_answer(reason, [])


def test_answerability_gate_requires_citation_after_generation(make_agent):
    agent = make_agent()
    sources = [{"id": "S1", "score": 1.0, "snippet": "relevant evidence"}]
    ok, reason = agent._answerability_status(sources, valid_cited=[])
    assert ok is False
    assert "引用" in reason
    ok, _ = agent._answerability_status(sources, valid_cited=["S1"])
    assert ok is True
