"""评估历史记录的离线测试：追加 / 读取 / 过滤 / 增量对比 / 配置快照。"""
from __future__ import annotations

from evaluation.results_log import (
    config_snapshot,
    diff_metrics,
    load_history,
    record_run,
)


def test_record_and_load_roundtrip(settings, tmp_path):
    log = tmp_path / "history.jsonl"
    rec = record_run("retrieval", "qasper.json", 144,
                     {"hit@5": 0.7222, "mrr": 0.4959}, settings, log_path=log)
    assert rec["kind"] == "retrieval" and rec["n"] == 144
    assert rec["metrics"]["hit@5"] == 0.7222
    assert "git" in rec and "config" in rec and rec["config"]["top_n_rerank"] == settings.index.top_n_rerank

    rows = load_history(log)
    assert len(rows) == 1 and rows[0]["dataset"] == "qasper.json"


def test_load_filters_by_kind(settings, tmp_path):
    log = tmp_path / "history.jsonl"
    record_run("retrieval", "q", 1, {"mrr": 0.5}, settings, log_path=log)
    record_run("generation", "g", 1, {"faithfulness": 0.8}, settings, log_path=log)
    record_run("retrieval", "q", 1, {"mrr": 0.6}, settings, log_path=log)

    assert len(load_history(log)) == 3
    ret = load_history(log, kind="retrieval")
    assert len(ret) == 2 and all(r["kind"] == "retrieval" for r in ret)
    assert len(load_history(log, kind="generation")) == 1


def test_load_missing_file_is_empty(tmp_path):
    assert load_history(tmp_path / "nope.jsonl") == []


def test_diff_metrics():
    prev = {"faithfulness": 0.80, "answer_relevancy": 0.40}
    cur = {"faithfulness": 0.84, "answer_relevancy": 0.33, "context_precision": 0.66}
    d = diff_metrics(prev, cur)
    assert d["faithfulness"] == (0.84, 0.04)
    assert d["answer_relevancy"] == (0.33, -0.07)
    assert d["context_precision"] == (0.66, None)  # 新指标，上次没有


def test_config_snapshot_has_key_knobs(settings):
    snap = config_snapshot(settings)
    for key in ("chunk_size", "top_n_rerank", "use_cross_encoder",
                "low_confidence_threshold", "provider", "model_name"):
        assert key in snap
