"""ingest_arxiv_papers 工具 + 按工具超时覆盖的离线测试。全程 mock，不打网络/arxiv/嵌入。"""
from __future__ import annotations

import dataclasses
import time

import pytest

import tools.arxiv_ingest_tool as ingest_mod
from llm.model import ToolCall
from retrieval.chunker import Chunk
from retrieval.retriever import RetrievalResult
from tools.arxiv_ingest_tool import ArxivIngestTool


def _result(aid: str, score: float = 1.0) -> RetrievalResult:
    c = Chunk(
        chunk_id=f"{aid}::0", paper_id=aid, paper_title=f"Paper {aid}",
        section="Method", content=f"全文内容 {aid}", source=f"/x/{aid}.pdf",
    )
    return RetrievalResult(chunk=c, score=score)


@pytest.fixture
def make_tool(settings, tmp_path, monkeypatch):
    # registry 刷新 / 自动淘汰按 BASE_DIR 推导真实目录，测试中中性化，避免碰真实 data/papers/arxiv
    monkeypatch.setattr(ingest_mod, "touch_papers", lambda *a, **k: None)
    monkeypatch.setattr(ingest_mod, "prune_collection", lambda *a, **k: [])

    def _make(**arxiv_overrides):
        s = settings
        if arxiv_overrides:
            s = dataclasses.replace(
                settings, arxiv=dataclasses.replace(settings.arxiv, **arxiv_overrides)
            )
        tool = ArxivIngestTool(s)
        tool.data_dir = tmp_path / "papers"   # 隔离到 tmp，避免碰真实 data/papers/arxiv
        tool.index_dir = tmp_path / "idx"
        tool.data_dir.mkdir(parents=True, exist_ok=True)
        return tool

    return _make


def test_cap_download_build_and_sources(make_tool, monkeypatch):
    tool = make_tool(max_ingest_papers=2)
    attempted: list[str] = []

    def fake_dl(aid, target):
        attempted.append(aid)
        target.write_bytes(b"%PDF-1.4 fake")
        return True

    builds = {"n": 0}
    monkeypatch.setattr(tool, "_ensure_downloaded", fake_dl)
    monkeypatch.setattr(ingest_mod, "build_collection", lambda *a, **k: builds.__setitem__("n", builds["n"] + 1))
    monkeypatch.setattr(tool, "_search", lambda q: [_result("A"), _result("B")])

    out = tool.run(["A", "B", "C", "D", "E"], "transformer 注意力", _id_base=3)

    assert attempted == ["A", "B"]          # 受 max_ingest_papers=2 上限约束
    assert builds["n"] == 1                  # 增量重建仅一次
    assert [s["id"] for s in out.sources] == ["S4", "S5"]  # _id_base 偏移
    assert all(s["collection"] == "arxiv" for s in out.sources)  # 指向全文集合供预览
    assert "新入库 2 篇" in out.text


def test_already_present_is_reused_not_redownloaded(make_tool, monkeypatch):
    tool = make_tool(max_ingest_papers=3)
    (tool.data_dir / "A.pdf").write_bytes(b"%PDF old")  # A 已在库
    downloaded: list[str] = []

    def fake_dl(aid, target):
        downloaded.append(aid)
        target.write_bytes(b"%PDF new")
        return True

    monkeypatch.setattr(tool, "_ensure_downloaded", fake_dl)
    monkeypatch.setattr(ingest_mod, "build_collection", lambda *a, **k: None)
    monkeypatch.setattr(tool, "_search", lambda q: [_result("A")])

    out = tool.run(["A", "B"], "q")
    assert downloaded == ["B"]               # A 复用，不重新下载
    assert "复用 1 篇" in out.text


def test_all_download_fail_no_build_no_sources(make_tool, monkeypatch):
    tool = make_tool()
    monkeypatch.setattr(tool, "_ensure_downloaded", lambda aid, target: False)
    called = {"build": False}
    monkeypatch.setattr(ingest_mod, "build_collection", lambda *a, **k: called.__setitem__("build", True))
    monkeypatch.setattr(tool, "_search", lambda q: [])

    out = tool.run(["A", "B"], "q")
    assert called["build"] is False          # 无 PDF 落地，不触发重建
    assert out.sources == []
    assert "下载失败" in out.text


def test_empty_input_short_circuits(make_tool):
    tool = make_tool()
    assert tool.run([], "q").sources == []
    assert tool.run(["A"], "  ").sources == []


def test_normalize_ids_dedup_keeps_order():
    assert ArxivIngestTool._normalize_ids(["A", " A ", "B", "", "C"]) == ["A", "B", "C"]


# ---------- harness：按工具覆盖超时 ----------
class SlowTimeoutTool:
    name = "slow"
    timeout_seconds = 0.1  # 远小于默认 30s

    @staticmethod
    def schema():
        return {"name": "slow", "input_schema": {"type": "object", "properties": {}}}

    def run(self, _id_base=0, **kw):
        time.sleep(1.0)
        from tools.base import ToolResult
        return ToolResult(text="late", sources=[])


def test_per_tool_timeout_override(make_agent):
    agent = make_agent(tools={"slow": SlowTimeoutTool()})
    out = agent._run_tool(ToolCall("1", "slow", {}), [], [], [])
    assert out.is_error and "超时" in out.content  # 用了工具自带的 0.1s 而非默认 30s
