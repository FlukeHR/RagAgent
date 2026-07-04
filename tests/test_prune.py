"""arxiv 全文集合淘汰的离线测试：LRU / 龄期 / protect / registry 同步 / 端到端。"""
from __future__ import annotations

import dataclasses

import indexing.prune as prune_mod
from indexing.prune import (
    load_registry,
    plan_eviction,
    prune_collection,
    save_registry,
    touch_papers,
)

DAY = 86400


# ---------- plan_eviction 纯函数 ----------
def test_lru_evicts_least_recently_used():
    now = 1000 * DAY
    entries = [("A", now - 5 * DAY), ("B", now - 1 * DAY), ("C", now - 3 * DAY)]
    # 容量上限 2 → 淘汰最久未用的 A
    out = plan_eviction(entries, max_papers=2, max_age_days=0, protect=set(), now=now)
    assert out == ["A"]


def test_protect_never_evicted():
    now = 1000 * DAY
    entries = [("A", now - 5 * DAY), ("B", now - 1 * DAY), ("C", now - 3 * DAY)]
    # A 最久未用，但被保护 → 改淘汰次久的 C
    out = plan_eviction(entries, max_papers=2, max_age_days=0, protect={"A"}, now=now)
    assert out == ["C"]


def test_age_eviction():
    now = 1000 * DAY
    entries = [("A", now - 100 * DAY), ("B", now - 1 * DAY)]
    out = plan_eviction(entries, max_papers=0, max_age_days=90, protect=set(), now=now)
    assert out == ["A"]  # A 超 90 天未用


def test_zero_limits_disable_eviction():
    now = 1000 * DAY
    entries = [("A", now - 999 * DAY), ("B", now - 1 * DAY)]
    assert plan_eviction(entries, 0, 0, set(), now) == []  # 两个维度都不限


def test_age_and_capacity_combined():
    now = 1000 * DAY
    entries = [
        ("old", now - 200 * DAY),   # 超龄
        ("A", now - 4 * DAY),
        ("B", now - 3 * DAY),
        ("C", now - 1 * DAY),
    ]
    # 先龄期淘汰 old，再容量(=2)对剩 3 个淘汰最久未用的 A
    out = plan_eviction(entries, max_papers=2, max_age_days=90, protect=set(), now=now)
    assert set(out) == {"old", "A"}


# ---------- registry 读写 + touch ----------
def test_touch_updates_last_used(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(prune_mod, "BASE_DIR", tmp_path)
    data_dir = tmp_path / settings.project.data_root / "arxiv"
    data_dir.mkdir(parents=True)
    t0 = 100.0
    touch_papers(settings, "arxiv", ["X"], now=t0)
    reg = load_registry(data_dir)
    assert reg["X"]["added_at"] == t0 and reg["X"]["last_used_at"] == t0
    touch_papers(settings, "arxiv", ["X"], now=t0 + 50)
    reg = load_registry(data_dir)
    assert reg["X"]["added_at"] == t0 and reg["X"]["last_used_at"] == t0 + 50  # added 不变，last_used 刷新


# ---------- prune_collection 端到端 ----------
def _mk_settings(settings, **arxiv):
    return dataclasses.replace(settings, arxiv=dataclasses.replace(settings.arxiv, **arxiv))


def test_prune_collection_deletes_and_rebuilds(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(prune_mod, "BASE_DIR", tmp_path)
    s = _mk_settings(settings, max_collection_papers=2, max_age_days=0)
    data_dir = tmp_path / s.project.data_root / "arxiv"
    data_dir.mkdir(parents=True)
    now = 1000 * DAY
    reg = {}
    for pid, age in [("A", 5), ("B", 1), ("C", 3)]:
        (data_dir / f"{pid}.pdf").write_bytes(b"%PDF")
        reg[pid] = {"added_at": now - age * DAY, "last_used_at": now - age * DAY}
    save_registry(data_dir, reg)

    builds = {"n": 0}
    monkeypatch.setattr(prune_mod, "build_collection", lambda *a, **k: builds.__setitem__("n", builds["n"] + 1))

    evicted = prune_collection(s, "arxiv", protect={"B"}, now=now)
    assert evicted == ["A"]                              # 容量2 + 保护B → 淘汰最久的 A
    assert not (data_dir / "A.pdf").exists()             # 文件已删
    assert (data_dir / "B.pdf").exists() and (data_dir / "C.pdf").exists()
    assert "A" not in load_registry(data_dir)            # registry 已剔除
    assert builds["n"] == 1                              # 触发了一次增量重建


def test_prune_legacy_pdf_without_registry(settings, tmp_path, monkeypatch):
    """历史遗留 PDF（registry 无条目）应按 mtime 补条目，仍可参与淘汰。"""
    monkeypatch.setattr(prune_mod, "BASE_DIR", tmp_path)
    s = _mk_settings(settings, max_collection_papers=1, max_age_days=0)
    data_dir = tmp_path / s.project.data_root / "arxiv"
    data_dir.mkdir(parents=True)
    old = data_dir / "OLD.pdf"; old.write_bytes(b"%PDF")
    new = data_dir / "NEW.pdf"; new.write_bytes(b"%PDF")
    import os
    os.utime(old, (1000, 1000))            # OLD 更久
    os.utime(new, (2_000_000_000, 2_000_000_000))
    monkeypatch.setattr(prune_mod, "build_collection", lambda *a, **k: None)

    evicted = prune_collection(s, "arxiv")
    assert evicted == ["OLD"]              # 无 registry 也能按 mtime LRU 淘汰最久的


def test_prune_noop_when_under_cap(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(prune_mod, "BASE_DIR", tmp_path)
    s = _mk_settings(settings, max_collection_papers=10, max_age_days=0)
    data_dir = tmp_path / s.project.data_root / "arxiv"
    data_dir.mkdir(parents=True)
    (data_dir / "A.pdf").write_bytes(b"%PDF")
    called = {"build": False}
    monkeypatch.setattr(prune_mod, "build_collection", lambda *a, **k: called.__setitem__("build", True))
    assert prune_collection(s, "arxiv") == []      # 未超量
    assert called["build"] is False                # 不重建
