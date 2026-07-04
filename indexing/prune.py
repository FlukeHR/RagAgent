"""arXiv 全文集合的容量治理：LRU + 龄期淘汰，复用增量索引删除已淘汰论文的 chunk。

- registry（`<collection>/.registry.json`）记录每篇论文的 added_at / last_used_at；
  入库与被检索命中时刷新 last_used_at（见 tools/arxiv_ingest_tool.py）。
- 淘汰 = 删 PDF + 增量重建（plan_incremental 自动剔除已删文件的 chunk）。
- 既可由 ingest 入库后自动调用（protect 本轮用到的论文），也可 CLI 手动调用。

CLI：python3 indexing/prune.py [collection] [--max-papers N] [--max-age-days D] [--dry-run]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import BASE_DIR, Settings, load_settings
from indexing.build_index import build_collection

REGISTRY_NAME = ".registry.json"
_INDEX_FILES = ("vectors.npy", "metadata.pkl", "faiss.index", "manifest.json")


# ---------- registry 读写 ----------
def _registry_path(data_dir: Path) -> Path:
    return data_dir / REGISTRY_NAME


def load_registry(data_dir: Path) -> dict:
    p = _registry_path(data_dir)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_registry(data_dir: Path, reg: dict) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    _registry_path(data_dir).write_text(
        json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _sync_registry(reg: dict, pdfs: dict[str, Path], now: float) -> dict:
    """对齐 registry 与磁盘：删去磁盘已无的条目，给缺失的（如历史遗留 PDF）按 mtime 补条目。"""
    reg = {k: v for k, v in reg.items() if k in pdfs}
    for pid, path in pdfs.items():
        if pid not in reg:
            mt = path.stat().st_mtime
            reg[pid] = {"added_at": mt, "last_used_at": mt}
        else:
            reg[pid].setdefault("added_at", now)
            reg[pid].setdefault("last_used_at", reg[pid]["added_at"])
    return reg


def touch_papers(settings: Settings, collection: str, paper_ids, now: float | None = None) -> None:
    """刷新若干论文的 last_used_at（入库 / 被检索命中时调用），并补 added_at。"""
    now = now or time.time()
    data_dir = BASE_DIR / settings.project.data_root / collection
    reg = load_registry(data_dir)
    for pid in paper_ids:
        e = reg.setdefault(pid, {"added_at": now})
        e["added_at"] = e.get("added_at", now)
        e["last_used_at"] = now
    save_registry(data_dir, reg)


# ---------- 淘汰规划（纯函数，便于测试） ----------
def plan_eviction(
    entries: list[tuple[str, float]],
    max_papers: int,
    max_age_days: int,
    protect: set[str],
    now: float,
) -> list[str]:
    """决定要淘汰的 paper_id。

    entries: [(paper_id, last_used_at), ...]（全部在库论文）。
    先按龄期（超 max_age_days 未用）淘汰，再按容量 LRU（最久未用优先）补齐到 max_papers。
    protect 内的论文永不淘汰（如本轮刚检索命中）。max_papers/max_age_days 为 0 表示该维度不限。
    """
    evict: set[str] = set()
    if max_age_days and max_age_days > 0:
        cutoff = now - max_age_days * 86400
        for pid, lu in entries:
            if pid not in protect and lu < cutoff:
                evict.add(pid)
    if max_papers and max_papers > 0:
        remaining = len(entries) - len(evict)
        over = remaining - max_papers
        if over > 0:
            survivors = sorted(
                (e for e in entries if e[0] not in evict and e[0] not in protect),
                key=lambda x: x[1],  # last_used_at 升序：最久未用在前
            )
            for pid, _ in survivors[:over]:
                evict.add(pid)
    return sorted(evict)


def prune_collection(
    settings: Settings,
    collection: str,
    protect=None,
    now: float | None = None,
    dry_run: bool = False,
) -> list[str]:
    """按配置淘汰 arxiv 全文集合中超量/过期的论文，返回被淘汰的 paper_id 列表。"""
    protect = set(protect or [])
    now = now or time.time()
    data_dir = BASE_DIR / settings.project.data_root / collection
    if not data_dir.exists():
        return []

    pdfs = {p.stem: p for p in data_dir.glob("*.pdf")}
    reg = _sync_registry(load_registry(data_dir), pdfs, now)
    entries = [(pid, reg[pid]["last_used_at"]) for pid in pdfs]
    evict = plan_eviction(
        entries,
        settings.arxiv.max_collection_papers,
        settings.arxiv.max_age_days,
        protect,
        now,
    )
    if dry_run or not evict:
        save_registry(data_dir, reg)  # 至少把同步后的 registry 落盘
        return evict

    for pid in evict:
        pdfs[pid].unlink(missing_ok=True)
        reg.pop(pid, None)
    save_registry(data_dir, reg)

    index_dir = BASE_DIR / settings.index.index_root / collection
    if any(data_dir.glob("*.pdf")):
        build_collection(settings, collection, incremental=True)
    else:
        # 集合已空：清掉索引产物，避免残留指向已删文件的 chunk
        for name in _INDEX_FILES:
            (index_dir / name).unlink(missing_ok=True)
    return evict


def main() -> None:
    parser = argparse.ArgumentParser(description="arXiv 全文集合容量治理（LRU + 龄期淘汰）")
    parser.add_argument("collection", nargs="?", default=None, help="集合名，默认 arxiv 全文集合")
    parser.add_argument("--max-papers", type=int, default=None, help="覆盖容量上限")
    parser.add_argument("--max-age-days", type=int, default=None, help="覆盖龄期上限（天）")
    parser.add_argument("--dry-run", action="store_true", help="只列出将淘汰的论文，不实际删除")
    args = parser.parse_args()

    settings = load_settings()
    collection = args.collection or settings.arxiv.full_text_collection
    if args.max_papers is not None or args.max_age_days is not None:
        settings = dataclasses.replace(
            settings,
            arxiv=dataclasses.replace(
                settings.arxiv,
                max_collection_papers=args.max_papers
                if args.max_papers is not None
                else settings.arxiv.max_collection_papers,
                max_age_days=args.max_age_days
                if args.max_age_days is not None
                else settings.arxiv.max_age_days,
            ),
        )

    evicted = prune_collection(settings, collection, dry_run=args.dry_run)
    tag = "将淘汰" if args.dry_run else "已淘汰"
    print(f"[Prune] collection={collection} {tag} {len(evicted)} 篇：{evicted or '无'}")


if __name__ == "__main__":
    main()
