"""统一论文库的容量治理：按龄期和 LRU 淘汰论文。"""
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
from indexing.build_index import build_index

REGISTRY_NAME = ".registry.json"
_INDEX_FILES = ("vectors.npy", "metadata.pkl", "faiss.index", "manifest.json")


def _data_dir(settings: Settings) -> Path:
    return BASE_DIR / settings.project.data_root


def _registry_path(data_dir: Path) -> Path:
    return data_dir / REGISTRY_NAME


def load_registry(data_dir: Path) -> dict:
    path = _registry_path(data_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_registry(data_dir: Path, registry: dict) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    _registry_path(data_dir).write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sync_registry(
    registry: dict, pdfs: dict[str, Path], now: float
) -> dict:
    """只同步已登记的在线论文；用户手动放入的论文不参与自动淘汰。"""
    registry = {key: value for key, value in registry.items() if key in pdfs}
    for paper_id in registry:
        registry[paper_id].setdefault("added_at", now)
        registry[paper_id].setdefault(
            "last_used_at", registry[paper_id]["added_at"]
        )
    return registry


def touch_papers(
    settings: Settings, paper_ids, now: float | None = None
) -> None:
    """刷新论文的最后使用时间。"""
    now = now or time.time()
    data_dir = _data_dir(settings)
    registry = load_registry(data_dir)
    for paper_id in paper_ids:
        entry = registry.setdefault(paper_id, {"added_at": now})
        entry["added_at"] = entry.get("added_at", now)
        entry["last_used_at"] = now
    save_registry(data_dir, registry)


def plan_eviction(
    entries: list[tuple[str, float]],
    max_papers: int,
    max_age_days: int,
    protect: set[str],
    now: float,
) -> list[str]:
    """返回需要淘汰的 paper_id。"""
    evict: set[str] = set()
    if max_age_days > 0:
        cutoff = now - max_age_days * 86400
        evict.update(
            paper_id
            for paper_id, last_used in entries
            if paper_id not in protect and last_used < cutoff
        )
    if max_papers > 0:
        over = len(entries) - len(evict) - max_papers
        if over > 0:
            survivors = sorted(
                (
                    entry
                    for entry in entries
                    if entry[0] not in evict and entry[0] not in protect
                ),
                key=lambda item: item[1],
            )
            evict.update(paper_id for paper_id, _ in survivors[:over])
    return sorted(evict)


def prune_library(
    settings: Settings,
    protect=None,
    now: float | None = None,
    dry_run: bool = False,
) -> list[str]:
    """按配置治理统一论文库，返回淘汰的 paper_id。"""
    protect = set(protect or [])
    now = now or time.time()
    data_dir = _data_dir(settings)
    if not data_dir.exists():
        return []

    all_pdfs = {path.stem: path for path in data_dir.glob("*.pdf")}
    registry = _sync_registry(load_registry(data_dir), all_pdfs, now)
    pdfs = {
        paper_id: all_pdfs[paper_id]
        for paper_id in registry
        if paper_id in all_pdfs
    }
    entries = [
        (paper_id, registry[paper_id]["last_used_at"])
        for paper_id in pdfs
    ]
    evict = plan_eviction(
        entries,
        settings.arxiv.max_papers,
        settings.arxiv.max_age_days,
        protect,
        now,
    )
    if dry_run or not evict:
        save_registry(data_dir, registry)
        return evict

    for paper_id in evict:
        pdfs[paper_id].unlink(missing_ok=True)
        registry.pop(paper_id, None)
    save_registry(data_dir, registry)

    index_dir = BASE_DIR / settings.index.index_root
    if any(data_dir.glob("*.pdf")):
        build_index(settings, incremental=True)
    else:
        for name in _INDEX_FILES:
            (index_dir / name).unlink(missing_ok=True)
        for pattern in ("vectors-*.npy", "metadata-*.pkl", "faiss-*.index"):
            for path in index_dir.glob(pattern):
                path.unlink(missing_ok=True)
    return evict


def main() -> None:
    parser = argparse.ArgumentParser(description="论文库容量治理")
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--max-age-days", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    if args.max_papers is not None or args.max_age_days is not None:
        settings = dataclasses.replace(
            settings,
            arxiv=dataclasses.replace(
                settings.arxiv,
                max_papers=(
                    args.max_papers
                    if args.max_papers is not None
                    else settings.arxiv.max_papers
                ),
                max_age_days=(
                    args.max_age_days
                    if args.max_age_days is not None
                    else settings.arxiv.max_age_days
                ),
            ),
        )

    evicted = prune_library(settings, dry_run=args.dry_run)
    action = "将淘汰" if args.dry_run else "已淘汰"
    print(f"[Prune] {action} {len(evicted)} 篇：{evicted or '无'}")


if __name__ == "__main__":
    main()
