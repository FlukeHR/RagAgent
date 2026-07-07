"""评估历史记录：把每次 eval 的「代码版本 + 配置快照 + 指标」追加成一行，便于回溯改进。

记录落在 evaluation/results/history.jsonl（append-only，每行一次 run）。检索侧 / 生成侧共用
同一 schema，用 kind 区分。配合 eval_qasper.py / eval_generation.py / eval_pdf_grounding.py
的 --record 开关使用。

CLI：
    python3 evaluation/results_log.py                 # 打印全部历史
    python3 evaluation/results_log.py --kind generation --last 10
    python3 evaluation/results_log.py --compare       # 对比每个 kind 的最新两次（回归门禁视角）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_LOG = PROJECT_ROOT / "evaluation" / "results" / "history.jsonl"


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - 非 git 环境 / 无 git 时降级
        return ""


def git_info() -> dict[str, Any]:
    """当前提交短哈希、分支、工作区是否有未提交改动。"""
    return {
        "git": _git("rev-parse", "--short", "HEAD") or "unknown",
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "dirty": bool(_git("status", "--porcelain")),
    }


def config_snapshot(settings) -> dict[str, Any]:
    """记录影响指标的关键配置，使分数能归因到具体改动。"""
    s = settings
    return {
        "chunk_size": s.index.chunk_size,
        "chunk_overlap": s.index.chunk_overlap,
        "top_k_recall": s.index.top_k_recall,
        "top_n_rerank": s.index.top_n_rerank,
        "embedding_model": s.embedding.model_name,
        "use_sentence_transformers": s.embedding.use_sentence_transformers,
        "rerank_model": s.rerank.model_name,
        "use_cross_encoder": s.rerank.use_cross_encoder,
        "low_confidence_threshold": s.retrieval.low_confidence_threshold,
        "weak_confidence_threshold": s.retrieval.weak_confidence_threshold,
        "min_confident_sources": s.retrieval.min_confident_sources,
        "answerability_min_sources": s.retrieval.answerability_min_sources,
        "answerability_min_score": s.retrieval.answerability_min_score,
        "answerability_require_citation": s.retrieval.answerability_require_citation,
        "pdf_parse_provider": s.pdf_parse.provider,
        "pdf_parse_auto_ocr": s.pdf_parse.auto_ocr,
        "image_search_enabled": s.image_search.enabled,
        "provider": s.llm.provider,
        "model_name": s.llm.model_name,
        "effort": s.llm.effort,
        "max_tool_iters": s.llm.max_tool_iters,
    }


def record_run(
    kind: str,
    dataset: str,
    n: int,
    metrics: dict[str, float],
    settings=None,
    note: str = "",
    log_path: Path = DEFAULT_LOG,
) -> dict[str, Any]:
    """追加一条评估记录并返回它。kind: retrieval / generation / pdf_grounding。"""
    rec: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **git_info(),
        "kind": kind,
        "dataset": dataset,
        "n": n,
        "config": config_snapshot(settings) if settings is not None else {},
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "note": note,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def load_history(log_path: Path = DEFAULT_LOG, kind: str | None = None) -> list[dict]:
    if not log_path.exists():
        return []
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if kind is None or rec.get("kind") == kind:
            rows.append(rec)
    return rows


def diff_metrics(prev: dict, cur: dict) -> dict[str, tuple[float, float]]:
    """返回 {指标: (当前值, 相对上次的增量)}；上次缺该指标则增量为 None。"""
    out = {}
    for k, v in cur.items():
        pv = prev.get(k)
        out[k] = (v, (round(v - pv, 4) if isinstance(pv, (int, float)) else None))
    return out


def _fmt_delta(d: float | None) -> str:
    if d is None:
        return "  (新)"
    sign = "+" if d >= 0 else ""
    return f" ({sign}{d})"


def _print_history(rows: list[dict], last: int | None) -> None:
    rows = rows[-last:] if last else rows
    if not rows:
        print("（暂无评估记录）")
        return
    for r in rows:
        m = " ".join(f"{k}={v}" for k, v in r.get("metrics", {}).items())
        dirty = "*" if r.get("dirty") else ""
        print(f"{r['ts']} [{r['kind']}] {r.get('git')}{dirty}@{r.get('branch')} "
              f"n={r.get('n')} {r.get('dataset')} :: {m}")


def _print_compare(log_path: Path) -> None:
    for kind in ("retrieval", "generation", "pdf_grounding"):
        rows = load_history(log_path, kind)
        print(f"\n===== {kind} 最新对比 =====")
        if len(rows) < 1:
            print("（无记录）")
            continue
        cur = rows[-1]
        prev = rows[-2] if len(rows) >= 2 else {}
        print(f"最新 {cur['ts']} {cur.get('git')}@{cur.get('branch')} n={cur.get('n')}")
        if prev:
            print(f"上次 {prev['ts']} {prev.get('git')}")
        for k, (v, d) in diff_metrics(prev.get("metrics", {}), cur.get("metrics", {})).items():
            print(f"  {k:<40} {v}{_fmt_delta(d)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="查看 / 对比评估历史记录")
    parser.add_argument("--kind", choices=["retrieval", "generation", "pdf_grounding"], default=None)
    parser.add_argument("--last", type=int, default=None, help="只看最近 N 条")
    parser.add_argument("--compare", action="store_true", help="对比每个 kind 的最新两次")
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    args = parser.parse_args()

    log_path = Path(args.log)
    if args.compare:
        _print_compare(log_path)
    else:
        _print_history(load_history(log_path, args.kind), args.last)


if __name__ == "__main__":
    main()
