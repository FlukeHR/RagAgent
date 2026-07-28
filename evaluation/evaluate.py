"""Unified entry point for the project's key offline and online evaluations.

Examples::

    python evaluation/evaluate.py --profile smoke
    python evaluation/evaluate.py --profile key
    python evaluation/evaluate.py --profile key --yes

Without ``--yes`` only deterministic/local stages run.  Supplying ``--yes``
also runs generation evaluation and E2E evaluation against ``--e2e-url``.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_ROOT = PROJECT_ROOT / "evaluation"
DEFAULT_RESULTS = EVALUATION_ROOT / "results" / "latest"


@dataclass(frozen=True)
class Stage:
    """One bounded evaluation process and its expected report."""

    name: str
    command: list[str]
    output: Path
    paid: bool = False


def _script(name: str) -> str:
    return str(EVALUATION_ROOT / name)


def _profile_values(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "qasper_limit": 10,
            "qasper_top_k": "12",
            "qasper_top_n": "5",
            "finance_limit": 10,
            "chunk_sizes": "900",
            "overlap_ratios": "0.15",
            "top_k": "5,10",
            "generation_limit": 5,
            "e2e_repeat": 1,
            "omnidoc_limit": 20,
        }
    if profile == "key":
        return {
            "qasper_limit": 0,
            "qasper_top_k": "12",
            "qasper_top_n": "5",
            "finance_limit": 150,
            "chunk_sizes": "900,1400",
            "overlap_ratios": "0.10,0.15",
            "top_k": "5,10,20",
            "generation_limit": 20,
            "e2e_repeat": 3,
            "omnidoc_limit": 0,
        }
    return {
        "qasper_limit": 0,
        "qasper_top_k": "8,12,24",
        "qasper_top_n": "3,5,8",
        "finance_limit": 0,
        "chunk_sizes": "500,900,1400",
        "overlap_ratios": "0.10,0.15,0.20",
        "top_k": "5,10,20",
        "generation_limit": 20,
        "e2e_repeat": 5,
        "omnidoc_limit": 0,
    }


def build_stages(args: argparse.Namespace) -> list[Stage]:
    """Build the selected profile without authorizing paid work implicitly."""

    values = _profile_values(args.profile)
    output = args.output_dir.resolve()
    python = sys.executable
    stages = [
        Stage(
            "qasper_retrieval",
            [
                python,
                _script("benchmark_retrieval.py"),
                "--limit",
                str(values["qasper_limit"]),
                "--top-k",
                values["qasper_top_k"],
                "--top-n",
                values["qasper_top_n"],
                "--output",
                str(output / "qasper_retrieval.json"),
            ],
            output / "qasper_retrieval.json",
        ),
        Stage(
            "financebench",
            [
                python,
                _script("eval_financebench.py"),
                str(EVALUATION_ROOT / "data" / "financebench"),
                "--limit",
                str(values["finance_limit"]),
                "--chunk-sizes",
                values["chunk_sizes"],
                "--overlap-ratios",
                values["overlap_ratios"],
                "--top-k",
                values["top_k"],
                "--mode",
                "hybrid",
                "--output",
                str(output / "financebench.json"),
            ],
            output / "financebench.json",
        ),
    ]
    if args.omnidoc or args.profile == "full":
        command = [
            python,
            _script("eval_omnidocbench.py"),
            str(EVALUATION_ROOT / "data" / "omnidocbench" / "OmniDocBench.json"),
            "--images-root",
            str(EVALUATION_ROOT / "data" / "omnidocbench" / "images"),
            "--language",
            "english",
            "--data-source",
            "academic_literature",
            "--ocr",
            "--output",
            str(output / "omnidocbench.json"),
        ]
        if values["omnidoc_limit"]:
            command.extend(["--limit", str(values["omnidoc_limit"])])
        stages.append(Stage("omnidocbench", command, output / "omnidocbench.json"))
    if args.yes:
        stages.extend(
            [
                Stage(
                    "generation",
                    [
                        python,
                        _script("eval_generation.py"),
                        "--limit",
                        str(values["generation_limit"]),
                        "--yes",
                        "--output",
                        str(output / "generation.json"),
                    ],
                    output / "generation.json",
                    paid=True,
                ),
                Stage(
                    "e2e",
                    [
                        python,
                        _script("benchmark_e2e.py"),
                        "--url",
                        args.e2e_url,
                        "--repeat",
                        str(values["e2e_repeat"]),
                        "--yes",
                        "--output",
                        str(output / "e2e.json"),
                    ],
                    output / "e2e.json",
                    paid=True,
                ),
            ]
        )
    skip = {value.strip() for value in args.skip.split(",") if value.strip()}
    return [stage for stage in stages if stage.name not in skip]


def _run_stage(stage: Stage) -> dict[str, Any]:
    started = time.perf_counter()
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    print(f"\n[{stage.name}] {' '.join(stage.command)}", flush=True)
    completed = subprocess.run(
        stage.command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
    )
    return {
        "name": stage.name,
        "status": "passed" if completed.returncode == 0 and stage.output.is_file() else "failed",
        "returncode": completed.returncode,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "paid": stage.paid,
        "output": str(stage.output),
    }


def _load(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _summarize_qasper(payload: dict[str, Any]) -> dict[str, Any]:
    runs = payload.get("runs") or [payload.get("metrics", {})]
    comparable = [run for run in runs if isinstance(run, dict)]
    best = max(comparable, key=lambda run: float(run.get("mrr", 0.0)), default={})
    metric_k = int(best.get("top_n_rerank") or 0)
    return {
        "samples": best.get("questions"),
        "best_mode_by_mrr": best.get("mode"),
        "reranker": best.get("reranker"),
        "reranker_backend": best.get("reranker_backend"),
        "top_k_recall": best.get("top_k_recall"),
        "top_n_rerank": metric_k,
        "hit": best.get(f"hit@{metric_k}"),
        "mrr": best.get("mrr"),
        "ndcg": best.get(f"ndcg@{metric_k}"),
        "recall": best.get(f"recall@{metric_k}"),
    }


def _summarize_finance(payload: dict[str, Any]) -> dict[str, Any]:
    runs = [run for run in payload.get("runs", []) if isinstance(run, dict)]
    best = max(
        runs,
        key=lambda run: (
            float(run.get("evidence_hit_rate", 0.0)),
            float(run.get("mrr", 0.0)),
        ),
        default={},
    )
    return {
        "samples": payload.get("sample_count"),
        "pdfs": payload.get("pdf_count"),
        "parsing": payload.get("parsing"),
        "embedding_backend": (payload.get("embedding") or {}).get("backend"),
        "reranker_backend": payload.get("reranker_backend"),
        "best_by_evidence_hit_then_mrr": {
            key: best.get(key)
            for key in (
                "chunk_size",
                "chunk_overlap",
                "top_k",
                "paper_hit_rate",
                "page_hit_rate",
                "evidence_hit_rate",
                "evidence_recall",
                "mrr",
            )
        },
    }


def _summarize_generation(payload: dict[str, Any]) -> dict[str, Any]:
    metrics: Any = payload.get("metrics")
    if isinstance(metrics, str):
        try:
            metrics = ast.literal_eval(metrics)
        except (SyntaxError, ValueError):
            pass
    usage = payload.get("usage", [])
    return {
        "samples": payload.get("sample_count"),
        "metrics": metrics,
        "answer_generation_input_tokens": sum(item.get("input_tokens", 0) for item in usage),
        "answer_generation_output_tokens": sum(item.get("output_tokens", 0) for item in usage),
        "judge_usage": payload.get("judge_usage"),
    }


def build_key_metrics(output_dir: Path, stages: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a compact cross-stage report from successfully written artifacts."""

    metrics: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stages": stages,
        "results": {},
    }
    loaders = {
        "qasper_retrieval": ("qasper_retrieval.json", _summarize_qasper),
        "financebench": ("financebench.json", _summarize_finance),
        "generation": ("generation.json", _summarize_generation),
        "e2e": ("e2e.json", lambda payload: payload.get("summary", {})),
        "citation_audit": (
            "citation_audit.summary.json",
            lambda payload: payload,
        ),
        "omnidocbench": (
            "omnidocbench.json",
            lambda payload: {
                "pages": payload.get("pages"),
                "empty_prediction_rate": payload.get("empty_prediction_rate"),
                "metrics": payload.get("metrics"),
            },
        ),
    }
    for name, (filename, summarize) in loaders.items():
        payload = _load(output_dir / filename)
        if payload is not None:
            metrics["results"][name] = summarize(payload)
    return metrics


def _run_audit_if_possible(output_dir: Path, stages: list[dict[str, Any]]) -> None:
    e2e = output_dir / "e2e.json"
    e2e_passed = any(
        stage["name"] == "e2e" and stage["status"] == "passed" for stage in stages
    )
    if not e2e_passed or not e2e.is_file():
        return
    stage = Stage(
        "citation_audit",
        [
            sys.executable,
            _script("audit_citations.py"),
            str(e2e),
            "--output",
            str(output_dir / "citation_audit.csv"),
            "--summary-output",
            str(output_dir / "citation_audit.summary.json"),
        ],
        output_dir / "citation_audit.csv",
    )
    stages.append(_run_stage(stage))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="一次运行关键检索、PDF、生成、E2E 与引用审计指标",
    )
    parser.add_argument("--profile", choices=("smoke", "key", "full"), default="key")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS,
        help="本次评测的统一输出目录",
    )
    parser.add_argument("--yes", action="store_true", help="显式授权生成和 E2E 的真实 API 调用")
    parser.add_argument("--e2e-url", default="http://127.0.0.1:8000/ask")
    parser.add_argument("--omnidoc", action="store_true", help="key/smoke 额外运行本地 OCR 页面评测")
    parser.add_argument(
        "--skip",
        default="",
        help="逗号分隔：qasper_retrieval,financebench,omnidocbench,generation,e2e",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = build_stages(args)
    if not args.yes:
        print("未提供 --yes：只运行不调用付费 LLM API 的本地评测。")
    stage_reports = [_run_stage(stage) for stage in selected]
    _run_audit_if_possible(args.output_dir.resolve(), stage_reports)
    combined = build_key_metrics(args.output_dir.resolve(), stage_reports)
    combined_path = args.output_dir.resolve() / "key_metrics.json"
    combined_path.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    failures = [stage["name"] for stage in stage_reports if stage["status"] == "failed"]
    print(f"\n统一报告：{combined_path}")
    if failures:
        raise SystemExit(f"以下评测阶段失败：{', '.join(failures)}")


if __name__ == "__main__":
    main()
