"""Measure end-to-end API behavior, latency, usage, and cost.

The ``/ask`` endpoint currently returns one non-streaming JSON response.  The
benchmark therefore records time-to-response-headers, not time-to-first-token.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for a finite sample."""

    if not values:
        return 0.0
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def looks_mojibake(text: str) -> bool:
    """Detect common UTF-8/GBK text decoded as a Latin single-byte codec."""

    value = text or ""
    if not value:
        return False
    suspicious = sum("\u00c0" <= character <= "\u00ff" for character in value)
    return suspicious / len(value) >= 0.05


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load enabled JSONL cases and reject malformed benchmark definitions."""

    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    enabled = [case for case in cases if case.get("enabled", True)]
    if not enabled:
        raise ValueError(f"没有启用的 E2E 样本：{path}")
    required = {"id", "category", "question", "expected_status"}
    for case in enabled:
        missing = required - case.keys()
        if missing:
            raise ValueError(f"E2E 样本缺少字段 {sorted(missing)}：{case}")
        texts = [str(case.get("question") or "")]
        texts.extend(str(item.get("content") or "") for item in case.get("history", []))
        if any(looks_mojibake(text) for text in texts):
            raise ValueError(f"E2E 样本疑似乱码：{case['id']}")
    return enabled


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def summarize(records: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate metrics and explicit fitness-for-purpose diagnostics."""

    latencies = [float(record["latency_ms"]) for record in records]
    header_times = [float(record["response_headers_ms"]) for record in records]
    statuses = Counter(str(record.get("status") or "missing") for record in records)
    expected_matches = sum(bool(record.get("status_matches_expected")) for record in records)
    answered = [record for record in records if record.get("status") == "answered"]
    cited_answers = [
        record
        for record in answered
        if record.get("sources") and "[S" in str(record.get("answer") or "")
    ]
    enabled_categories = sorted({str(case["category"]) for case in cases})
    has_answerable_case = any(case.get("expected_status") == "answered" for case in cases)
    mojibake_records = sum(looks_mojibake(str(record.get("answer") or "")) for record in records)
    warnings: list[str] = []
    if not has_answerable_case:
        warnings.append("未启用 expected_status=answered 的样本，不能评估问答、引用或检索链")
    if not answered:
        warnings.append("本次没有 answered 响应，引用审计没有可审计对象")
    if mojibake_records:
        warnings.append(f"{mojibake_records} 个响应疑似乱码，本次文本质量结果无效")
    return {
        "requests": len(records),
        "cases": len(cases),
        "categories": enabled_categories,
        "status_distribution": dict(sorted(statuses.items())),
        "expected_status_accuracy": expected_matches / len(records) if records else 0.0,
        "answered_requests": len(answered),
        "cited_answer_requests": len(cited_answers),
        "mean_ms": statistics.mean(latencies) if latencies else 0.0,
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
        "response_headers_p50_ms": percentile(header_times, 0.50),
        "response_headers_p95_ms": percentile(header_times, 0.95),
        "total_input_tokens": sum(int(record["input_tokens"]) for record in records),
        "total_output_tokens": sum(int(record["output_tokens"]) for record in records),
        "mean_input_tokens": (
            statistics.mean(int(record["input_tokens"]) for record in records)
            if records
            else 0.0
        ),
        "mean_output_tokens": (
            statistics.mean(int(record["output_tokens"]) for record in records)
            if records
            else 0.0
        ),
        "total_estimated_cost": sum(float(record["estimated_cost"]) for record in records),
        "mojibake_responses": mojibake_records,
        "valid_for_latency": not bool(mojibake_records),
        "valid_for_answer_quality": has_answerable_case and bool(answered) and not mojibake_records,
        "valid_for_citation_audit": bool(cited_answers) and not mojibake_records,
        "warnings": warnings,
    }


def run_e2e(
    *,
    url: str,
    cases_path: Path,
    repeat: int,
    timeout: float,
) -> dict[str, Any]:
    """Run enabled cases against a live API and return a complete report."""

    import requests

    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    cases = load_cases(cases_path)
    records: list[dict[str, Any]] = []
    for case in cases:
        for repetition in range(repeat):
            payload = {
                "question": case["question"],
                "history": case.get("history", []),
                "session_id": f"bench-{case['id']}-{repetition}",
            }
            started = time.perf_counter()
            response = requests.post(url, json=payload, timeout=timeout, stream=True)
            response_headers_ms = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            raw = b"".join(response.iter_content(chunk_size=65536))
            elapsed_ms = (time.perf_counter() - started) * 1000
            body = json.loads(raw)
            usage = next(
                (
                    event
                    for event in reversed(body.get("trace", []))
                    if event.get("type") == "usage"
                ),
                {},
            )
            status = body.get("status")
            records.append(
                {
                    "case_id": case["id"],
                    "category": case["category"],
                    "repetition": repetition,
                    "expected_status": case["expected_status"],
                    "status": status,
                    "status_matches_expected": status == case["expected_status"],
                    "latency_ms": elapsed_ms,
                    "response_headers_ms": response_headers_ms,
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "estimated_cost": usage.get("estimated_cost", 0),
                    "tool_calls": sum(
                        event.get("type") == "tool" for event in body.get("trace", [])
                    ),
                    "answer": body.get("answer"),
                    "sources": body.get("sources", []),
                    "trace": body.get("trace", []),
                }
            )
    return {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "commit": _git_commit(),
            "url": url,
            "repeat": repeat,
            "cases": str(cases_path),
            "response_mode": "non_streaming_json",
            "latency_note": "response_headers_ms is not model time-to-first-token",
        },
        "summary": summarize(records, cases),
        "records": records,
    }


def save_report(report: dict[str, Any], path: Path) -> None:
    """Save an E2E report as UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="端到端 API 评测")
    parser.add_argument("--url", default="http://127.0.0.1:8000/ask")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("evaluation/data/business_cases.jsonl"),
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--yes", action="store_true", help="确认可能调用真实付费 API")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/e2e_benchmark.json"),
    )
    args = parser.parse_args()
    if not args.yes and os.getenv("RAG_EVAL_ALLOW_API") != "1":
        raise SystemExit("本测试可能调用真实 API；请加 --yes 或设置 RAG_EVAL_ALLOW_API=1")
    try:
        report = run_e2e(
            url=args.url,
            cases_path=args.cases,
            repeat=args.repeat,
            timeout=args.timeout,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    save_report(report, args.output)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
