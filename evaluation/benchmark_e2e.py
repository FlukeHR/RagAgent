"""Measure end-to-end latency, token usage and estimated cost against a running API."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[int((len(ordered) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser(description="端到端 API benchmark")
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

    import requests

    cases = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases = [case for case in cases if case.get("enabled", True)]
    records: list[dict] = []
    for case in cases:
        for repetition in range(args.repeat):
            payload = {
                "question": case["question"],
                "history": case.get("history", []),
                "session_id": f"bench-{case['id']}-{repetition}",
            }
            started = time.perf_counter()
            response = requests.post(
                args.url,
                json=payload,
                timeout=args.timeout,
                stream=True,
            )
            first_byte = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            raw = b"".join(response.iter_content(chunk_size=65536))
            elapsed = (time.perf_counter() - started) * 1000
            body = json.loads(raw)
            usage = next(
                (
                    event
                    for event in reversed(body.get("trace", []))
                    if event.get("type") == "usage"
                ),
                {},
            )
            records.append(
                {
                    "case_id": case["id"],
                    "category": case["category"],
                    "repetition": repetition,
                    "latency_ms": elapsed,
                    "first_byte_ms": first_byte,
                    "status": body.get("status"),
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
    latencies = [record["latency_ms"] for record in records]
    first_bytes = [record["first_byte_ms"] for record in records]
    report = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip(),
            "url": args.url,
            "repeat": args.repeat,
        },
        "summary": {
            "requests": len(records),
            "mean_ms": statistics.mean(latencies) if latencies else 0,
            "p50_ms": percentile(latencies, 0.50),
            "p95_ms": percentile(latencies, 0.95),
            "p99_ms": percentile(latencies, 0.99),
            "first_byte_p50_ms": percentile(first_bytes, 0.50),
            "first_byte_p95_ms": percentile(first_bytes, 0.95),
            "total_input_tokens": sum(record["input_tokens"] for record in records),
            "total_output_tokens": sum(record["output_tokens"] for record in records),
            "total_estimated_cost": sum(record["estimated_cost"] for record in records),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
