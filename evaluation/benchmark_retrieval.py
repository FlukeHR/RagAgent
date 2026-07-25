"""Run retrieval ablations and staged top-k/top-n sweeps, saving comparable reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import load_settings
from evaluation.eval_qasper import (
    DEFAULT_DATA,
    evaluate_qasper,
    result_metadata,
    save_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="QASPER 检索消融与参数 sweep")
    parser.add_argument("data", nargs="?", default=str(DEFAULT_DATA))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--top-k", default="8,12,24")
    parser.add_argument("--top-n", default="3,5,8")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/retrieval_benchmark.json"),
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    dataset = json.loads(data_path.read_text(encoding="utf-8"))
    settings = load_settings()
    runs: list[dict] = []
    ablations = [
        ("dense", False),
        ("sparse", False),
        ("hybrid", False),
        ("hybrid", True),
    ]
    for mode, use_reranker in ablations:
        runs.append(
            evaluate_qasper(
                dataset,
                settings,
                limit=args.limit or None,
                mode=mode,
                use_reranker=use_reranker,
            )
        )

    top_ks = [int(value) for value in args.top_k.split(",") if value]
    top_ns = [int(value) for value in args.top_n.split(",") if value]
    for top_k in top_ks:
        for top_n in top_ns:
            if top_n > top_k:
                continue
            runs.append(
                evaluate_qasper(
                    dataset,
                    settings,
                    limit=args.limit or None,
                    mode="hybrid",
                    use_reranker=True,
                    top_k=top_k,
                    top_n=top_n,
                )
            )
    report = {
        "metadata": result_metadata(settings, data_path),
        "runs": runs,
        "note": (
            "Chunk-size sweep requires the production-grounded dataset because "
            "QASPER uses annotated paragraphs as chunks."
        ),
    }
    save_report(report, args.output)
    print(f"saved {len(runs)} runs to {args.output}")


if __name__ == "__main__":
    main()
