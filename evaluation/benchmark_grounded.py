"""Evaluate production chunks against locally curated paper/page/evidence gold labels."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import load_settings
from indexing.build_index import build_index
from retrieval.retriever import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="生产论文库 chunk/top-k 评测")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("evaluation/data/business_cases.jsonl"),
    )
    parser.add_argument("--chunk-sizes", default="500,900,1400")
    parser.add_argument("--overlap-ratios", default="0.10,0.15,0.20")
    parser.add_argument("--top-k", default="8,12,24")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/grounded_benchmark.json"),
    )
    args = parser.parse_args()
    cases = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases = [
        case
        for case in cases
        if case.get("enabled", True) and case.get("gold_paper_ids")
    ]
    if not cases:
        raise SystemExit("请先在 business_cases.jsonl 启用带 gold_paper_ids 的真实业务样本")

    base = load_settings()
    runs: list[dict] = []
    for chunk_size in [int(value) for value in args.chunk_sizes.split(",")]:
        for ratio in [float(value) for value in args.overlap_ratios.split(",")]:
            overlap = int(chunk_size * ratio)
            index_root = f"./evaluation/results/index-{chunk_size}-{overlap}"
            settings = replace(
                base,
                index=replace(
                    base.index,
                    index_root=index_root,
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                ),
            )
            build_index(settings, incremental=False)
            retriever = Retriever(settings, str(PROJECT_ROOT / index_root))
            for top_k in [int(value) for value in args.top_k.split(",")]:
                hits = 0
                reciprocal_ranks: list[float] = []
                for case in cases:
                    results = retriever.search(case["question"], top_k=top_k)
                    gold = set(case["gold_paper_ids"])
                    rank = next(
                        (
                            index
                            for index, result in enumerate(results, start=1)
                            if result.chunk.paper_id in gold
                        ),
                        None,
                    )
                    hits += int(rank is not None)
                    reciprocal_ranks.append(1.0 / rank if rank else 0.0)
                runs.append(
                    {
                        "chunk_size": chunk_size,
                        "overlap": overlap,
                        "top_k": top_k,
                        "samples": len(cases),
                        "paper_hit_rate": hits / len(cases),
                        "mrr": sum(reciprocal_ranks) / len(cases),
                    }
                )
    report = {"runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved {len(runs)} runs to {args.output}")


if __name__ == "__main__":
    main()
