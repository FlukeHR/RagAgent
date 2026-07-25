"""Export claim/citation pairs for heuristic and manual support auditing."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.analyzer import QueryAnalyzer


_CITATION = re.compile(r"\[(S\d+)\]")


def main() -> None:
    parser = argparse.ArgumentParser(description="引用支持度审计")
    parser.add_argument("input", type=Path, help="E2E benchmark JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/citation_audit.csv"),
    )
    parser.add_argument("--min-overlap", type=float, default=0.12)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    analyzer = QueryAnalyzer()
    rows: list[dict] = []
    for record in payload.get("records", []):
        by_id = {source.get("id"): source for source in record.get("sources", [])}
        for sentence in re.split(r"(?<=[。！？.!?])\s*", record.get("answer") or ""):
            citations = _CITATION.findall(sentence)
            claim = _CITATION.sub("", sentence).strip()
            for citation in citations:
                source = by_id.get(citation, {})
                overlap = analyzer.overlap(claim, source.get("snippet") or "")
                rows.append(
                    {
                        "case_id": record.get("case_id"),
                        "claim": claim,
                        "citation": citation,
                        "paper_id": source.get("paper_id"),
                        "snippet": source.get("snippet"),
                        "token_overlap": round(overlap, 4),
                        "heuristic": (
                            "possibly_supported"
                            if overlap >= args.min_overlap
                            else "needs_review"
                        ),
                        "manual_label": "",
                        "manual_note": "",
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"exported {len(rows)} claim/citation pairs to {args.output}")


if __name__ == "__main__":
    main()
