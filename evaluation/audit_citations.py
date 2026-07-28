"""Export E2E claims and citations for heuristic plus manual auditing."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.analyzer import QueryAnalyzer


_CITATION = re.compile(r"\[(S\d+)\]")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？.!?])\s*")
AUDIT_FIELDS = [
    "case_id",
    "claim",
    "citation",
    "paper_id",
    "snippet",
    "token_overlap",
    "heuristic",
    "manual_label",
    "manual_note",
]


def build_audit(payload: dict[str, Any], min_overlap: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create audit rows and diagnostics from an E2E benchmark payload."""

    analyzer = QueryAnalyzer()
    rows: list[dict[str, Any]] = []
    answered_records = 0
    cited_answer_records = 0
    cited_claims = 0
    uncited_claims = 0
    for record in payload.get("records", []):
        if record.get("status") != "answered":
            continue
        answered_records += 1
        answer = str(record.get("answer") or "").strip()
        if not answer:
            continue
        by_id = {source.get("id"): source for source in record.get("sources", [])}
        record_has_citation = False
        for sentence in _SENTENCE_BOUNDARY.split(answer):
            claim = _CITATION.sub("", sentence).strip()
            if not claim:
                continue
            citations = _CITATION.findall(sentence)
            if not citations:
                uncited_claims += 1
                rows.append(
                    {
                        "case_id": record.get("case_id"),
                        "claim": claim,
                        "citation": "",
                        "paper_id": "",
                        "snippet": "",
                        "token_overlap": "",
                        "heuristic": "missing_citation",
                        "manual_label": "",
                        "manual_note": "",
                    }
                )
                continue
            cited_claims += 1
            record_has_citation = True
            for citation in citations:
                source = by_id.get(citation)
                snippet = str((source or {}).get("snippet") or "")
                overlap = analyzer.overlap(claim, snippet) if source else 0.0
                if source is None:
                    heuristic = "missing_source"
                elif overlap >= min_overlap:
                    heuristic = "possibly_supported"
                else:
                    heuristic = "needs_review"
                rows.append(
                    {
                        "case_id": record.get("case_id"),
                        "claim": claim,
                        "citation": citation,
                        "paper_id": (source or {}).get("paper_id", ""),
                        "snippet": snippet,
                        "token_overlap": round(overlap, 4),
                        "heuristic": heuristic,
                        "manual_label": "",
                        "manual_note": "",
                    }
                )
        cited_answer_records += int(record_has_citation)

    counts = Counter(str(row["heuristic"]) for row in rows)
    citation_pairs = sum(bool(row["citation"]) for row in rows)
    summary = {
        "answered_records": answered_records,
        "cited_answer_records": cited_answer_records,
        "cited_claims": cited_claims,
        "uncited_claims": uncited_claims,
        "citation_pairs": citation_pairs,
        "heuristic_distribution": dict(sorted(counts.items())),
        "audit_ready": answered_records > 0,
        "manual_labels_complete": False,
        "warnings": [],
    }
    if not answered_records:
        summary["warnings"].append("E2E 结果没有 answered 响应，没有可审计回答")
    elif not citation_pairs:
        summary["warnings"].append("answered 响应没有有效 [S编号] 引用")
    summary["warnings"].append("token overlap 只用于筛选；幻觉率必须以人工标签计算")
    return rows, summary


def save_audit(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output: Path,
    summary_output: Path,
) -> None:
    """Write a CSV with stable headers and a machine-readable summary."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def audit_file(
    input_path: Path,
    output: Path,
    *,
    min_overlap: float = 0.12,
    summary_output: Path | None = None,
) -> dict[str, Any]:
    """Audit one E2E result file and return its summary."""

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows, summary = build_audit(payload, min_overlap)
    destination = summary_output or output.with_suffix(".summary.json")
    save_audit(rows, summary, output, destination)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="引用支持度审计")
    parser.add_argument("input", type=Path, help="E2E benchmark JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/citation_audit.csv"),
    )
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--min-overlap", type=float, default=0.12)
    args = parser.parse_args()
    summary = audit_file(
        args.input,
        args.output,
        min_overlap=args.min_overlap,
        summary_output=args.summary_output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
