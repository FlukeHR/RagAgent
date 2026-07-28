"""Offline FinanceBench PDF parsing, chunk preservation and retrieval evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import load_settings
from evaluation.eval_qasper import result_metadata
from evaluation.pdf_benchmark import (
    EvidenceCase,
    find_pdf,
    find_structured_file,
    load_records,
    parse_list,
    run_retrieval_benchmark,
    save_json,
)
from retrieval.loader import PaperLoader
from retrieval.pdf_parse import provider_from_config


def load_financebench_cases(path: Path, page_offset: int = 1) -> list[EvidenceCase]:
    """Convert official FinanceBench rows to the common evidence schema."""

    cases: list[EvidenceCase] = []
    for index, row in enumerate(load_records(path), start=1):
        evidence_items = parse_list(row.get("evidence"))
        document_ids: list[str] = []
        pages: list[int] = []
        evidence_texts: list[str] = []
        page_texts: dict[int, str] = {}
        for item in evidence_items:
            if not isinstance(item, dict):
                continue
            document = str(
                item.get("evidence_doc_name") or row.get("doc_name") or ""
            ).strip()
            if document:
                document_ids.append(Path(document).stem)
            page_value = item.get("evidence_page_num")
            try:
                page = int(page_value) + page_offset if page_value is not None else None
            except (TypeError, ValueError):
                page = None
            if page is not None:
                pages.append(page)
                full_page = str(item.get("evidence_text_full_page") or "").strip()
                if full_page:
                    page_texts[page] = full_page
            text = str(item.get("evidence_text") or "").strip()
            if text:
                evidence_texts.append(text)
        fallback_document = str(row.get("doc_name") or "").strip()
        if fallback_document:
            document_ids.append(Path(fallback_document).stem)
        document_ids = list(dict.fromkeys(document_ids))
        cases.append(
            EvidenceCase(
                case_id=str(row.get("financebench_id") or f"financebench-{index}"),
                question=str(row.get("question") or "").strip(),
                document_ids=document_ids,
                gold_pages=sorted(set(pages)),
                gold_evidence=evidence_texts,
                answer=str(row.get("answer") or "").strip() or None,
                answerable=bool(evidence_texts),
                evidence_types=[str(row.get("question_type") or "financial")],
                gold_page_texts=page_texts,
            )
        )
    return [case for case in cases if case.question and case.document_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description="FinanceBench 离线 PDF/evidence 评测")
    parser.add_argument(
        "dataset_root",
        type=Path,
        nargs="?",
        default=Path("evaluation/data/financebench"),
    )
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--pdf-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chunk-sizes", default="500,900,1400")
    parser.add_argument("--overlap-ratios", default="0.10,0.15,0.20")
    parser.add_argument("--top-k", default="5,10,20")
    parser.add_argument("--mode", choices=("dense", "sparse", "hybrid"), default="hybrid")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--evidence-threshold", type=float, default=0.8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/financebench_pdf.json"),
    )
    args = parser.parse_args()

    questions = args.questions or find_structured_file(
        args.dataset_root,
        {"question", "evidence", "doc_name"},
    )
    pdf_dir = args.pdf_dir or args.dataset_root / "pdfs"
    cases = load_financebench_cases(questions)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("FinanceBench 没有可评测样本")

    settings = load_settings()
    provider = provider_from_config(
        settings.pdf_parse.provider,
        settings.pdf_parse.auto_ocr,
        settings.pdf_parse.timeout_seconds,
    )
    loader = PaperLoader(str(pdf_dir), pdf_provider=provider)
    documents = []
    for document_id in sorted({item for case in cases for item in case.document_ids}):
        path = find_pdf(pdf_dir, document_id)
        document = loader.load_file(path)
        if document is not None:
            document.paper_id = document_id
            documents.append(document)

    report = run_retrieval_benchmark(
        documents,
        cases,
        settings,
        chunk_sizes=[int(value) for value in args.chunk_sizes.split(",")],
        overlap_ratios=[float(value) for value in args.overlap_ratios.split(",")],
        top_ks=[int(value) for value in args.top_k.split(",")],
        mode=args.mode,
        use_reranker=not args.no_rerank,
        evidence_threshold=args.evidence_threshold,
        scope="global",
    )
    report["dataset"] = "FinanceBench"
    report["metadata"] = result_metadata(settings, questions)
    report["sample_count"] = len(cases)
    report["pdf_count"] = len(documents)
    save_json(report, args.output)
    print(f"saved FinanceBench report to {args.output}")


if __name__ == "__main__":
    main()
