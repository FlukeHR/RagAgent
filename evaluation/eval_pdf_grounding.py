"""PDF grounding evaluation helpers.

This lightweight eval complements QASPER text-evidence retrieval with checks that
matter for page-aware and multimodal PDF RAG:

- page hit: did retrieved sources land on any expected page?
- table/value consistency: did the answer preserve expected table/numeric values?
- bbox hit: did a source bbox overlap the expected region on the expected page?
- OCR hit: did OCR-modal evidence surface expected scanned-page text?
- visual semantic hit: did figure/image evidence or the answer include expected visual terms?

Input JSON is a list of samples:
[
  {
    "question": "...",
    "expected_pages": [3, 4],
    "sources": [{"page_start": 3, "page_end": 3}],
    "expected_values": {"accuracy": "91.2", "dataset": "QASPER"},
    "answer": "..."
  }
]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable


def source_pages(source: dict) -> set[int]:
    """Expand a source's page_start/page_end into a page set."""
    start = source.get("page_start")
    end = source.get("page_end")
    if start is None or end is None:
        return set()
    try:
        a, b = int(start), int(end)
    except (TypeError, ValueError):
        return set()
    if a > b:
        a, b = b, a
    return set(range(a, b + 1))


def retrieved_pages(sample: dict) -> set[int]:
    """Return retrieved pages from either explicit retrieved_pages or source metadata."""
    if "retrieved_pages" in sample:
        return {int(p) for p in sample.get("retrieved_pages") or []}
    pages: set[int] = set()
    for source in sample.get("sources") or []:
        pages.update(source_pages(source))
    return pages


def page_hit(sample: dict) -> float:
    """1.0 if any retrieved page overlaps expected_pages, else 0.0."""
    expected = {int(p) for p in sample.get("expected_pages") or []}
    if not expected:
        return 0.0
    return 1.0 if expected & retrieved_pages(sample) else 0.0


def page_recall(sample: dict) -> float:
    """Fraction of expected pages covered by retrieved pages."""
    expected = {int(p) for p in sample.get("expected_pages") or []}
    if not expected:
        return 0.0
    return len(expected & retrieved_pages(sample)) / len(expected)


_NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?")


def _normalize_value(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _numeric_forms(value: object) -> set[str]:
    text = str(value)
    forms: set[str] = set()
    for match in _NUMBER_RE.findall(text):
        raw = match.replace(",", "")
        forms.add(raw)
        if raw.endswith("%"):
            forms.add(raw[:-1])
    return forms


def value_present(answer: str, expected: object) -> bool:
    """Check whether one expected value is preserved in answer text."""
    answer_norm = _normalize_value(answer)
    expected_norm = _normalize_value(expected)
    if expected_norm and expected_norm in answer_norm:
        return True

    expected_numbers = _numeric_forms(expected)
    if not expected_numbers:
        return False
    answer_numbers = set().union(*(_numeric_forms(x) for x in _NUMBER_RE.findall(answer)))
    return bool(expected_numbers & answer_numbers)


def iter_expected_values(sample: dict) -> Iterable[object]:
    values = sample.get("expected_values") or []
    if isinstance(values, dict):
        return values.values()
    return values


def value_consistency(sample: dict) -> float:
    """Fraction of expected table/numeric values found in the answer."""
    values = list(iter_expected_values(sample))
    if not values:
        return 0.0
    answer = sample.get("answer") or sample.get("response") or ""
    hits = sum(1 for value in values if value_present(answer, value))
    return hits / len(values)


def _bbox(value) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(float(x) for x in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def bbox_iou(a, b) -> float:
    """Intersection-over-union for PDF-space bboxes."""
    box_a = _bbox(a)
    box_b = _bbox(b)
    if box_a is None or box_b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def bbox_hit(sample: dict, threshold: float = 0.5) -> float:
    expected = sample.get("expected_bboxes") or []
    if isinstance(expected, dict):
        expected = [expected]
    if not expected:
        return 0.0
    sources = sample.get("sources") or []
    hits = 0
    for item in expected:
        if not isinstance(item, dict):
            continue
        page = item.get("page") or item.get("page_start")
        box = item.get("bbox")
        matched = False
        for source in sources:
            if page is not None and int(page) not in source_pages(source):
                continue
            if bbox_iou(box, source.get("bbox")) >= threshold:
                matched = True
                break
        hits += int(matched)
    return hits / len(expected)


def _terms_present(terms: list[str], text: str) -> bool:
    haystack = _normalize_value(text)
    return all(_normalize_value(term) in haystack for term in terms if str(term).strip())


def ocr_hit(sample: dict) -> float:
    terms = [str(x) for x in sample.get("expected_ocr_terms") or []]
    if not terms:
        return 0.0
    evidence = " ".join(
        str(s.get("snippet") or "")
        for s in sample.get("sources") or []
        if s.get("modality") == "ocr" or s.get("element_type") == "ocr"
    )
    evidence += " " + str(sample.get("answer") or sample.get("response") or "")
    return 1.0 if _terms_present(terms, evidence) else 0.0


def visual_semantic_hit(sample: dict) -> float:
    terms = [str(x) for x in sample.get("expected_visual_terms") or []]
    if not terms:
        return 0.0
    evidence = " ".join(
        str(s.get("snippet") or "")
        for s in sample.get("sources") or []
        if s.get("element_type") in {"figure", "image", "page_image", "region_image"}
        or s.get("modality") == "image"
    )
    evidence += " " + str(sample.get("answer") or sample.get("response") or "")
    return 1.0 if _terms_present(terms, evidence) else 0.0


def evaluate(samples: list[dict]) -> dict[str, float]:
    if not samples:
        return {"samples": 0.0, "page_hit": 0.0, "page_recall": 0.0, "value_consistency": 0.0}
    return {
        "samples": float(len(samples)),
        "page_hit": sum(page_hit(s) for s in samples) / len(samples),
        "page_recall": sum(page_recall(s) for s in samples) / len(samples),
        "value_consistency": sum(value_consistency(s) for s in samples) / len(samples),
        "bbox_hit": sum(bbox_hit(s) for s in samples) / len(samples),
        "ocr_hit": sum(ocr_hit(s) for s in samples) / len(samples),
        "visual_semantic_hit": sum(visual_semantic_hit(s) for s in samples) / len(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PDF page grounding and table/value consistency.")
    parser.add_argument("data", help="JSON file with page/value grounding samples")
    parser.add_argument("--record", action="store_true", help="Append metrics to evaluation history")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"未找到评测集：{data_path}")
        sys.exit(1)

    samples = json.loads(data_path.read_text(encoding="utf-8"))
    metrics = evaluate(samples)
    print("===== PDF Grounding Eval =====")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    if args.record:
        from config.settings import load_settings
        from evaluation.results_log import record_run

        settings = load_settings()
        rec = record_run("pdf_grounding", data_path.name, int(metrics["samples"]), metrics, settings)
        print(f"\n[已记录] {rec['git']}@{rec['branch']} -> evaluation/results/history.jsonl")


if __name__ == "__main__":
    main()
