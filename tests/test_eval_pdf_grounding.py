from __future__ import annotations

from evaluation.eval_pdf_grounding import (
    bbox_hit,
    bbox_iou,
    evaluate,
    ocr_hit,
    page_hit,
    page_recall,
    value_consistency,
    visual_semantic_hit,
)


def test_page_hit_and_recall_from_sources():
    sample = {
        "expected_pages": [2, 3],
        "sources": [{"page_start": 3, "page_end": 4}],
    }
    assert page_hit(sample) == 1.0
    assert page_recall(sample) == 0.5


def test_value_consistency_for_table_values_and_numbers():
    sample = {
        "expected_values": {"dataset": "QASPER", "accuracy": "91.2%"},
        "answer": "The QASPER setting reaches 91.2 accuracy.",
    }
    assert value_consistency(sample) == 1.0


def test_evaluate_aggregates_pdf_grounding_metrics():
    samples = [
        {
            "expected_pages": [1],
            "retrieved_pages": [1],
            "expected_values": ["42"],
            "answer": "The table reports 42.",
        },
        {
            "expected_pages": [5],
            "retrieved_pages": [2],
            "expected_values": ["needle"],
            "answer": "No relevant value.",
        },
    ]
    metrics = evaluate(samples)
    assert metrics["samples"] == 2.0
    assert metrics["page_hit"] == 0.5
    assert metrics["page_recall"] == 0.5
    assert metrics["value_consistency"] == 0.5
    assert metrics["bbox_hit"] == 0.0
    assert metrics["ocr_hit"] == 0.0
    assert metrics["visual_semantic_hit"] == 0.0


def test_bbox_ocr_and_visual_metrics():
    assert bbox_iou([0, 0, 10, 10], [5, 5, 15, 15]) > 0
    sample = {
        "expected_bboxes": [{"page": 2, "bbox": [10, 10, 50, 50]}],
        "expected_ocr_terms": ["scanned theorem"],
        "expected_visual_terms": ["rising curve"],
        "answer": "The scanned theorem is shown with a rising curve.",
        "sources": [
            {
                "page_start": 2,
                "page_end": 2,
                "bbox": [12, 12, 48, 48],
                "modality": "ocr",
                "element_type": "figure",
                "snippet": "scanned theorem rising curve",
            }
        ],
    }
    assert bbox_hit(sample) == 1.0
    assert ocr_hit(sample) == 1.0
    assert visual_semantic_hit(sample) == 1.0
