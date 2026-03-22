from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import BASE_DIR, load_settings
from retrieval.retriever import CodeRetriever


def recall_at_k(relevant: str, retrieved: list[str], k: int) -> float:
    topk = retrieved[:k]
    return 1.0 if any(relevant in item for item in topk) else 0.0


def mrr(relevant: str, retrieved: list[str]) -> float:
    for i, item in enumerate(retrieved, start=1):
        if relevant in item:
            return 1.0 / i
    return 0.0


def main() -> None:
    settings = load_settings()
    repo_name = settings.project.default_repo
    index_dir = BASE_DIR / settings.index.index_root / repo_name

    retriever = CodeRetriever(settings=settings, index_dir=str(index_dir))

    dataset_path = BASE_DIR / "evaluation" / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    recall_scores: list[float] = []
    mrr_scores: list[float] = []
    hit_scores: list[float] = []

    for sample in dataset:
        question = sample["question"]
        expected = sample["expected_file_keyword"]

        results = retriever.search(question)
        files = [r.chunk.file_path for r in results]

        r = recall_at_k(expected, files, k=settings.index.top_n_rerank)
        rr = mrr(expected, files)
        hit = 1.0 if any(expected in f for f in files) else 0.0

        recall_scores.append(r)
        mrr_scores.append(rr)
        hit_scores.append(hit)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    avg_hit = sum(hit_scores) / len(hit_scores) if hit_scores else 0.0

    print("===== Evaluation Result =====")
    print(f"Recall@{settings.index.top_n_rerank}: {avg_recall:.4f}")
    print(f"MRR: {avg_mrr:.4f}")
    print(f"Hit Rate: {avg_hit:.4f}")


if __name__ == "__main__":
    main()
