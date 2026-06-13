from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import BASE_DIR, load_settings
from retrieval.retriever import Retriever


def recall_at_k(relevant: str, retrieved: list[str], k: int) -> float:
    return 1.0 if any(relevant in item for item in retrieved[:k]) else 0.0


def mrr(relevant: str, retrieved: list[str]) -> float:
    for i, item in enumerate(retrieved, start=1):
        if relevant in item:
            return 1.0 / i
    return 0.0


def main() -> None:
    settings = load_settings()
    collection = (
        sys.argv[1] if len(sys.argv) > 1 else settings.project.default_collection
    )
    index_dir = BASE_DIR / settings.index.index_root / collection
    retriever = Retriever(settings=settings, index_dir=str(index_dir))

    dataset_path = BASE_DIR / "evaluation" / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    recall_scores: list[float] = []
    mrr_scores: list[float] = []
    hit_scores: list[float] = []

    for sample in dataset:
        question = sample["question"]
        expected = sample["expected_paper_keyword"]

        results = retriever.search(question)
        # 用 paper_id 与论文标题共同作为匹配依据
        retrieved = [f"{r.chunk.paper_id} {r.chunk.paper_title}" for r in results]

        recall_scores.append(recall_at_k(expected, retrieved, k=settings.index.top_n_rerank))
        mrr_scores.append(mrr(expected, retrieved))
        hit_scores.append(1.0 if any(expected in r for r in retrieved) else 0.0)

    n = len(dataset) or 1
    print("===== Evaluation Result =====")
    print(f"collection: {collection}  samples: {len(dataset)}")
    print(f"Recall@{settings.index.top_n_rerank}: {sum(recall_scores) / n:.4f}")
    print(f"MRR: {sum(mrr_scores) / n:.4f}")
    print(f"Hit Rate: {sum(hit_scores) / n:.4f}")


if __name__ == "__main__":
    main()
