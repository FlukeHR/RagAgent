"""Offline QASPER retrieval evaluation; never calls a paid generation API."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings, load_settings
from retrieval.analyzer import QueryAnalyzer
from retrieval.chunker import Chunk
from retrieval.embedder import Embedder
from retrieval.pipeline import rank_in_memory
from retrieval.reranker import Reranker

DEFAULT_DATA = (
    PROJECT_ROOT / "evaluation" / "data" / "qasper" / "qasper-dev-v0.3.json"
)


def _paragraphs(paper: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    paper_id = paper.get("title", "paper")[:60]
    for section in paper.get("full_text", []):
        section_name = section.get("section_name") or "Body"
        for paragraph in section.get("paragraphs", []):
            if paragraph and paragraph.strip():
                chunks.append(
                    Chunk(
                        chunk_id=str(len(chunks)),
                        paper_id=paper_id,
                        paper_title=paper_id,
                        section=section_name,
                        content=paragraph,
                        source=paper_id,
                        parent_id=f"{paper_id}::{section_name}",
                    )
                )
    return chunks


def _gold_evidence(question: dict) -> set[str]:
    gold: set[str] = set()
    for annotation in question.get("answers", []):
        answer = annotation.get("answer", {})
        if answer.get("unanswerable"):
            continue
        for evidence in answer.get("evidence", []) or []:
            if evidence and not evidence.startswith("FLOAT SELECTED"):
                gold.add(evidence.strip())
    return gold


def _metrics(
    ranked: list[str],
    gold: set[str],
    k: int,
) -> tuple[float, float, float, float]:
    top = ranked[:k]
    relevance = [1 if chunk_id in gold else 0 for chunk_id in top]
    hit = float(any(relevance))
    mrr = next(
        (1.0 / rank for rank, chunk_id in enumerate(ranked, start=1) if chunk_id in gold),
        0.0,
    )
    dcg = sum(value / math.log2(index + 1) for index, value in enumerate(relevance, 1))
    ideal = min(len(gold), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal + 1)) or 1.0
    recall = sum(relevance) / len(gold) if gold else 0.0
    return hit, mrr, dcg / idcg, recall


def evaluate_qasper(
    dataset: dict,
    settings: Settings,
    *,
    limit: int | None,
    mode: str,
    use_reranker: bool,
    top_k: int | None = None,
    top_n: int | None = None,
) -> dict:
    analyzer = QueryAnalyzer(settings.retrieval.cjk_ngram_size)
    embedder = Embedder(
        settings.embedding.model_name,
        settings.embedding.use_sentence_transformers,
        settings.embedding.fallback_dimension,
    )
    reranker = Reranker(
        settings.rerank.model_name,
        settings.rerank.use_cross_encoder and use_reranker,
        analyzer=analyzer,
    )
    recall_k = top_k or settings.index.top_k_recall
    output_k = top_n or settings.index.top_n_rerank
    values: dict[str, list[float]] = {
        "hit": [],
        "mrr": [],
        "ndcg": [],
        "recall": [],
        "confidence": [],
    }
    paper_count = 0
    question_count = 0

    papers = list(dataset.values())
    if limit:
        papers = papers[:limit]
    for paper in papers:
        chunks = _paragraphs(paper)
        if len(chunks) < 2:
            continue
        paragraph_index = {chunk.content.strip(): chunk.chunk_id for chunk in chunks}
        vectors = embedder.encode([chunk.content for chunk in chunks])
        paper_count += 1
        for question in paper.get("qas", []):
            gold_text = _gold_evidence(question)
            gold_ids = {
                paragraph_index[text] for text in gold_text if text in paragraph_index
            }
            if not gold_ids:
                continue
            query = question["question"]
            query_vector = embedder.encode([query])[0]
            results = rank_in_memory(
                query,
                chunks,
                vectors,
                query_vector,
                analyzer=analyzer,
                reranker=reranker,
                top_k=recall_k,
                top_n=output_k,
                mode=mode,
                use_reranker=use_reranker,
                rrf_k=settings.index.rrf_k,
                max_chunks_per_parent=settings.index.max_chunks_per_parent,
            )
            ranked = [result.chunk.chunk_id for result in results]
            hit, mrr, ndcg, recall = _metrics(ranked, gold_ids, output_k)
            values["hit"].append(hit)
            values["mrr"].append(mrr)
            values["ndcg"].append(ndcg)
            values["recall"].append(recall)
            values["confidence"].append(results[0].confidence if results else 0.0)
            question_count += 1
    if not question_count:
        raise ValueError("没有可评估的问题")
    return {
        "papers": paper_count,
        "questions": question_count,
        "mode": mode,
        "reranker": use_reranker,
        "reranker_backend": reranker.backend,
        "embedding": embedder.signature,
        "top_k_recall": recall_k,
        "top_n_rerank": output_k,
        f"hit@{output_k}": sum(values["hit"]) / question_count,
        "mrr": sum(values["mrr"]) / question_count,
        f"ndcg@{output_k}": sum(values["ndcg"]) / question_count,
        f"recall@{output_k}": sum(values["recall"]) / question_count,
        "mean_top_confidence": sum(values["confidence"]) / question_count,
    }


def result_metadata(settings: Settings, data_path: Path) -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.TimeoutExpired):
        commit = "unknown"
        dirty = True
    settings_snapshot = asdict(settings)
    if settings_snapshot.get("llm", {}).get("openai_api_key"):
        settings_snapshot["llm"]["openai_api_key"] = "***redacted***"
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": commit or "unknown",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "working_tree_dirty": dirty,
        "dataset": str(data_path),
        "settings": settings_snapshot,
    }


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        flat = {
            key: json.dumps(value, ensure_ascii=False)
            if isinstance(value, (dict, list))
            else value
            for key, value in report.items()
        }
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat))
            writer.writeheader()
            writer.writerow(flat)
    else:
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="QASPER 离线检索评估")
    parser.add_argument("data", nargs="?", default=str(DEFAULT_DATA))
    parser.add_argument("--limit", type=int, default=50, help="论文数上限；0 表示全部")
    parser.add_argument(
        "--mode",
        choices=("dense", "sparse", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--sweep", action="store_true", help="保留兼容；输出平均 top confidence")
    parser.add_argument("--output", type=Path, default=None, help="保存 JSON 或 CSV")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[QASPER] 未找到数据文件：{data_path}")
        sys.exit(1)
    settings = load_settings()
    dataset = json.loads(data_path.read_text(encoding="utf-8"))
    try:
        metrics = evaluate_qasper(
            dataset,
            settings,
            limit=args.limit or None,
            mode=args.mode,
            use_reranker=not args.no_rerank,
            top_k=args.top_k,
            top_n=args.top_n,
        )
    except ValueError as exc:
        print(f"[QASPER] {exc}")
        sys.exit(1)
    report = {"metadata": result_metadata(settings, data_path), "metrics": metrics}
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output:
        save_report(report, args.output)
        print(f"[QASPER] saved: {args.output}")


if __name__ == "__main__":
    main()
