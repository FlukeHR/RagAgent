"""QASPER 检索侧评估（离线，不打任何 API）。

QASPER（Question Answering over Scientific Papers, Allen AI）是面向 NLP 论文的单文档问答数据集，
每个问题标注了支撑答案的 evidence 段落。本脚本把它当成**检索基准**：对每个可答、且有文本 evidence
的问题，在其所属论文的段落集合内检索，用 gold evidence 段落算 Hit@k / MRR / nDCG@k / Recall@k。

检索链路与生产一致：稠密向量 + BM25 → RRF 融合 → 可选 CrossEncoder 重排（复用 retrieval/ 下的组件）。

用法：
    1. 从 https://allenai.org/data/qasper 下载（建议 dev 集），把 `qasper-dev-v0.3.json`
       放到  evaluation/data/qasper/  下。
    2. python3 evaluation/eval_qasper.py [json路径] [--limit N] [--no-rerank]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from config.settings import load_settings
from retrieval.chunker import Chunk
from retrieval.embedder import Embedder
from retrieval.reranker import Reranker
from retrieval.retriever import Retriever

DEFAULT_DATA = PROJECT_ROOT / "evaluation" / "data" / "qasper" / "qasper-dev-v0.3.json"


def _paragraphs(paper: dict) -> list[Chunk]:
    """把一篇论文的 full_text 段落摊平为 Chunk（chunk_id = 段落序号）。"""
    chunks: list[Chunk] = []
    pid = paper.get("title", "paper")[:60]
    for section in paper.get("full_text", []):
        name = section.get("section_name") or "Body"
        for para in section.get("paragraphs", []):
            if para and para.strip():
                chunks.append(
                    Chunk(
                        chunk_id=str(len(chunks)), paper_id=pid, paper_title=pid,
                        section=name, content=para, source=pid,
                    )
                )
    return chunks


def _gold_evidence(qa: dict) -> set[str]:
    """汇总所有标注者的文本 evidence，剔除图表（FLOAT SELECTED）标记。"""
    gold: set[str] = set()
    for ans in qa.get("answers", []):
        a = ans.get("answer", {})
        if a.get("unanswerable"):
            continue
        for ev in a.get("evidence", []) or []:
            if ev and not ev.startswith("FLOAT SELECTED"):
                gold.add(ev.strip())
    return gold


def _sigmoid(x: float) -> float:
    x = max(-30.0, min(30.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def _retrieve(query: str, chunks, vecs, bm25, embedder, reranker, settings) -> list[tuple[str, float]]:
    """返回重排后的 (chunk_id, score) 列表（与生产 Retriever.search 同构）。"""
    top_k = settings.index.top_k_recall
    qv = embedder.encode([query])[0]
    dense_scores = vecs @ qv
    dense_idx = np.argsort(-dense_scores)[:top_k]
    dense = [(chunks[int(i)], float(dense_scores[int(i)])) for i in dense_idx]

    if bm25 is not None:
        bs = bm25.get_scores(query.lower().split())
        sparse_idx = np.argsort(-bs)[:top_k]
        sparse = [(chunks[int(i)], float(bs[int(i)])) for i in sparse_idx]
        cands = Retriever._rrf_fuse([dense, sparse])[:top_k]
    else:
        cands = dense

    reranked = reranker.rerank(query, cands, top_n=settings.index.top_n_rerank)
    return [(c.chunk_id, float(sc)) for c, sc in reranked]


def _sweep_report(confs: list[float], labels: list[int]) -> None:
    """阈值扫描：把检索是否命中(Hit@k)当标签，扫强阈值 t，找最能区分
    「该自信 vs 该二次检索」的点（最大 F1，附 Youden's J）。用于标定 low_confidence_threshold。"""
    import numpy as _np

    hits = [c for c, y in zip(confs, labels) if y]
    miss = [c for c, y in zip(confs, labels) if not y]
    n_pos, n_neg = len(hits), len(miss)
    print("\n===== low_confidence_threshold 阈值扫描 =====")
    print(f"样本：命中(应自信) {n_pos} 条，未命中(应二次检索) {n_neg} 条")
    if hits:
        print(f"命中相关概率：均值 {sum(hits)/n_pos:.3f}  最小 {min(hits):.3f}")
    if miss:
        print(f"未命中相关概率：均值 {sum(miss)/n_neg:.3f}  最大 {max(miss):.3f}")
    if not n_pos or not n_neg:
        print("（命中/未命中样本不全，无法标定阈值）")
        return

    print(f"\n{'t':>5} {'precision':>10} {'recall':>8} {'F1':>7} {'YoudenJ':>8}")
    best = (None, -1.0)  # (t, F1)
    best_j = (None, -1.0)
    for t in _np.linspace(0.0, 1.0, 21):
        tp = sum(1 for c, y in zip(confs, labels) if c >= t and y)
        fp = sum(1 for c, y in zip(confs, labels) if c >= t and not y)
        fn = sum(1 for c, y in zip(confs, labels) if c < t and y)
        tn = n_neg - fp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        tpr = tp / n_pos
        tnr = tn / n_neg
        j = tpr + tnr - 1
        print(f"{t:5.2f} {prec:10.3f} {rec:8.3f} {f1:7.3f} {j:8.3f}")
        if f1 > best[1]:
            best = (t, f1)
        if j > best_j[1]:
            best_j = (t, j)
    print(f"\n推荐 low_confidence_threshold ≈ {best[0]:.2f}（最大 F1={best[1]:.3f}）"
          f"；或 {best_j[0]:.2f}（最大 Youden's J={best_j[1]:.3f}）")


def _metrics(ranked: list[str], gold: set[str], k: int) -> tuple[float, float, float, float]:
    """Hit@k / MRR / nDCG@k / Recall@k（二值相关）。"""
    topk = ranked[:k]
    rels = [1 if cid in gold else 0 for cid in topk]
    hit = 1.0 if any(rels) else 0.0
    mrr = 0.0
    for i, cid in enumerate(ranked, start=1):
        if cid in gold:
            mrr = 1.0 / i
            break
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rels, start=1))
    ideal = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1)) or 1.0
    ndcg = dcg / idcg
    recall = (sum(rels) / len(gold)) if gold else 0.0
    return hit, mrr, ndcg, recall


def main() -> None:
    parser = argparse.ArgumentParser(description="QASPER 检索侧评估")
    parser.add_argument("data", nargs="?", default=str(DEFAULT_DATA), help="QASPER JSON 路径")
    parser.add_argument("--limit", type=int, default=50, help="评估论文数上限（默认 50，加速）")
    parser.add_argument("--no-rerank", action="store_true", help="禁用 CrossEncoder 重排")
    parser.add_argument("--sweep", action="store_true",
                        help="额外扫描 low_confidence_threshold（以 Hit@k 为标签找最优强阈值）")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[QASPER] 未找到数据文件：{data_path}")
        print("请从 https://allenai.org/data/qasper 下载（建议 dev 集），")
        print(f"把 qasper-dev-v0.3.json 放到：{DEFAULT_DATA.parent}/")
        sys.exit(1)

    settings = load_settings()
    embedder = Embedder(
        settings.embedding.model_name, settings.embedding.use_sentence_transformers
    )
    use_ce = settings.rerank.use_cross_encoder and not args.no_rerank
    reranker = Reranker(settings.rerank.model_name, use_cross_encoder=use_ce)
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        BM25Okapi = None

    dataset = json.loads(data_path.read_text(encoding="utf-8"))
    k = settings.index.top_n_rerank

    hits, mrrs, ndcgs, recalls = [], [], [], []
    sweep_confs, sweep_labels = [], []  # (top 相关概率, 是否 Hit@k)，供阈值扫描
    n_papers = n_questions = 0

    for paper in list(dataset.values())[: args.limit]:
        chunks = _paragraphs(paper)
        if len(chunks) < 2:
            continue
        para_index = {c.content.strip(): c.chunk_id for c in chunks}
        vecs = embedder.encode([c.content for c in chunks])
        bm25 = BM25Okapi([c.content.lower().split() for c in chunks]) if BM25Okapi else None
        n_papers += 1

        for qa in paper.get("qas", []):
            gold_text = _gold_evidence(qa)
            gold_ids = {para_index[t] for t in gold_text if t in para_index}
            if not gold_ids:  # 跳过 unanswerable / 仅图表 / evidence 非逐字段落
                continue
            pairs = _retrieve(
                qa["question"], chunks, vecs, bm25, embedder, reranker, settings
            )
            ranked = [cid for cid, _ in pairs]
            hit, mrr, ndcg, recall = _metrics(ranked, gold_ids, k)
            hits.append(hit); mrrs.append(mrr); ndcgs.append(ndcg); recalls.append(recall)
            if args.sweep:
                sweep_confs.append(_sigmoid(pairs[0][1]) if pairs else 0.0)
                sweep_labels.append(int(hit))
            n_questions += 1

    if not n_questions:
        print("[QASPER] 没有可评估的问题（可能数据为空或 evidence 均为图表）。")
        sys.exit(1)

    print("===== QASPER 检索评估 =====")
    print(f"papers: {n_papers}  questions: {n_questions}  "
          f"rerank(CE): {use_ce}  bm25: {bm25 is not None}")
    print(f"Hit@{k}:    {sum(hits) / n_questions:.4f}")
    print(f"MRR:       {sum(mrrs) / n_questions:.4f}")
    print(f"nDCG@{k}:   {sum(ndcgs) / n_questions:.4f}")
    print(f"Recall@{k}: {sum(recalls) / n_questions:.4f}")

    if args.sweep:
        _sweep_report(sweep_confs, sweep_labels)

if __name__ == "__main__":
    main()
