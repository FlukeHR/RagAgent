"""生成侧评估（RAGAS）：faithfulness / answer relevancy / context precision。

**会调用真实 LLM API（计费）**，需显式授权后才运行。
judge LLM 复用项目的 OpenAI-compatible 配置；嵌入用本地模型避免嵌入 API。

依赖（可选，仅本脚本需要）：
    pip install ragas langchain-openai langchain-huggingface

跑法：
    RAG_EVAL_ALLOW_API=1 python3 evaluation/eval_generation.py --limit 20
    # --yes 可代替 RAG_EVAL_ALLOW_API=1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import load_settings, resolve_model_path
from evaluation.eval_qasper import result_metadata

DEFAULT_QASPER = PROJECT_ROOT / "evaluation" / "data" / "qasper" / "qasper-dev-v0.3.json"


def _patch_langchain_community() -> None:
    """兼容垫片：ragas 0.4.x 的 llms/base.py 仍硬 import
    `langchain_community.chat_models.vertexai.ChatVertexAI`，而 langchain-community 1.x
    已把 vertexai 移出该路径，导致 `import ragas` 直接失败。这里在导入 ragas 前注入一个
    占位模块（项目使用 OpenAI-compatible API，不使用 vertexai）。
    若该模块真实存在则不动。彻底方案是对齐 ragas 与 langchain 版本，此处优先保证可跑通。
    """
    import importlib
    import types

    name = "langchain_community.chat_models.vertexai"
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001 - 缺失则注入占位
        pass
    mod = types.ModuleType(name)
    setattr(
        mod,
        "ChatVertexAI",
        type("ChatVertexAI", (), {}),
    )  # 仅占位，不会被实际调用
    sys.modules[name] = mod


def authorized(yes_flag: bool) -> bool:
    """是否已显式授权调用真实 API。"""
    return bool(yes_flag) or os.getenv("RAG_EVAL_ALLOW_API") == "1"


def evaluation_api_key(settings) -> str:
    """Resolve the paid-evaluation credential without logging its value."""

    key = settings.llm.openai_api_key or os.getenv("OPENAI_API_KEY") or ""
    if not key.strip():
        raise ValueError(
            "缺少 OPENAI_API_KEY。RAG_EVAL_ALLOW_API=1 只表示允许付费调用，"
            "不是 API 凭证。请在当前终端设置 OPENAI_API_KEY；"
            "本地免认证接口可显式设置为 not-needed。"
        )
    return key


# ---------- 官方 QASPER 生成侧（单文档 QA：在该论文内检索→生成→以 gold answer 为 reference） ----------
def qasper_reference(qa: dict) -> str | None:
    """从 QASPER 标注里取一条参考答案（free_form / extractive / yes_no），无则返回 None。"""
    for ans in qa.get("answers", []):
        a = ans.get("answer", {})
        if a.get("unanswerable"):
            continue
        if a.get("free_form_answer"):
            return a["free_form_answer"].strip()
        if a.get("extractive_spans"):
            return " ".join(a["extractive_spans"]).strip()
        if a.get("yes_no") is not None:
            return "Yes" if a["yes_no"] else "No"
    return None


def _retrieve_results(query, chunks, vecs, embedder, reranker, settings, top_n):
    """在单篇论文内返回完整检索结果，保留分数、置信度和 backend。"""
    from retrieval.analyzer import QueryAnalyzer
    from retrieval.pipeline import rank_in_memory

    analyzer = QueryAnalyzer(settings.retrieval.cjk_ngram_size)
    return rank_in_memory(
        query,
        chunks,
        vecs,
        embedder.encode([query])[0],
        analyzer=analyzer,
        reranker=reranker,
        top_k=settings.index.top_k_recall,
        top_n=top_n,
        mode="hybrid",
        use_reranker=True,
        rrf_k=settings.index.rrf_k,
        max_chunks_per_parent=settings.index.max_chunks_per_parent,
    )


def _gen_prompt(question: str, contexts) -> str:
    from llm.prompt_builder import build_generation_prompt

    return build_generation_prompt(question, contexts)


def collect_samples_qasper(dataset: dict, settings, llm, limit: int | None = None,
                           top_n: int | None = None) -> list[dict]:
    """对每个 QASPER 问题在其所属论文内检索→生成答案，组装带 reference 的 ragas 样本。"""
    from evaluation.eval_qasper import _paragraphs
    from retrieval.embedder import Embedder
    from retrieval.reranker import Reranker

    from retrieval.analyzer import QueryAnalyzer

    analyzer = QueryAnalyzer(settings.retrieval.cjk_ngram_size)
    embedder = Embedder(
        settings.embedding.model_name,
        settings.embedding.use_sentence_transformers,
        settings.embedding.fallback_dimension,
    )
    reranker = Reranker(
        settings.rerank.model_name,
        settings.rerank.use_cross_encoder,
        analyzer=analyzer,
    )
    top_n = top_n or settings.index.top_n_rerank

    samples: list[dict] = []
    for paper in dataset.values():
        chunks = _paragraphs(paper)
        if len(chunks) < 2:
            continue
        vecs = embedder.encode([c.content for c in chunks])
        for qa in paper.get("qas", []):
            ref = qasper_reference(qa)
            if not ref:  # 跳过 unanswerable / 无可用参考答案
                continue
            top = _retrieve_results(
                qa["question"],
                chunks,
                vecs,
                embedder,
                reranker,
                settings,
                top_n,
            )
            response = llm.generate(_gen_prompt(qa["question"], top))
            samples.append({
                "user_input": qa["question"],
                "retrieved_contexts": [item.chunk.content for item in top]
                or ["(empty)"],
                "response": response,
                "reference": ref,
            })
            if limit and len(samples) >= limit:
                return samples
    return samples


# ---------- ragas 后端构造 ----------
def _build_judge(settings, api_key: str):
    """按项目 LLM 配置构造 ragas 的 judge LLM + 本地嵌入（避免嵌入 API）。"""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    cfg = settings.llm
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=cfg.model_name,
        base_url=cfg.openai_api_base or None,
        api_key=api_key,
    )

    from langchain_huggingface import HuggingFaceEmbeddings

    emb = HuggingFaceEmbeddings(model_name=resolve_model_path(settings.embedding.model_name))
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb)


def _metrics(with_reference: bool = False):
    """构造指标列表（跨 ragas 版本鲁棒）。

    始终含 faithfulness + answer relevancy；context precision 有参考用 WithReference 版；
    有参考时再尽量加 context recall / answer(factual) correctness（版本不一定有，缺则跳过）。
    """
    metrics: list = []
    try:  # ragas >= 0.2
        from ragas.metrics import Faithfulness, ResponseRelevancy

        metrics += [Faithfulness(), ResponseRelevancy()]
    except ImportError:  # 旧版本回退
        from ragas.metrics import answer_relevancy, faithfulness

        metrics += [faithfulness, answer_relevancy]

    import importlib

    mm = importlib.import_module("ragas.metrics")
    cp = "LLMContextPrecisionWithReference" if with_reference else "LLMContextPrecisionWithoutReference"
    for name in [cp, *(["LLMContextRecall", "AnswerCorrectness", "FactualCorrectness"] if with_reference else [])]:
        cls = getattr(mm, name, None)
        if cls is not None:
            try:
                metrics.append(cls())
            except Exception:  # noqa: BLE001 - 个别指标构造失败不阻断整体
                pass
    if not any("Precision" in type(m).__name__ or "precision" in getattr(m, "name", "") for m in metrics):
        cp_obj = getattr(mm, "context_precision", None)  # 极旧版本回退
        if cp_obj is not None:
            metrics.append(cp_obj)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="生成侧评估（RAGAS），会调用真实 LLM API")
    parser.add_argument(
        "qasper",
        nargs="?",
        default=str(DEFAULT_QASPER),
        help="QASPER JSON 路径",
    )
    parser.add_argument("--limit", type=int, default=None, help="评估问题数上限")
    parser.add_argument("--yes", action="store_true", help="确认调用真实 API（等价 RAG_EVAL_ALLOW_API=1）")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/generation_evaluation.json"),
    )
    args = parser.parse_args()

    if not authorized(args.yes):
        print("⚠️ 本评估会调用真实 LLM API（计费）。确认后用 --yes 或设 RAG_EVAL_ALLOW_API=1 再跑。")
        sys.exit(2)

    _patch_langchain_community()  # 必须在 import ragas 之前
    try:
        from ragas import EvaluationDataset, evaluate
    except ImportError:
        print("缺少 ragas，请先安装：pip install ragas langchain-openai langchain-huggingface")
        sys.exit(1)

    settings = load_settings()
    try:
        api_key = evaluation_api_key(settings)
    except ValueError as exc:
        print(f"[GenEval] {exc}")
        sys.exit(2)

    qpath = Path(args.qasper)
    if not qpath.exists():
        print(
            f"未找到 QASPER 数据：{qpath}\n"
            "请从 https://allenai.org/data/qasper 下载放到该路径。"
        )
        sys.exit(1)
    from llm.model import LLMClient

    print("[GenEval] QASPER 模式，收集样本（openai-compatible）…")
    llm = LLMClient(settings.llm)
    samples = collect_samples_qasper(
        json.loads(qpath.read_text(encoding="utf-8")),
        settings,
        llm,
        args.limit,
    )

    if not samples:
        print("没有可评估的样本（QASPER 全 unanswerable，或评估集为空）。")
        sys.exit(1)

    judge_llm, judge_emb = _build_judge(settings, api_key)
    dataset = EvaluationDataset.from_list(samples)
    print(f"[GenEval] 评分 {len(samples)} 条样本…")
    judge_usage = {}
    try:
        from langchain_core.callbacks import get_usage_metadata_callback

        with get_usage_metadata_callback() as callback:
            result = evaluate(
                dataset=dataset,
                metrics=_metrics(with_reference=True),
                llm=judge_llm,
                embeddings=judge_emb,
            )
        judge_usage = callback.usage_metadata
    except ImportError:
        result = evaluate(
            dataset=dataset,
            metrics=_metrics(with_reference=True),
            llm=judge_llm,
            embeddings=judge_emb,
        )
    print("\n===== RAGAS 生成侧评估 =====")
    print(result)
    payload = {
        "metadata": result_metadata(settings, qpath),
        "sample_count": len(samples),
        "metrics": str(result),
        "usage": llm.consume_usage_events(),
        "judge_usage": judge_usage,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[GenEval] saved: {args.output}")


if __name__ == "__main__":
    main()
