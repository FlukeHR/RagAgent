from __future__ import annotations

from retrieval.retriever import RetrievalResult


def build_generation_prompt(question: str, contexts: list[RetrievalResult]) -> str:
    """降级模式（非 agentic）下，基于本地检索结果构造的 RAG 提示。"""
    blocks: list[str] = []
    for i, item in enumerate(contexts, start=1):
        c = item.chunk
        blocks.append(
            f"[S{i}] 《{c.paper_title}》· {c.section} (paper_id={c.paper_id})\n{c.content}"
        )
    context_text = "\n\n".join(blocks) if blocks else "(无可用上下文)"

    return (
        "请基于以下论文检索结果回答用户问题。每条片段前有 [S编号] 标识，"
        "在关键结论后紧跟对应的 [S编号] 标注引用（如 [S1]，可连写 [S1][S2]）。\n"
        "只能引用下面出现过的编号，若上下文不足以回答请明确说明，不要编造。\n\n"
        f"用户问题:\n{question}\n\n"
        f"检索到的论文片段:\n{context_text}\n"
    )
