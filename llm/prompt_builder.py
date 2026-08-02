from __future__ import annotations

from retrieval.search import RetrievalResult


def build_generation_prompt(question: str, contexts: list[RetrievalResult]) -> str:
    """降级模式（非 agentic）下，基于本地检索结果构造的 RAG 提示。"""
    blocks: list[str] = []
    for i, item in enumerate(contexts, start=1):
        c = item.chunk
        page_label = ""
        if c.page_start is not None and c.page_end is not None:
            page_label = (
                f" p.{c.page_start}" if c.page_start == c.page_end else f" pp.{c.page_start}-{c.page_end}"
            )
        blocks.append(
            f"[S{i}] 《{c.paper_title}》· {c.section}{page_label} [{c.element_type}] "
            f"(modality={c.modality}, paper_id={c.paper_id})\n"
            f"Context: {c.chunk_context or c.section}\n{c.content}"
        )
    context_text = "\n\n".join(blocks) if blocks else "(无可用上下文)"

    return (
        "请基于以下论文检索结果回答用户问题。每条片段前有 [S编号] 标识，"
        "在关键结论后紧跟对应的 [S编号] 标注引用（如 [S1]，可连写 [S1][S2]）。\n"
        "只能引用下面出现过的编号，若上下文不足以回答请明确说明，不要编造。\n\n"
        f"用户问题:\n{question}\n\n"
        f"检索到的论文片段:\n{context_text}\n"
    )
