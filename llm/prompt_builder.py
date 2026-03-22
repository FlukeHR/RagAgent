from __future__ import annotations

from retrieval.retriever import RetrievalResult


def build_generation_prompt(question: str, contexts: list[RetrievalResult], steps: list[str]) -> str:
    context_blocks: list[str] = []
    for i, item in enumerate(contexts, start=1):
        chunk = item.chunk
        context_blocks.append(
            f"[Context {i}] {chunk.file_path}:{chunk.start_line}-{chunk.end_line}\n{chunk.content}"
        )

    steps_text = "\n".join([f"- {s}" for s in steps]) if steps else "- 无"
    context_text = "\n\n".join(context_blocks) if context_blocks else "(无可用上下文)"

    return (
        "你是一个代码分析助手。请基于给定上下文，回答用户问题。\n"
        "要求：\n"
        "1. 回答要准确、结构化。\n"
        "2. 如果上下文不足，明确指出。\n"
        "3. 给出关键推理过程。\n\n"
        f"用户问题:\n{question}\n\n"
        f"已执行步骤:\n{steps_text}\n\n"
        f"代码上下文:\n{context_text}\n"
    )
