from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolResult:
    """工具执行结果：text 回传给 LLM，sources 用于最终答案的引用溯源。"""

    text: str
    sources: list[dict] = field(default_factory=list)
