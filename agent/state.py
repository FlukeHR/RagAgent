from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentAnswer:
    answer: str
    status: str = "answered"
    steps: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)  # 结构化可观测事件，见 graph._trace
