from __future__ import annotations

import re
from dataclasses import dataclass, field

from config.settings import Settings
from retrieval.analyzer import QueryAnalyzer


@dataclass
class ConversationState:
    goal: str = ""
    constraints: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    cited_papers: list[str] = field(default_factory=list)
    summary: str = ""

    @classmethod
    def from_dict(cls, value: dict | None) -> "ConversationState":
        fields = cls.__dataclass_fields__
        data = {key: item for key, item in (value or {}).items() if key in fields}
        return cls(**data)


@dataclass(frozen=True)
class PreparedConversation:
    history: list[dict]
    summary: str
    dropped_messages: int


class ConversationManager:
    """Ambiguity, drift and bounded-history policy for multi-turn conversations."""

    vague_patterns = (
        r"^(这个|那个|它|这篇|上面|刚才).{0,8}$",
        r"^(帮我看看|分析一下|评价一下|比较一下)[。？?]?$",
        r"^(哪个|哪种).{0,4}(更好|最好)[。？?]?$",
    )
    continuation_terms = ("继续", "它", "这个", "那个", "上述", "刚才", "再说")
    switch_terms = ("换个话题", "另一个问题", "接下来问", "重新开始", "切换到")

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.analyzer = QueryAnalyzer(settings.retrieval.cjk_ngram_size)

    def prepare(self, history: list[dict]) -> PreparedConversation:
        valid = [
            {"role": item.get("role"), "content": str(item.get("content") or "")}
            for item in history
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]
        message_limit = self.settings.harness.history_max_messages
        recent_limit = min(
            message_limit,
            self.settings.harness.recent_history_messages,
        )
        recent = valid[-recent_limit:]
        older = valid[:-recent_limit]
        summary = self._summarize(older)
        budget = self.settings.harness.history_max_chars
        bounded: list[dict] = []
        used = len(summary)
        for item in reversed(recent):
            remaining = budget - used
            if remaining <= 0:
                break
            content = str(item.get("content") or "")[:remaining]
            bounded.append({"role": item["role"], "content": content})
            used += len(content)
        bounded.reverse()
        if summary:
            bounded.insert(
                0,
                {
                    "role": "assistant",
                    "content": f"[较早对话的系统摘要]\n{summary}",
                },
            )
        return PreparedConversation(
            history=bounded,
            summary=summary,
            dropped_messages=max(0, len(valid) - len(recent)),
        )

    def clarification_question(
        self,
        question: str,
        state: ConversationState,
    ) -> str | None:
        if not self.settings.harness.clarification_enabled:
            return None
        stripped = question.strip()
        vague = len(stripped) < self.settings.harness.clarification_min_chars or any(
            re.match(pattern, stripped) for pattern in self.vague_patterns
        )
        if not vague:
            return None
        if state.goal and any(term in stripped for term in self.continuation_terms):
            return None
        return (
            "这个问题还缺少足够信息。请补充研究对象或论文、希望比较的对象，"
            "以及评价标准（例如准确率、速度、成本或适用场景）；如有时间范围和期望输出格式也请说明。"
        )

    def drift_question(
        self,
        question: str,
        state: ConversationState,
    ) -> str | None:
        if not state.goal or any(term in question for term in self.switch_terms):
            return None
        if not any(term in question for term in self.continuation_terms):
            return None
        goal_tokens = set(self.analyzer.tokens(state.goal))
        question_tokens = set(self.analyzer.tokens(question))
        if goal_tokens and question_tokens and goal_tokens & question_tokens:
            return None
        return (
            f"你是想继续原目标“{state.goal[:120]}”，还是切换到一个新问题？"
            "请明确目标对象后我再检索，避免多轮对话偏移。"
        )

    def update_state(
        self,
        state: ConversationState,
        question: str,
        sources: list[dict],
        summary: str,
    ) -> ConversationState:
        if not state.goal or any(term in question for term in self.switch_terms):
            state.goal = question[:500]
        state.summary = summary
        for source in sources:
            paper_id = source.get("paper_id")
            if paper_id and paper_id not in state.cited_papers:
                state.cited_papers.append(paper_id)
        state.cited_papers = state.cited_papers[-50:]
        return state

    def _summarize(self, messages: list[dict]) -> str:
        if not messages:
            return ""
        lines = []
        for item in messages:
            role = "用户" if item["role"] == "user" else "助手"
            content = re.sub(r"\s+", " ", item["content"]).strip()
            lines.append(f"{role}: {content[:300]}")
        return "\n".join(lines)[-self.settings.harness.history_summary_max_chars :]
