from __future__ import annotations

from typing import TypedDict

from retrieval.retriever import RetrievalResult


class AgentState(TypedDict):
    question: str
    repo_name: str
    needs_multi_hop: bool
    max_hops: int
    current_hop: int
    steps: list[str]
    contexts: list[RetrievalResult]
    final_answer: str
