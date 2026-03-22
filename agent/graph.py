from __future__ import annotations

from dataclasses import dataclass

from agent.nodes import generator_node, judge_node, planner_node, retriever_node
from agent.state import AgentState
from config.settings import Settings
from llm.model import LLMClient
from tools.code_search_tool import CodeSearchTool


@dataclass
class AgentAnswer:
    answer: str
    steps: list[str]
    sources: list[dict[str, str | int | float]]


class CodeRAGAgent:
    def __init__(self, settings: Settings, repo_name: str) -> None:
        self.settings = settings
        self.repo_name = repo_name
        self.code_search_tool = CodeSearchTool(settings, repo_name)
        self.llm_client = LLMClient(settings.llm)

    def ask(self, question: str) -> AgentAnswer:
        state: AgentState = {
            "question": question,
            "repo_name": self.repo_name,
            "needs_multi_hop": False,
            "max_hops": 1,
            "current_hop": 0,
            "steps": [],
            "contexts": [],
            "final_answer": "",
        }

        state = planner_node(state)

        while state["current_hop"] < state["max_hops"]:
            state = retriever_node(state, self.code_search_tool)
            state, enough = judge_node(state)
            state["current_hop"] += 1
            if enough:
                break

        state = generator_node(state, self.llm_client)

        unique_sources: dict[str, dict[str, str | int | float]] = {}
        for item in state["contexts"]:
            key = f"{item.chunk.file_path}:{item.chunk.start_line}-{item.chunk.end_line}"
            unique_sources[key] = {
                "file_path": item.chunk.file_path,
                "start_line": item.chunk.start_line,
                "end_line": item.chunk.end_line,
                "score": round(item.score, 4),
            }

        return AgentAnswer(
            answer=state["final_answer"],
            steps=state["steps"],
            sources=list(unique_sources.values())[: self.settings.index.top_n_rerank],
        )
