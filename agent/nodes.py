from __future__ import annotations

from agent.prompts import JUDGE_MIN_CONTEXTS, PLANNER_HINTS
from agent.state import AgentState
from llm.model import LLMClient
from llm.prompt_builder import build_generation_prompt
from tools.code_search_tool import CodeSearchTool


def planner_node(state: AgentState) -> AgentState:
    q = state["question"].lower()
    needs_multi = any(token in q for token in PLANNER_HINTS)
    state["needs_multi_hop"] = needs_multi
    state["max_hops"] = 2 if needs_multi else 1
    state["steps"].append(f"Planner: multi-hop={needs_multi}")
    return state


def retriever_node(state: AgentState, code_search_tool: CodeSearchTool) -> AgentState:
    query = state["question"]
    if state["needs_multi_hop"] and state["current_hop"] > 0:
        query = f"{query} implementation call chain"

    results = code_search_tool.run(query)
    state["contexts"].extend(results)
    state["steps"].append(f"Retriever: hop={state['current_hop'] + 1}, got={len(results)}")
    return state


def judge_node(state: AgentState) -> tuple[AgentState, bool]:
    enough = len(state["contexts"]) >= JUDGE_MIN_CONTEXTS
    state["steps"].append(f"Judge: enough={enough}")
    return state, enough


def generator_node(state: AgentState, llm_client: LLMClient) -> AgentState:
    prompt = build_generation_prompt(
        question=state["question"],
        contexts=state["contexts"],
        steps=state["steps"],
    )
    output = llm_client.generate(prompt)
    state["final_answer"] = output.text
    state["steps"].append("Generator: answer_ready")
    return state
