
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from architectures import AgentState, extract_final_answer


# ── Prompt Templates ──────────────────────────────────────────────────────────

PROPOSE_PROMPT = """You are Agent {agent_id}, an expert in logical reasoning. Independently 
solve the following problem. Do NOT consider what any other agent might say — this is 
your independent solution.

PROBLEM:
{problem}

Show your complete reasoning step by step, then state your answer.

Format:
REASONING:
[Your step-by-step reasoning]

MY ANSWER: [Your answer]"""


ARGUE_PROMPT = """You are Agent {agent_id}. You have just read Agent {opponent_id}'s 
solution to a reasoning problem. Your task is to critically evaluate their reasoning 
and identify any logical errors.

PROBLEM:
{problem}

{opponent_id}'s SOLUTION:
{opponent_proposal}

Instructions:
1. Check each step of {opponent_id}'s reasoning for logical errors.
2. Identify any assumptions that are not supported by the problem.
3. If you find errors, explain what the correct reasoning should be.
4. State whether you agree or disagree with {opponent_id}'s answer, and why.

Be rigorous and specific. Focus on LOGIC, not style."""


JUDGE_PROMPT = """You are a neutral Judge evaluating two agents' solutions to a 
reasoning problem. You have read both solutions and their critiques of each other.

PROBLEM:
{problem}

AGENT A'S SOLUTION:
{proposal_a}

AGENT B'S SOLUTION:
{proposal_b}

AGENT A'S CRITIQUE OF B:
{argument_a}

AGENT B'S CRITIQUE OF A:
{argument_b}

Your task:
1. Identify which reasoning chain is more logically sound.
2. Consider the critiques — did either agent find genuine errors?
3. Work through the problem yourself, using the best reasoning from both agents.
4. Provide the correct answer.

Format:
EVALUATION:
[Your analysis of both solutions]

CORRECT REASONING:
[Your step-by-step reasoning]

FINAL ANSWER: [The correct answer]"""


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_intermediate(state: AgentState, role: str) -> str:
    return next(
        (item["content"] for item in state["intermediate"] if item["role"] == role),
        f"[{role} not yet available]"
    )


def _update_tokens(meta: dict, response) -> dict:
    meta = meta.copy()
    usage = response.usage_metadata or {}
    meta["prompt_tokens"]     += usage.get("input_tokens", 0)
    meta["completion_tokens"] += usage.get("output_tokens", 0)
    meta["total_tokens"]      += usage.get("total_tokens", 0)
    return meta


# ── Node Definitions ──────────────────────────────────────────────────────────

def make_propose_a(llm):
    def propose_a(state: AgentState) -> AgentState:
        prompt = PROPOSE_PROMPT.format(
            agent_id="A",
            problem=state["problem"],
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        proposal = "".join([t.get("text", "") if isinstance(t, dict) else str(t) for t in response.content]) if isinstance(response.content, list) else str(response.content)

        new_intermediate = state["intermediate"] + [
            {"role": "proposal_a", "content": proposal}
        ]
        meta = _update_tokens(state["metadata"], response)
        return {**state, "intermediate": new_intermediate, "metadata": meta}

    return propose_a


def make_propose_b(llm):
    def propose_b(state: AgentState) -> AgentState:
        prompt = PROPOSE_PROMPT.format(
            agent_id="B",
            problem=state["problem"],
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        proposal = "".join([t.get("text", "") if isinstance(t, dict) else str(t) for t in response.content]) if isinstance(response.content, list) else str(response.content)

        new_intermediate = state["intermediate"] + [
            {"role": "proposal_b", "content": proposal}
        ]
        meta = _update_tokens(state["metadata"], response)
        return {**state, "intermediate": new_intermediate, "metadata": meta}

    return propose_b


def make_argue_a(llm):
    def argue_a(state: AgentState) -> AgentState:
        proposal_b = _get_intermediate(state, "proposal_b")

        prompt = ARGUE_PROMPT.format(
            agent_id="A",
            opponent_id="B",
            problem=state["problem"],
            opponent_proposal=proposal_b,
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        argument = "".join([t.get("text", "") if isinstance(t, dict) else str(t) for t in response.content]) if isinstance(response.content, list) else str(response.content)

        new_intermediate = state["intermediate"] + [
            {"role": "argument_a", "content": argument}
        ]
        meta = _update_tokens(state["metadata"], response)
        return {**state, "intermediate": new_intermediate, "metadata": meta}

    return argue_a


def make_argue_b(llm):
    def argue_b(state: AgentState) -> AgentState:
        proposal_a = _get_intermediate(state, "proposal_a")

        prompt = ARGUE_PROMPT.format(
            agent_id="B",
            opponent_id="A",
            problem=state["problem"],
            opponent_proposal=proposal_a,
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        argument = "".join([t.get("text", "") if isinstance(t, dict) else str(t) for t in response.content]) if isinstance(response.content, list) else str(response.content)

        new_intermediate = state["intermediate"] + [
            {"role": "argument_b", "content": argument}
        ]
        meta = _update_tokens(state["metadata"], response)
        return {**state, "intermediate": new_intermediate, "metadata": meta}

    return argue_b


def make_judge(llm):
    def judge(state: AgentState) -> AgentState:
        prompt = JUDGE_PROMPT.format(
            problem=state["problem"],
            proposal_a=_get_intermediate(state, "proposal_a"),
            proposal_b=_get_intermediate(state, "proposal_b"),
            argument_a=_get_intermediate(state, "argument_a"),
            argument_b=_get_intermediate(state, "argument_b"),
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        synthesis = "".join([t.get("text", "") if isinstance(t, dict) else str(t) for t in response.content]) if isinstance(response.content, list) else str(response.content)

        new_intermediate = state["intermediate"] + [
            {"role": "synthesis", "content": synthesis}
        ]
        meta = _update_tokens(state["metadata"], response)
        return {
            **state,
            "intermediate": new_intermediate,
            "solution"    : synthesis,
            "metadata"    : meta,
        }

    return judge


def format_output(state: AgentState) -> AgentState:
    extracted = extract_final_answer(state["solution"])

    header = f"[A3 — Debate Agents | {state['metadata']['model']}]\n{'─' * 60}\n"
    return {**state, "solution": header + state["solution"], "extracted_answer": extracted}


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph(llm) -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("propose_a",     make_propose_a(llm))
    graph.add_node("propose_b",     make_propose_b(llm))
    graph.add_node("argue_a",       make_argue_a(llm))
    graph.add_node("argue_b",       make_argue_b(llm))
    graph.add_node("judge",         make_judge(llm))
    graph.add_node("format_output", format_output)

    graph.set_entry_point("propose_a")

    # Sequential flow to respect rate limits
    graph.add_edge("propose_a",     "propose_b")
    graph.add_edge("propose_b",     "argue_a")
    graph.add_edge("argue_a",       "argue_b")
    graph.add_edge("argue_b",       "judge")
    graph.add_edge("judge",         "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()
