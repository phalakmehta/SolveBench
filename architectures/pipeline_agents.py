
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from architectures import AgentState, extract_final_answer


# ── Prompt Templates ──────────────────────────────────────────────────────────

ANALYSE_PROMPT = """You are an expert analyst. Your task is to break down the following 
reasoning problem into its logical structure.

PROBLEM:
{problem}

Provide a structured analysis covering:
1. What type of reasoning is required (deductive, causal, spatial, etc.)
2. The key facts and constraints given
3. The logical steps needed to reach a solution
4. Any potential pitfalls or traps in the problem

Be concise and precise. Your analysis will be used by a Solver agent to construct the answer."""


SOLVE_PROMPT = """You are an expert problem-solver. Using the problem analysis below, 
work through each logical step to reach the correct answer.

PROBLEM:
{problem}

PROBLEM ANALYSIS (from Analyst):
{analysis}

Work through each step identified in the analysis. Show your reasoning clearly.
After your reasoning, state your conclusion.

Format:
STEP-BY-STEP SOLUTION:
[Your detailed reasoning here]

CONCLUSION: [Your answer here]"""


CRITIQUE_PROMPT = """You are a rigorous Critic specialising in logical reasoning. Your job is to 
find errors in the draft solution below and produce a corrected answer.

PROBLEM:
{problem}

DRAFT SOLUTION:
{draft_solution}

Your task:
1. Check each logical step for errors, missing steps, or invalid inferences.
2. Identify any assumptions that are not justified by the problem statement.
3. If you find errors, provide the corrected reasoning and answer.
4. If the solution is correct, confirm it and state the answer.

Output format:
--- ERROR CHECK ---
[List any errors found, or "No errors found"]

--- CORRECTED REASONING ---
[Full corrected reasoning if errors were found, or confirmation if correct]

FINAL ANSWER: [The correct answer]"""


# ── Helper ────────────────────────────────────────────────────────────────────

def _update_tokens(meta: dict, response) -> dict:
    meta = meta.copy()
    usage = response.usage_metadata or {}
    meta["prompt_tokens"]     += usage.get("input_tokens", 0)
    meta["completion_tokens"] += usage.get("output_tokens", 0)
    meta["total_tokens"]      += usage.get("total_tokens", 0)
    return meta


# ── Node Definitions ──────────────────────────────────────────────────────────

def make_analyse(llm):
    def analyse(state: AgentState) -> AgentState:
        prompt = ANALYSE_PROMPT.format(problem=state["problem"])

        response = llm.invoke([HumanMessage(content=prompt)])
        analysis_text = "".join([t.get("text", "") if isinstance(t, dict) else str(t) for t in response.content]) if isinstance(response.content, list) else str(response.content)

        meta = _update_tokens(state["metadata"], response)

        new_intermediate = state["intermediate"] + [
            {"role": "analysis", "content": analysis_text}
        ]

        return {**state, "intermediate": new_intermediate, "metadata": meta}

    return analyse


def make_solve(llm):
    def solve(state: AgentState) -> AgentState:
        analysis = next(
            (item["content"] for item in state["intermediate"] if item["role"] == "analysis"),
            "No analysis available."
        )

        prompt = SOLVE_PROMPT.format(
            problem=state["problem"],
            analysis=analysis,
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        draft = "".join([t.get("text", "") if isinstance(t, dict) else str(t) for t in response.content]) if isinstance(response.content, list) else str(response.content)

        meta = _update_tokens(state["metadata"], response)

        new_intermediate = state["intermediate"] + [
            {"role": "draft_solution", "content": draft}
        ]

        return {**state, "intermediate": new_intermediate, "metadata": meta}

    return solve


def make_critique(llm):
    def critique(state: AgentState) -> AgentState:
        draft = next(
            (item["content"] for item in state["intermediate"] if item["role"] == "draft_solution"),
            "No draft available."
        )

        prompt = CRITIQUE_PROMPT.format(
            problem=state["problem"],
            draft_solution=draft,
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        critique_and_solution = "".join([t.get("text", "") if isinstance(t, dict) else str(t) for t in response.content]) if isinstance(response.content, list) else str(response.content)

        meta = _update_tokens(state["metadata"], response)

        new_intermediate = state["intermediate"] + [
            {"role": "critique", "content": critique_and_solution}
        ]

        return {
            **state,
            "intermediate": new_intermediate,
            "solution": critique_and_solution,
            "metadata": meta,
        }

    return critique


def format_output(state: AgentState) -> AgentState:
    extracted = extract_final_answer(state["solution"])

    header = f"[A2 — Pipeline Agents | {state['metadata']['model']}]\n{'─' * 60}\n"
    return {**state, "solution": header + state["solution"], "extracted_answer": extracted}


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph(llm) -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("analyse",       make_analyse(llm))
    graph.add_node("solve",         make_solve(llm))
    graph.add_node("critique",      make_critique(llm))
    graph.add_node("format_output", format_output)

    graph.set_entry_point("analyse")
    graph.add_edge("analyse",       "solve")
    graph.add_edge("solve",         "critique")
    graph.add_edge("critique",      "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()
