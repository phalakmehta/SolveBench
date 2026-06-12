
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from architectures import AgentState, extract_final_answer

MAX_ITERATIONS = 3


# ── Prompt Templates ──────────────────────────────────────────────────────────

SOLVE_PROMPT = """You are an expert reasoning agent. Solve the following problem step by step.

PROBLEM:
{problem}

Show your complete reasoning process, then state your answer.

Format:
REASONING:
[Your step-by-step reasoning]

ANSWER: [Your answer]"""


CRITIQUE_PROMPT = """You are a rigorous Critic specialising in logical reasoning. Carefully 
evaluate the following solution to the problem below.

PROBLEM:
{problem}

CURRENT SOLUTION (Iteration {iteration}):
{solution}

Evaluate:
1. Check each logical step — are all inferences valid?
2. Are there any missing steps or unjustified assumptions?
3. Is the final answer consistent with the reasoning?
4. Rate the confidence level: HIGH / MEDIUM / LOW

Provide exactly 3 specific, actionable improvements the solver should make.
If the solution is already correct, confirm and explain why.

Format:
CONFIDENCE: [HIGH/MEDIUM/LOW]
ERRORS FOUND: [List specific errors, or "None"]
IMPROVEMENTS:
1. [First improvement]
2. [Second improvement]
3. [Third improvement]"""


REVISE_PROMPT = """You are an expert reasoning agent. You previously solved a problem, 
and a Critic has reviewed your solution. Revise and improve based on the feedback.

PROBLEM:
{problem}

YOUR PREVIOUS SOLUTION:
{previous_solution}

CRITIC'S FEEDBACK:
{critique}

Instructions:
- Address every point raised by the Critic
- Keep what was correct in the previous solution
- Fix any logical errors identified
- Show your complete revised reasoning

Format:
REVISED REASONING:
[Your corrected step-by-step reasoning]

FINAL ANSWER: [Your corrected answer]"""


# ── Helper ────────────────────────────────────────────────────────────────────

def _update_tokens(meta: dict, response) -> dict:
    meta = meta.copy()
    usage = response.usage_metadata or {}
    meta["prompt_tokens"]     += usage.get("input_tokens", 0)
    meta["completion_tokens"] += usage.get("output_tokens", 0)
    meta["total_tokens"]      += usage.get("total_tokens", 0)
    return meta


# ── Node Definitions ──────────────────────────────────────────────────────────

def make_solve(llm):
    def solve(state: AgentState) -> AgentState:
        prompt = SOLVE_PROMPT.format(problem=state["problem"])

        response = llm.invoke([HumanMessage(content=prompt)])
        solution = "".join([t.get("text", "") if isinstance(t, dict) else str(t) for t in response.content]) if isinstance(response.content, list) else str(response.content)
        meta = _update_tokens(state["metadata"], response)

        return {**state, "solution": solution, "iteration": 0, "metadata": meta}

    return solve


def make_critique(llm):
    def critique(state: AgentState) -> AgentState:
        prompt = CRITIQUE_PROMPT.format(
            problem=state["problem"],
            solution=state["solution"],
            iteration=state["iteration"],
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        critique_text = "".join([t.get("text", "") if isinstance(t, dict) else str(t) for t in response.content]) if isinstance(response.content, list) else str(response.content)
        meta = _update_tokens(state["metadata"], response)

        # Log the critique with its iteration number
        new_intermediate = state["intermediate"] + [
            {"role": f"critique_iter_{state['iteration']}", "content": critique_text}
        ]

        return {**state, "intermediate": new_intermediate, "metadata": meta}

    return critique


def make_revise(llm):
    def revise(state: AgentState) -> AgentState:
        # Get the most recent critique
        latest_critique = next(
            (
                item["content"]
                for item in reversed(state["intermediate"])
                if item["role"].startswith("critique_iter_")
            ),
            "No critique available."
        )

        prompt = REVISE_PROMPT.format(
            problem=state["problem"],
            previous_solution=state["solution"],
            critique=latest_critique,
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        revised_solution = "".join([t.get("text", "") if isinstance(t, dict) else str(t) for t in response.content]) if isinstance(response.content, list) else str(response.content)
        meta = _update_tokens(state["metadata"], response)
        new_iteration = state["iteration"] + 1

        # Log revised solution snapshot
        new_intermediate = state["intermediate"] + [
            {"role": f"solution_iter_{new_iteration}", "content": revised_solution}
        ]

        return {
            **state,
            "solution"    : revised_solution,
            "iteration"   : new_iteration,
            "intermediate": new_intermediate,
            "metadata"    : meta,
        }

    return revise


def format_output(state: AgentState) -> AgentState:
    extracted = extract_final_answer(state["solution"])

    header = (
        f"[A4 — Reflection Agent | {state['metadata']['model']} | "
        f"Iterations: {state['iteration']}]\n{'─' * 60}\n"
    )
    return {**state, "solution": header + state["solution"], "extracted_answer": extracted}


# ── Conditional Edge ──────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    if state["iteration"] < MAX_ITERATIONS:
        return "revise"
    return "format_output"


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph(llm, max_iterations: int = MAX_ITERATIONS) -> StateGraph:
    # Allow overriding MAX_ITERATIONS for ablation studies
    global MAX_ITERATIONS
    MAX_ITERATIONS = max_iterations

    graph = StateGraph(AgentState)

    graph.add_node("solve",         make_solve(llm))
    graph.add_node("critique",      make_critique(llm))
    graph.add_node("revise",        make_revise(llm))
    graph.add_node("format_output", format_output)

    graph.set_entry_point("solve")
    graph.add_edge("solve", "critique")

    # Conditional edge: from critique, check iteration count
    graph.add_conditional_edges(
        "critique",
        should_continue,
        {
            "revise"       : "revise",
            "format_output": "format_output",
        }
    )

    graph.add_edge("revise",        "critique")   # loop back
    graph.add_edge("format_output", END)

    return graph.compile()
