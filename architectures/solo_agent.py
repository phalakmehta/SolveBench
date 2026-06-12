
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from architectures import AgentState, extract_final_answer


# ── Prompt Templates ──────────────────────────────────────────────────────────

SOLVE_PROMPT = """You are an expert reasoning agent. Solve the following problem step by step.

PROBLEM:
{problem}

INSTRUCTIONS:
1. Think through the problem carefully, showing each step of your reasoning.
2. Consider all the information given before reaching a conclusion.
3. After your reasoning, state your final answer clearly.

Format your response as:
REASONING:
[Your step-by-step reasoning here]

FINAL ANSWER: [Your answer here]
"""


# ── Node Definitions ──────────────────────────────────────────────────────────

def make_solve(llm):
    def solve(state: AgentState) -> AgentState:
        prompt = SOLVE_PROMPT.format(problem=state["problem"])

        response = llm.invoke([HumanMessage(content=prompt)])

        solution = "".join([t.get("text", "") if isinstance(t, dict) else str(t) for t in response.content]) if isinstance(response.content, list) else str(response.content)
        if isinstance(solution, list):
            solution = "".join([t.get("text", "") if isinstance(t, dict) else str(t) for t in solution])

        # Accumulate token counts in metadata
        meta = state["metadata"].copy()
        usage = response.usage_metadata or {}
        meta["prompt_tokens"]     += usage.get("input_tokens", 0)
        meta["completion_tokens"] += usage.get("output_tokens", 0)
        meta["total_tokens"]      += usage.get("total_tokens", 0)

        return {**state, "solution": solution, "metadata": meta}

    return solve


def format_output(state: AgentState) -> AgentState:
    extracted = extract_final_answer(state["solution"])

    header = f"[A1 — Solo Agent | {state['metadata']['model']}]\n{'─' * 60}\n"
    final_solution = header + state["solution"]

    return {**state, "solution": final_solution, "extracted_answer": extracted}


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph(llm) -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("solve",         make_solve(llm))
    graph.add_node("format_output", format_output)

    # Define edges (linear flow)
    graph.set_entry_point("solve")
    graph.add_edge("solve",         "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()


# ── Quick Smoke Test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.7,
        max_output_tokens=1024,
    )

    graph = build_graph(llm=llm)

    import time
    result = graph.invoke({
        "problem"          : "If a person tells the truth, does the statement 'I always lie' become a paradox? Answer Yes or No.",
        "problem_id"       : "test_001",
        "subset"           : "web_of_lies",
        "ground_truth"     : "Yes",
        "solution"         : "",
        "extracted_answer"  : "",
        "intermediate"     : [],
        "metadata"         : {"arch_name": "A1_solo", "model": "gemini-1.5-flash",
                              "start_time": time.time(), "total_tokens": 0,
                              "prompt_tokens": 0, "completion_tokens": 0},
        "iteration"        : 0,
    })

    print(result["solution"])
    print(f"\nExtracted answer: {result['extracted_answer']}")
    print(f"Tokens: {result['metadata']['total_tokens']}")
