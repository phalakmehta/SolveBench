
import time
from typing import Literal, Optional
from typing_extensions import TypedDict


# ── Shared State Schema ───────────────────────────────────────────────────────

class AgentState(TypedDict):
    problem          : str
    problem_id       : str
    subset           : str
    ground_truth     : str
    solution         : str
    extracted_answer : str
    intermediate     : list
    metadata         : dict
    iteration        : int


# ── Answer Extraction ─────────────────────────────────────────────────────────

def extract_final_answer(text: str) -> str:
    import re

    if isinstance(text, list):
        text_clean = "".join([t.get("text", "") if isinstance(t, dict) else str(t) for t in text]).strip()
    else:
        text_clean = str(text).strip()

    # Pattern 1: $\boxed{X}$
    match = re.search(r"\\boxed\{(.+?)\}", text_clean)
    if match:
        # Strip \text{} if present inside the box
        inner = match.group(1).strip()
        text_match = re.search(r"\\text\{(.+?)\}", inner)
        return text_match.group(1).strip() if text_match else inner

    # Pattern 2: "FINAL ANSWER: X" or "Final Answer: X"
    match = re.search(r"(?:FINAL\s*ANSWER|final\s*answer)\s*[:=]\s*(.+?)(?:\n|$)", text_clean, re.IGNORECASE)
    if match:
        return match.group(1).strip().strip(".")

    # Pattern 3: "the answer is (X)" or "the answer is X"
    match = re.search(r"the\s+answer\s+is\s*[:\s]*\(?([^)\n.]+)\)?", text_clean, re.IGNORECASE)
    if match:
        return match.group(1).strip().strip(".")

    # Pattern 4: "Therefore, X" or "So, X" at end
    match = re.search(r"(?:therefore|so|thus|hence)[,\s]+(.+?)\.?\s*$", text_clean, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        if len(candidate) < 200:  # sanity check — not a whole paragraph
            return candidate.strip(".")

    # Fallback: last non-empty line
    lines = [l.strip() for l in text_clean.split("\n") if l.strip()]
    if lines:
        return lines[-1].strip(".")

    return text_clean


# ── Unified Runner ────────────────────────────────────────────────────────────

ArchName = Literal["A1_solo", "A2_pipeline", "A3_debate", "A4_reflection"]


def run_architecture(
    arch_name   : ArchName,
    problem     : str,
    problem_id  : str,
    subset      : str,
    llm,
    ground_truth: str = "",
) -> AgentState:
    # ── Initial state ─────────────────────────────────────────────────────
    initial_state: AgentState = {
        "problem"          : problem,
        "problem_id"       : problem_id,
        "subset"           : subset,
        "ground_truth"     : ground_truth,
        "solution"         : "",
        "extracted_answer"  : "",
        "intermediate"     : [],
        "metadata"         : {
            "arch_name"       : arch_name,
            "model"           : getattr(llm, "model", "unknown"),
            "start_time"      : time.time(),
            "total_tokens"    : 0,
            "prompt_tokens"   : 0,
            "completion_tokens": 0,
        },
        "iteration"        : 0,
    }

    # ── Route to the right architecture ──────────────────────────────────
    if arch_name == "A1_solo":
        from architectures.solo_agent import build_graph
    elif arch_name == "A2_pipeline":
        from architectures.pipeline_agents import build_graph
    elif arch_name == "A3_debate":
        from architectures.debate_agents import build_graph
    elif arch_name == "A4_reflection":
        from architectures.reflection_agent import build_graph
    else:
        raise ValueError(
            f"Unknown architecture: '{arch_name}'. "
            f"Must be one of: A1_solo, A2_pipeline, A3_debate, A4_reflection"
        )

    # ── Build graph and run ───────────────────────────────────────────────
    graph = build_graph(llm=llm)

    final_state = graph.invoke(initial_state)

    # ── Stamp total latency ───────────────────────────────────────────────
    final_state["metadata"]["latency_seconds"] = round(
        time.time() - final_state["metadata"]["start_time"], 2
    )

    # ── Extract final answer if not already done by a node ────────────────
    if not final_state.get("extracted_answer"):
        final_state["extracted_answer"] = extract_final_answer(
            final_state.get("solution", "")
        )

    return final_state
