
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from architectures import AgentState, extract_final_answer
from architectures.debate_agents import (
    make_propose_a, make_propose_b,
    make_argue_a, make_argue_b,
    make_judge, format_output,
    _get_intermediate, _update_tokens,
)
from data.bbh_loader import load_bbh_problems
from evaluation.ground_truth_accuracy import evaluate_accuracy, compute_accuracy


RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("experiments/checkpoints/ablation_debate")

API_CALL_DELAY = 4.5


def build_multi_round_debate_graph(llm, n_rounds: int = 2):
    graph = StateGraph(AgentState)

    # Proposal nodes (always 1 round)
    graph.add_node("propose_a", make_propose_a(llm))
    graph.add_node("propose_b", make_propose_b(llm))

    graph.set_entry_point("propose_a")
    graph.add_edge("propose_a", "propose_b")

    prev_node = "propose_b"

    # Argument rounds
    for round_num in range(1, n_rounds + 1):
        # Create argue nodes for this round
        a_name = f"argue_a_r{round_num}"
        b_name = f"argue_b_r{round_num}"

        # For round 1, argue against proposals. For later rounds, argue against prev arguments
        graph.add_node(a_name, make_argue_a(llm))
        graph.add_node(b_name, make_argue_b(llm))

        graph.add_edge(prev_node, a_name)
        graph.add_edge(a_name, b_name)
        prev_node = b_name

    # Judge and format
    graph.add_node("judge", make_judge(llm))
    graph.add_node("format_output", format_output)

    graph.add_edge(prev_node, "judge")
    graph.add_edge("judge", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()


def run_ablation(
    problems: list[dict],
    llm,
    n_rounds: int,
) -> list[dict]:
    graph = build_multi_round_debate_graph(llm, n_rounds=n_rounds)
    results = []

    for i, problem in enumerate(problems):
        print(f"[ablation] Debate {n_rounds}R — Problem {i+1}/{len(problems)}: {problem['id']}")

        initial_state: AgentState = {
            "problem":          problem["input"],
            "problem_id":       problem["id"],
            "subset":           problem["subset"],
            "ground_truth":     problem["target"],
            "solution":         "",
            "extracted_answer": "",
            "intermediate":     [],
            "metadata":         {
                "arch_name": f"A3_debate_{n_rounds}R",
                "model": "gemini-1.5-flash",
                "start_time": time.time(),
                "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0,
            },
            "iteration": 0,
        }

        try:
            state = graph.invoke(initial_state)
            state["metadata"]["latency_seconds"] = round(
                time.time() - state["metadata"]["start_time"], 2
            )
            if not state.get("extracted_answer"):
                state["extracted_answer"] = extract_final_answer(state.get("solution", ""))

            is_correct = evaluate_accuracy(state["extracted_answer"], problem["target"])

            results.append({
                "problem_id":       problem["id"],
                "subset":           problem["subset"],
                "n_rounds":         n_rounds,
                "extracted_answer": state["extracted_answer"],
                "ground_truth":     problem["target"],
                "is_correct":       is_correct,
                "latency_seconds":  state["metadata"]["latency_seconds"],
                "total_tokens":     state["metadata"]["total_tokens"],
            })

            print(f"  → {'✓' if is_correct else '✗'} (answer: {state['extracted_answer'][:30]})")

        except Exception as e:
            print(f"  → FAILED: {e}")
            results.append({
                "problem_id": problem["id"], "subset": problem["subset"],
                "n_rounds": n_rounds, "is_correct": False, "error": str(e),
            })

        time.sleep(API_CALL_DELAY)

    return results


def main():
    parser = argparse.ArgumentParser(description="Debate Rounds Ablation")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--n-per-subset", type=int, default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY required")

    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", google_api_key=api_key,
        temperature=0.7, max_output_tokens=2048,
    )

    problems = load_bbh_problems(n_per_subset=args.n_per_subset)

    print("\n=== Running Debate with 2 Argument Rounds ===")
    results_2r = run_ablation(problems, llm, n_rounds=2)

    print("\n=== Running Debate with 3 Argument Rounds ===")
    results_3r = run_ablation(problems, llm, n_rounds=3)

    # Summary
    acc_2r = compute_accuracy(results_2r)
    acc_3r = compute_accuracy(results_3r)

    print(f"\n{'='*50}")
    print(f"ABLATION: Debate Rounds")
    print(f"{'='*50}")
    print(f"2 Rounds: {acc_2r:.1%} accuracy")
    print(f"3 Rounds: {acc_3r:.1%} accuracy")
    print(f"Difference: {(acc_3r - acc_2r):.1%}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "ablation": "debate_rounds",
        "results_2_rounds": results_2r,
        "results_3_rounds": results_3r,
        "summary": {"accuracy_2r": acc_2r, "accuracy_3r": acc_3r},
    }
    with open(RESULTS_DIR / "ablation_debate_rounds.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {RESULTS_DIR / 'ablation_debate_rounds.json'}")


if __name__ == "__main__":
    main()
