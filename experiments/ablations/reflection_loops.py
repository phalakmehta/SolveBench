
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

from architectures import AgentState, extract_final_answer, run_architecture
from architectures.reflection_agent import build_graph
from data.bbh_loader import load_bbh_problems
from evaluation.ground_truth_accuracy import evaluate_accuracy, compute_accuracy


RESULTS_DIR = Path("results")
API_CALL_DELAY = 4.5


def run_ablation(
    problems: list[dict],
    llm,
    max_iterations: int,
) -> list[dict]:
    graph = build_graph(llm, max_iterations=max_iterations)
    results = []

    for i, problem in enumerate(problems):
        print(f"[ablation] Reflection {max_iterations}L — Problem {i+1}/{len(problems)}: {problem['id']}")

        initial_state: AgentState = {
            "problem":          problem["input"],
            "problem_id":       problem["id"],
            "subset":           problem["subset"],
            "ground_truth":     problem["target"],
            "solution":         "",
            "extracted_answer": "",
            "intermediate":     [],
            "metadata":         {
                "arch_name": f"A4_reflection_{max_iterations}L",
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
                "max_iterations":   max_iterations,
                "actual_iterations": state.get("iteration", 0),
                "extracted_answer": state["extracted_answer"],
                "ground_truth":     problem["target"],
                "is_correct":       is_correct,
                "latency_seconds":  state["metadata"]["latency_seconds"],
                "total_tokens":     state["metadata"]["total_tokens"],
            })

            print(f"  → {'✓' if is_correct else '✗'} (iters: {state.get('iteration', 0)}, "
                  f"answer: {state['extracted_answer'][:30]})")

        except Exception as e:
            print(f"  → FAILED: {e}")
            results.append({
                "problem_id": problem["id"], "subset": problem["subset"],
                "max_iterations": max_iterations, "is_correct": False, "error": str(e),
            })

        time.sleep(API_CALL_DELAY)

    return results


def main():
    parser = argparse.ArgumentParser(description="Reflection Loops Ablation")
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

    print("\n=== Running Reflection with 2 Loops ===")
    results_2l = run_ablation(problems, llm, max_iterations=2)

    print("\n=== Running Reflection with 3 Loops ===")
    results_3l = run_ablation(problems, llm, max_iterations=3)

    # Summary
    acc_2l = compute_accuracy(results_2l)
    acc_3l = compute_accuracy(results_3l)

    # Per-subset breakdown
    print(f"\n{'='*50}")
    print(f"ABLATION: Reflection Loops")
    print(f"{'='*50}")
    print(f"2 Loops: {acc_2l:.1%} accuracy")
    print(f"3 Loops: {acc_3l:.1%} accuracy")
    print(f"Difference: {(acc_3l - acc_2l):.1%}")

    # Average token cost comparison
    avg_tokens_2l = sum(r.get("total_tokens", 0) for r in results_2l) / max(len(results_2l), 1)
    avg_tokens_3l = sum(r.get("total_tokens", 0) for r in results_3l) / max(len(results_3l), 1)
    print(f"\nAvg tokens (2L): {avg_tokens_2l:.0f}")
    print(f"Avg tokens (3L): {avg_tokens_3l:.0f}")
    print(f"Token increase: {((avg_tokens_3l / max(avg_tokens_2l, 1)) - 1):.0%}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "ablation": "reflection_loops",
        "results_2_loops": results_2l,
        "results_3_loops": results_3l,
        "summary": {
            "accuracy_2l": acc_2l, "accuracy_3l": acc_3l,
            "avg_tokens_2l": avg_tokens_2l, "avg_tokens_3l": avg_tokens_3l,
        },
    }
    with open(RESULTS_DIR / "ablation_reflection_loops.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR / 'ablation_reflection_loops.json'}")


if __name__ == "__main__":
    main()
