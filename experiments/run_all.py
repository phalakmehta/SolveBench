
import os
import json
import time
import sys
import argparse
import glob
from pathlib import Path
from datetime import datetime, timezone

# Adjust path so we can import internal modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from architectures import run_architecture
from data.bbh_loader import load_bbh_problems
from evaluation.ground_truth_accuracy import evaluate_accuracy
from evaluation.llm_judge import score_reasoning
from evaluation.automated_metrics import compute_all_metrics
from langchain_openai import ChatOpenAI


# ── Configuration ─────────────────────────────────────────────────────────────

ARCHITECTURES = ["A1_solo", "A2_pipeline", "A3_debate", "A4_reflection"]
MODELS = ["open-mixtral-8x7b"]

CHECKPOINT_DIR = Path("experiments/checkpoints")
RESULTS_DIR    = Path("results")
RESULTS_FILE   = RESULTS_DIR / "all_results.json"

# Mistral allows 1 RPS, so 1.0 second is safe
API_CALL_DELAY_SECONDS = 1.0


# ── LLM Setup ────────────────────────────────────────────────────────────────

def create_llm(api_key: str, temperature: float = 0.7):
    from langchain_mistralai import ChatMistralAI
    return ChatMistralAI(
        model="open-mixtral-8x7b",
        mistral_api_key=api_key,
        temperature=temperature,
        max_tokens=2048,
    )

def create_judge_llm(api_key: str):
    from langchain_mistralai import ChatMistralAI
    return ChatMistralAI(
        model="open-mixtral-8x7b",
        mistral_api_key=api_key,
        temperature=0.1,
        max_tokens=1024,
    )


# ── Checkpointing ────────────────────────────────────────────────────────────

def get_checkpoint_path(arch: str, problem_id: str) -> Path:
    return CHECKPOINT_DIR / f"{arch}__{problem_id}.json"


def is_checkpointed(arch: str, problem_id: str) -> bool:
    return get_checkpoint_path(arch, problem_id).exists()


def save_checkpoint(result: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = get_checkpoint_path(result["architecture"], result["problem_id"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def load_all_checkpoints() -> list[dict]:
    if not CHECKPOINT_DIR.exists():
        return []

    results = []
    for path in sorted(CHECKPOINT_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            results.append(json.load(f))

    return results


# ── Run ID ────────────────────────────────────────────────────────────────────

def make_run_id(arch: str, problem_id: str) -> str:
    return f"{arch}__{MODELS[0]}__{problem_id}"


# ── Single Run ────────────────────────────────────────────────────────────────

def run_single(
    arch      : str,
    problem   : dict,
    llm,
    judge_llm,
) -> dict:
    run_id = make_run_id(arch, problem["id"])
    print(f"\n[run_all] -> {run_id}")

    # ── Step 1: Run the architecture ──────────────────────────────────────
    # Implement explicit retry logic to handle rate limit (429) errors gracefully.
    max_retries = 10
    state = None
    for attempt in range(max_retries):
        try:
            state = run_architecture(
                arch_name=arch,
                problem=problem["input"],
                problem_id=problem["id"],
                subset=problem["subset"],
                llm=llm,
                ground_truth=problem["target"]
            )
            break
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "Quota exceeded" in error_str:
                print(f"[run_all] Rate limit hit on {arch} / {problem['id']}. Waiting 60s before retry {attempt+1}/{max_retries}...")
                time.sleep(60)
            else:
                raise  # re-raise if it's not a rate limit error

    if state is None:
        print(f"[run_all] [FAIL] Architecture run failed after {max_retries} retries.")
        state = {
            "solution": "RATE_LIMIT_ERROR",
            "extracted_answer": "RATE_LIMIT_ERROR",
            "intermediate": [],
        }

    # ── Step 2: Ground truth accuracy (Layer 1) ──────────────────────────
    is_correct = evaluate_accuracy(
        state.get("extracted_answer", ""),
        problem["target"],
    )
    print(f"[run_all]    Answer: '{state.get('extracted_answer', '')[:50]}' "
          f"| Ground truth: '{problem['target']}' | Correct: {is_correct}")

    # ── Step 3: LLM Judge reasoning quality (Layer 2) — only if correct ──
    reasoning_scores = {
        "reasoning_depth": 0,
        "step_completeness": 0,
        "self_consistency": 0,
        "brief_justification": "SKIPPED — incorrect answer",
        "parse_success": False,
    }

    if is_correct:
        try:
            judge_state = None
            for attempt in range(max_retries):
                try:
                    judge_state = score_reasoning(
                        problem=problem["input"],
                        solution=state["solution"],
                        judge_llm=judge_llm
                    )
                    break
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "Quota exceeded" in error_str:
                        print(f"[run_all] Rate limit hit on judge. Waiting 60s before retry {attempt+1}/{max_retries}...")
                        time.sleep(60)
                    else:
                        raise

            if judge_state is None:
                raise Exception("Judge failed after all retries due to rate limit.")

            reasoning_scores = judge_state
            time.sleep(API_CALL_DELAY_SECONDS)
        except Exception as e:
            print(f"[run_all] [FAIL] LLM judge failed: {e}")
            reasoning_scores["brief_justification"] = f"ERROR: {e}"

    # ── Step 4: Automated metrics (Layer 3) ───────────────────────────────
    auto_metrics = compute_all_metrics(state)

    # ── Step 5: Package result ────────────────────────────────────────────
    result = {
        "run_id"           : run_id,
        "architecture"     : arch,
        "model"            : MODELS[0],
        "problem_id"       : problem["id"],
        "subset"           : problem["subset"],
        "subset_display"   : problem.get("subset_display", ""),
        "task_type"        : problem.get("task_type", ""),
        "problem"          : problem["input"],
        "ground_truth"     : problem["target"],
        "solution"         : state["solution"],
        "extracted_answer"  : state.get("extracted_answer", ""),
        "is_correct"       : is_correct,
        "reasoning_scores" : reasoning_scores,
        "metrics"          : auto_metrics,
        "intermediate"     : state.get("intermediate", []),
        "timestamp"        : datetime.now(timezone.utc).isoformat(),
    }

    return result


# ── Main Runner ───────────────────────────────────────────────────────────────

def run_experiment(
    archs    : list[str],
    problems : list[dict],
    llm,
    judge_llm,
    resume   : bool = False,
) -> list[dict]:
    total_runs = len(archs) * len(problems)
    completed = 0

    if resume:
        existing = load_all_checkpoints()
        completed = len(existing)
        print(f"[run_all] Resuming: {completed} checkpoints found")
    else:
        existing = []

    print(f"\n[run_all] Experiment matrix: {len(archs)} archs × {len(problems)} problems = {total_runs} runs")
    print(f"[run_all] Already completed: {completed} | Remaining: {total_runs - completed}\n")

    all_results = list(existing)
    run_number = completed

    for arch in archs:
        for problem in problems:
            # Skip if already done (resume mode)
            if resume and is_checkpointed(arch, problem["id"]):
                continue

            run_number += 1
            print(f"[run_all] Run {run_number}/{total_runs}")

            result = run_single(arch, problem, llm, judge_llm)

            if result is not None:
                all_results.append(result)
                save_checkpoint(result)

    # Save all results to a single file
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n[run_all] Experiment complete. {len(all_results)} results saved.")
    print(f"[run_all] Results: {RESULTS_FILE}")
    print(f"[run_all] Checkpoints: {CHECKPOINT_DIR}/")

    return all_results


# ── Summary Report ────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    from evaluation.ground_truth_accuracy import compute_accuracy_by_architecture, compute_accuracy_by_subset

    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    # Overall accuracy by architecture
    acc_by_arch = compute_accuracy_by_architecture(results)
    print("\nAccuracy by Architecture:")
    for arch, acc in sorted(acc_by_arch.items()):
        correct = sum(1 for r in results if r["architecture"] == arch and r["is_correct"])
        total = sum(1 for r in results if r["architecture"] == arch)
        print(f"  {arch:20s}  {acc:.1%}  ({correct}/{total})")

    # Accuracy by subset
    acc_by_subset = compute_accuracy_by_subset(results)
    print("\nAccuracy by Subset:")
    for subset, arch_acc in sorted(acc_by_subset.items()):
        print(f"\n  {subset}:")
        for arch, acc in sorted(arch_acc.items()):
            print(f"    {arch:20s}  {acc:.1%}")

    print("\n" + "=" * 70)


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="SolveBench v2 Experiment Runner")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 1 problem per subset, A1 only")
    parser.add_argument("--arch",  type=str, default=None,
                        help="Run only this architecture (e.g. A3_debate)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoints, skipping completed runs")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Google API key (defaults to GOOGLE_API_KEY env var)")
    parser.add_argument("--n-per-subset", type=int, default=None,
                        help="Override number of problems per subset")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── API Key ───────────────────────────────────────────────────────────
    api_key = args.api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "MISTRAL_API_KEY not found. Set it as an environment variable or pass --api-key."
        )

    # ── LLM Setup ─────────────────────────────────────────────────────────
    llm = create_llm(api_key)
    judge_llm = create_judge_llm(api_key)

    # ── Problem Set ───────────────────────────────────────────────────────
    problems = load_bbh_problems(n_per_subset=args.n_per_subset)

    # ── Dry-run mode ──────────────────────────────────────────────────────
    if args.dry_run:
        print("[run_all] DRY RUN MODE — 1 problem per subset, A1 only")
        # Take first problem from each subset
        seen_subsets = set()
        dry_problems = []
        for p in problems:
            if p["subset"] not in seen_subsets:
                dry_problems.append(p)
                seen_subsets.add(p["subset"])

        results = run_experiment(
            archs     = ["A1_solo"],
            problems  = dry_problems,
            llm       = llm,
            judge_llm = judge_llm,
            resume    = False,
        )
        import shutil
        dashboard_data = Path("bbh_dashboard/public/data.json")
        if dashboard_data.parent.exists():
            shutil.copy2(RESULTS_FILE, dashboard_data)
            print(f"[run_all] Copied results to {dashboard_data} for the dashboard!")
        print_summary(results)
        return

    # ── Filter by CLI args ────────────────────────────────────────────────
    archs = [args.arch] if args.arch else ARCHITECTURES

    for a in archs:
        if a not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {a}. Choose from {ARCHITECTURES}")

    # ── Run ───────────────────────────────────────────────────────────────
    results = run_experiment(
        archs     = archs,
        problems  = problems,
        llm       = llm,
        judge_llm = judge_llm,
        resume    = args.resume,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
