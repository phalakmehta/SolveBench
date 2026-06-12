
import csv
import json
import random
from pathlib import Path
from typing import Optional


DIMENSIONS = ["reasoning_depth", "step_completeness", "self_consistency"]
ACCEPTANCE_THRESHOLD = 0.70


# ── Step 1: Generate Calibration Sample ──────────────────────────────────────

def generate_calibration_sample(
    scored_results: list[dict],
    n: int = 20,
    seed: int = 42,
) -> list[dict]:
    eligible = [
        r for r in scored_results
        if r.get("is_correct", False)
        and r.get("reasoning_scores", {}).get("parse_success", False)
    ]

    if len(eligible) < n:
        print(f"[human_eval] Warning: only {len(eligible)} eligible results (need {n})")
        n = len(eligible)

    rng = random.Random(seed)
    sample = rng.sample(eligible, n)

    print(f"[human_eval] Selected {len(sample)} problems for calibration")
    return sample


# ── Step 2: Export Rating Template ────────────────────────────────────────────

def export_rating_template(
    sample: list[dict],
    output_path: str = "results/calibration_template.csv",
) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "problem_id", "architecture", "problem", "solution",
        "llm_reasoning_depth", "llm_step_completeness", "llm_self_consistency",
        "human_reasoning_depth", "human_step_completeness", "human_self_consistency",
    ]

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in sample:
            scores = r.get("reasoning_scores", {})
            writer.writerow({
                "problem_id":             r.get("problem_id", ""),
                "architecture":           r.get("architecture", ""),
                "problem":                r.get("problem", ""),
                "solution":               r.get("solution", ""),
                "llm_reasoning_depth":    scores.get("reasoning_depth", 0),
                "llm_step_completeness":  scores.get("step_completeness", 0),
                "llm_self_consistency":   scores.get("self_consistency", 0),
                "human_reasoning_depth":  "",  # human fills this
                "human_step_completeness":"",  # human fills this
                "human_self_consistency": "",  # human fills this
            })

    print(f"[human_eval] Rating template exported to {output}")
    return str(output)


# ── Step 3: Compute Judge Correlation ─────────────────────────────────────────

def compute_judge_correlation(
    llm_scores_file: str,
    human_scores_files: list[str],
) -> dict:
    try:
        from scipy.stats import pearsonr
    except ImportError:
        raise ImportError("scipy is required for correlation analysis. Install: pip install scipy")

    # Load LLM scores
    llm_data = _load_scores_csv(llm_scores_file, prefix="llm")

    # Load and average human scores across raters
    human_scores_per_rater = []
    for hf in human_scores_files:
        human_scores_per_rater.append(_load_scores_csv(hf, prefix="human"))

    results = {"per_dimension": {}, "dropped_dimensions": [], "n_raters": len(human_scores_files)}
    results["n_problems"] = len(llm_data)

    for dim in DIMENSIONS:
        llm_values = [row[dim] for row in llm_data]

        # Average human scores across raters
        human_values = []
        for i in range(len(llm_data)):
            rater_scores = [
                rater_data[i][dim]
                for rater_data in human_scores_per_rater
                if rater_data[i][dim] is not None
            ]
            if rater_scores:
                human_values.append(sum(rater_scores) / len(rater_scores))
            else:
                human_values.append(0)

        # Compute Pearson r
        if len(llm_values) >= 3 and len(human_values) >= 3:
            r, p_value = pearsonr(llm_values, human_values)
            r = round(r, 4)
        else:
            r = 0.0

        acceptable = r >= ACCEPTANCE_THRESHOLD
        results["per_dimension"][dim] = {"r": r, "acceptable": acceptable}

        if not acceptable:
            results["dropped_dimensions"].append(dim)

    results["overall_acceptable"] = len(results["dropped_dimensions"]) == 0
    return results


def _load_scores_csv(filepath: str, prefix: str) -> list[dict]:
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores = {}
            for dim in DIMENSIONS:
                key = f"{prefix}_{dim}"
                val = row.get(key, "")
                try:
                    scores[dim] = float(val) if val else None
                except ValueError:
                    scores[dim] = None
            rows.append(scores)
    return rows


# ── Smoke Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Create a dummy sample and export template
    dummy_sample = [
        {
            "problem_id": f"test_{i:03d}",
            "architecture": "A1_solo",
            "problem": f"Test problem {i}",
            "solution": f"Test solution {i}",
            "is_correct": True,
            "reasoning_scores": {
                "reasoning_depth": 4,
                "step_completeness": 3,
                "self_consistency": 5,
                "parse_success": True,
            },
        }
        for i in range(5)
    ]

    export_rating_template(dummy_sample, "results/test_calibration.csv")
    print("Template exported successfully.")
