
import math
from typing import Optional


# ── Score Normalisation ───────────────────────────────────────────────────────

def normalize_score(score: float, min_val: float = 1.0, max_val: float = 5.0) -> float:
    if max_val == min_val:
        return 0.0
    return max(0.0, min(1.0, (score - min_val) / (max_val - min_val)))


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    return {dim: normalize_score(val) for dim, val in scores.items()}


# ── Score Aggregation ─────────────────────────────────────────────────────────

# v2 weights: 3 dimensions (equal weighting)
DIMENSION_WEIGHTS = {
    "reasoning_depth"   : 0.35,
    "step_completeness" : 0.35,
    "self_consistency"  : 0.30,
}

# All 3 dimension names
DIMENSIONS = ["reasoning_depth", "step_completeness", "self_consistency"]


def compute_composite_score(scores: dict[str, float], weights: Optional[dict] = None) -> float:
    w = weights or DIMENSION_WEIGHTS
    total_weight = 0.0
    weighted_sum = 0.0

    for dim, weight in w.items():
        if dim in scores and scores[dim] > 0:  # skip 0 (unscored/failed)
            weighted_sum += scores[dim] * weight
            total_weight += weight

    if total_weight == 0:
        return 0.0
    return round(weighted_sum / total_weight, 4)


# ── Architecture Statistics ───────────────────────────────────────────────────

def compute_architecture_stats(results: list[dict], score_key: str = "reasoning_scores") -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for result in results:
        arch = result["architecture"]
        grouped.setdefault(arch, [])
        grouped[arch].append(result)

    stats = {}

    for arch, arch_results in grouped.items():
        arch_stats = {}

        for dim in DIMENSIONS:
            values = [
                r.get(score_key, {}).get(dim, 0.0)
                for r in arch_results
                if r.get(score_key, {}).get(dim, 0) > 0  # skip unscored
            ]
            if values:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                std = math.sqrt(variance)
                arch_stats[dim] = {"mean": round(mean, 4), "std": round(std, 4), "n": len(values)}
            else:
                arch_stats[dim] = {"mean": 0.0, "std": 0.0, "n": 0}

        # Composite
        composite_scores = [
            compute_composite_score(r.get(score_key, {}))
            for r in arch_results
            if compute_composite_score(r.get(score_key, {})) > 0
        ]
        if composite_scores:
            comp_mean = sum(composite_scores) / len(composite_scores)
            comp_var = sum((v - comp_mean) ** 2 for v in composite_scores) / len(composite_scores)
            arch_stats["composite"] = {"mean": round(comp_mean, 4), "std": round(math.sqrt(comp_var), 4)}
        else:
            arch_stats["composite"] = {"mean": 0.0, "std": 0.0}

        stats[arch] = arch_stats

    return stats


# ── Win Rate Analysis ─────────────────────────────────────────────────────────

def compute_win_rates(results: list[dict], score_key: str = "reasoning_scores") -> dict[str, dict]:
    # Group by problem_id
    by_problem: dict[str, list[dict]] = {}
    for result in results:
        pid = result["problem_id"]
        by_problem.setdefault(pid, [])
        by_problem[pid].append(result)

    # Count wins
    win_counts: dict[str, dict[str, int]] = {}
    for result in results:
        arch = result["architecture"]
        win_counts.setdefault(arch, {dim: 0 for dim in DIMENSIONS})

    total_problems = len(by_problem)

    for pid, problem_results in by_problem.items():
        for dim in DIMENSIONS:
            scores_for_dim = {
                r["architecture"]: r.get(score_key, {}).get(dim, 0.0)
                for r in problem_results
            }
            if not scores_for_dim:
                continue
            max_score = max(scores_for_dim.values())
            if max_score == 0:
                continue
            for arch, score in scores_for_dim.items():
                if score == max_score:
                    win_counts[arch][dim] += 1

    # Convert to fractions
    win_rates = {}
    for arch, counts in win_counts.items():
        win_rates[arch] = {
            dim: round(count / total_problems, 4) if total_problems > 0 else 0.0
            for dim, count in counts.items()
        }

    return win_rates


# ── Subset Breakdown ──────────────────────────────────────────────────────────

def compute_scores_by_subset(results: list[dict], score_key: str = "reasoning_scores") -> dict[str, dict[str, float]]:
    bucket: dict[str, dict[str, list]] = {}

    for result in results:
        subset = result.get("subset", "unknown")
        arch = result["architecture"]
        composite = compute_composite_score(result.get(score_key, {}))
        if composite > 0:  # skip unscored
            bucket.setdefault(subset, {}).setdefault(arch, []).append(composite)

    output = {}
    for subset, arch_scores in bucket.items():
        output[subset] = {
            arch: round(sum(scores) / len(scores), 4)
            for arch, scores in arch_scores.items()
            if scores
        }

    return output


# ── Hypothesis Testing ────────────────────────────────────────────────────────

def test_hypothesis_h1(accuracy_by_subset: dict) -> dict:
    findings = {}
    for subset, arch_acc in accuracy_by_subset.items():
        a1 = arch_acc.get("A1_solo", 0)
        a3 = arch_acc.get("A3_debate", 0)
        a4 = arch_acc.get("A4_reflection", 0)
        findings[subset] = {
            "A1_solo": a1,
            "A3_debate": a3,
            "A4_reflection": a4,
            "a3_advantage": round(a3 - a1, 4),
            "a4_advantage": round(a4 - a1, 4),
        }
    return {"hypothesis": "H1: Complexity threshold", "findings": findings}


def test_hypothesis_h2(arch_stats: dict) -> dict:
    a1_sc = arch_stats.get("A1_solo", {}).get("step_completeness", {}).get("mean", 0)
    a2_sc = arch_stats.get("A2_pipeline", {}).get("step_completeness", {}).get("mean", 0)

    return {
        "hypothesis": "H2: Pipeline improves step completeness",
        "A1_solo_step_completeness": a1_sc,
        "A2_pipeline_step_completeness": a2_sc,
        "advantage": round(a2_sc - a1_sc, 4),
        "supported": a2_sc > a1_sc,
    }


def test_hypothesis_h3(accuracy_by_subset: dict) -> dict:
    ff = accuracy_by_subset.get("formal_fallacies", {})
    a1 = ff.get("A1_solo", 0)
    a3 = ff.get("A3_debate", 0)

    return {
        "hypothesis": "H3: Debate improves accuracy on formal fallacies",
        "A1_solo_accuracy": a1,
        "A3_debate_accuracy": a3,
        "advantage": round(a3 - a1, 4),
        "supported": a3 > a1,
    }


def test_hypothesis_h4(results: list[dict]) -> dict:
    a1_scores = [
        r["metrics"]["self_contradiction"]["score"]
        for r in results
        if r["architecture"] == "A1_solo" and "self_contradiction" in r.get("metrics", {})
    ]
    a4_scores = [
        r["metrics"]["self_contradiction"]["score"]
        for r in results
        if r["architecture"] == "A4_reflection" and "self_contradiction" in r.get("metrics", {})
    ]

    a1_mean = sum(a1_scores) / len(a1_scores) if a1_scores else 0
    a4_mean = sum(a4_scores) / len(a4_scores) if a4_scores else 0

    return {
        "hypothesis": "H4: Reflection reduces self-contradiction",
        "A1_solo_mean_contradiction": round(a1_mean, 4),
        "A4_reflection_mean_contradiction": round(a4_mean, 4),
        "reduction": round(a1_mean - a4_mean, 4),
        "supported": a4_mean < a1_mean,
    }


def test_hypothesis_h5(results: list[dict]) -> dict:
    arch_metrics: dict[str, dict] = {}
    for r in results:
        arch = r["architecture"]
        arch_metrics.setdefault(arch, {"latency": [], "tokens": []})
        arch_metrics[arch]["latency"].append(r.get("metrics", {}).get("latency_seconds", 0))
        arch_metrics[arch]["tokens"].append(r.get("metrics", {}).get("total_tokens", 0))

    findings = {}
    a1_lat = sum(arch_metrics.get("A1_solo", {}).get("latency", [0])) / max(len(arch_metrics.get("A1_solo", {}).get("latency", [1])), 1)
    a1_tok = sum(arch_metrics.get("A1_solo", {}).get("tokens", [0])) / max(len(arch_metrics.get("A1_solo", {}).get("tokens", [1])), 1)

    for arch in ["A1_solo", "A2_pipeline", "A3_debate", "A4_reflection"]:
        data = arch_metrics.get(arch, {"latency": [0], "tokens": [0]})
        mean_lat = sum(data["latency"]) / max(len(data["latency"]), 1)
        mean_tok = sum(data["tokens"]) / max(len(data["tokens"]), 1)
        findings[arch] = {
            "mean_latency": round(mean_lat, 2),
            "mean_tokens": round(mean_tok, 0),
            "latency_ratio_vs_a1": round(mean_lat / a1_lat, 2) if a1_lat > 0 else 0,
            "token_ratio_vs_a1": round(mean_tok / a1_tok, 2) if a1_tok > 0 else 0,
        }

    return {"hypothesis": "H5: Architecture gains come at cost", "findings": findings}


# ── Quick Smoke Test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_scores = {"reasoning_depth": 4.0, "step_completeness": 3.5, "self_consistency": 4.5}

    print("Composite score:", compute_composite_score(test_scores))
    print("Normalised:", normalize_scores(test_scores))

    dummy_results = [
        {"architecture": "A1_solo",      "problem_id": "test_001", "subset": "web_of_lies",
         "reasoning_scores": {"reasoning_depth": 3.5, "step_completeness": 3.0, "self_consistency": 4.0}},
        {"architecture": "A2_pipeline",  "problem_id": "test_001", "subset": "web_of_lies",
         "reasoning_scores": {"reasoning_depth": 4.0, "step_completeness": 4.5, "self_consistency": 4.0}},
        {"architecture": "A3_debate",    "problem_id": "test_001", "subset": "web_of_lies",
         "reasoning_scores": {"reasoning_depth": 3.5, "step_completeness": 3.5, "self_consistency": 5.0}},
        {"architecture": "A4_reflection","problem_id": "test_001", "subset": "web_of_lies",
         "reasoning_scores": {"reasoning_depth": 4.8, "step_completeness": 4.0, "self_consistency": 4.8}},
    ]

    print("\nArchitecture stats:", compute_architecture_stats(dummy_results))
    print("\nWin rates:", compute_win_rates(dummy_results))
