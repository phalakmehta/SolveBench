
from evaluation.ground_truth_accuracy import (
    evaluate_accuracy,
    normalize_answer,
    compute_accuracy,
    compute_accuracy_by_architecture,
    compute_accuracy_by_subset,
    compute_accuracy_by_arch_and_subset,
    tag_results_with_accuracy,
)

from evaluation.llm_judge import (
    score_reasoning,
    batch_score_reasoning,
    parse_judge_response,
)

from evaluation.automated_metrics import (
    compute_latency,
    compute_token_cost,
    compute_self_contradiction_score,
    compute_nli_contradiction_score,
    compute_heuristic_contradiction,
    compute_all_metrics,
)

from evaluation.scoring import (
    normalize_score,
    normalize_scores,
    compute_composite_score,
    compute_architecture_stats,
    compute_win_rates,
    compute_scores_by_subset,
    DIMENSION_WEIGHTS,
    DIMENSIONS,
    test_hypothesis_h1,
    test_hypothesis_h2,
    test_hypothesis_h3,
    test_hypothesis_h4,
    test_hypothesis_h5,
)

from evaluation.human_eval_calibration import (
    generate_calibration_sample,
    export_rating_template,
    compute_judge_correlation,
)

__all__ = [
    # Layer 1 — Ground Truth Accuracy
    "evaluate_accuracy",
    "normalize_answer",
    "compute_accuracy",
    "compute_accuracy_by_architecture",
    "compute_accuracy_by_subset",
    "compute_accuracy_by_arch_and_subset",
    "tag_results_with_accuracy",
    # Layer 2 — LLM Judge
    "score_reasoning",
    "batch_score_reasoning",
    "parse_judge_response",
    # Layer 3 — Automated Metrics
    "compute_latency",
    "compute_token_cost",
    "compute_self_contradiction_score",
    "compute_nli_contradiction_score",
    "compute_heuristic_contradiction",
    "compute_all_metrics",
    # Layer 4 — Human Eval Calibration
    "generate_calibration_sample",
    "export_rating_template",
    "compute_judge_correlation",
    # Scoring Utils
    "normalize_score",
    "normalize_scores",
    "compute_composite_score",
    "compute_architecture_stats",
    "compute_win_rates",
    "compute_scores_by_subset",
    "DIMENSION_WEIGHTS",
    "DIMENSIONS",
    # Hypothesis Testing
    "test_hypothesis_h1",
    "test_hypothesis_h2",
    "test_hypothesis_h3",
    "test_hypothesis_h4",
    "test_hypothesis_h5",
]
