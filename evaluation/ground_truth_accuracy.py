
import re
from typing import Optional


# ── Single Answer Evaluation ──────────────────────────────────────────────────

def normalize_answer(answer: str) -> str:
    text = answer.strip().lower()

    # Remove common prefixes
    for prefix in ["the answer is", "answer:", "answer is", "therefore,"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Strip surrounding punctuation and whitespace
    text = text.strip(" .,:;!?\"'`")

    # Handle parenthetical answers: "(A)" → "a", "(A) Some text" → "a"
    paren_match = re.match(r"^\(?([a-z])\)?(?:\s.*)?$", text)
    if paren_match:
        text = paren_match.group(1)

    # Handle "valid" / "invalid" for formal fallacies
    # Handle "Yes" / "No" for boolean tasks
    # These are already lowercase at this point

    return text


def evaluate_accuracy(predicted: str, ground_truth: str) -> bool:
    norm_pred = normalize_answer(predicted)
    norm_gt   = normalize_answer(ground_truth)

    return norm_pred == norm_gt


# ── Batch Accuracy Computation ────────────────────────────────────────────────

def compute_accuracy(results: list[dict]) -> float:
    if not results:
        return 0.0

    correct = sum(
        1 for r in results
        if evaluate_accuracy(r["extracted_answer"], r["ground_truth"])
    )
    return round(correct / len(results), 4)


def compute_accuracy_by_architecture(results: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[dict]] = {}
    for r in results:
        arch = r["architecture"]
        grouped.setdefault(arch, []).append(r)

    return {
        arch: compute_accuracy(arch_results)
        for arch, arch_results in sorted(grouped.items())
    }


def compute_accuracy_by_subset(results: list[dict]) -> dict[str, dict[str, float]]:
    # Group by subset
    by_subset: dict[str, list[dict]] = {}
    for r in results:
        subset = r["subset"]
        by_subset.setdefault(subset, []).append(r)

    output = {}
    for subset, subset_results in sorted(by_subset.items()):
        output[subset] = compute_accuracy_by_architecture(subset_results)

    return output


def compute_accuracy_by_arch_and_subset(results: list[dict]) -> dict[str, dict[str, float]]:
    by_arch: dict[str, list[dict]] = {}
    for r in results:
        arch = r["architecture"]
        by_arch.setdefault(arch, []).append(r)

    output = {}
    for arch, arch_results in sorted(by_arch.items()):
        by_subset: dict[str, list[dict]] = {}
        for r in arch_results:
            subset = r["subset"]
            by_subset.setdefault(subset, []).append(r)

        output[arch] = {
            subset: compute_accuracy(subset_results)
            for subset, subset_results in sorted(by_subset.items())
        }

    return output


# ── Result Tagging ────────────────────────────────────────────────────────────

def tag_results_with_accuracy(results: list[dict]) -> list[dict]:
    for r in results:
        r["is_correct"] = evaluate_accuracy(
            r.get("extracted_answer", ""),
            r.get("ground_truth", ""),
        )
    return results


# ── Smoke Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test normalization
    test_cases = [
        ("(B)", "(B)", True),
        ("Yes", "yes", True),
        ("Valid", "valid", True),
        ("The answer is (C)", "(C)", True),
        ("(A)", "(B)", False),
        ("No", "Yes", False),
        ("invalid", "valid", False),
    ]

    print("Testing answer normalization:")
    for pred, gt, expected in test_cases:
        result = evaluate_accuracy(pred, gt)
        status = "✓" if result == expected else "✗"
        print(f"  {status} evaluate_accuracy('{pred}', '{gt}') = {result} (expected {expected})")

    # Test batch accuracy
    dummy_results = [
        {"architecture": "A1_solo", "subset": "web_of_lies",
         "extracted_answer": "Yes", "ground_truth": "Yes"},
        {"architecture": "A1_solo", "subset": "web_of_lies",
         "extracted_answer": "No", "ground_truth": "Yes"},
        {"architecture": "A2_pipeline", "subset": "web_of_lies",
         "extracted_answer": "Yes", "ground_truth": "Yes"},
        {"architecture": "A2_pipeline", "subset": "web_of_lies",
         "extracted_answer": "Yes", "ground_truth": "Yes"},
    ]

    print(f"\nOverall accuracy: {compute_accuracy(dummy_results)}")
    print(f"By architecture: {compute_accuracy_by_architecture(dummy_results)}")
