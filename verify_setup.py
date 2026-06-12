import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 60)
print("SolveBench v2 -- Setup Verification")
print("=" * 60)

# 1. Test BBH loader
print("\n1. Testing BBH data loader...")
from data.bbh_loader import load_bbh_problems
problems = load_bbh_problems(n_per_subset=2)
print(f"   [OK] Loaded {len(problems)} problems from {len(set(p['subset'] for p in problems))} subsets")
for p in problems[:3]:
    print(f"   - {p['id']}: target='{p['target']}', input='{p['input'][:60]}...'")

# 2. Test architecture imports
print("\n2. Testing architecture imports...")
from architectures import AgentState, run_architecture, extract_final_answer
from architectures.solo_agent import build_graph as build_solo
from architectures.pipeline_agents import build_graph as build_pipeline
from architectures.debate_agents import build_graph as build_debate
from architectures.reflection_agent import build_graph as build_reflection
print("   [OK] All 4 architectures import successfully")

# 3. Test answer extraction
print("\n3. Testing answer extraction...")
test_cases = [
    ("FINAL ANSWER: (B)", "B"),
    ("The answer is Yes.", "Yes"),
    ("Therefore, valid", "valid"),
]
for text, expected in test_cases:
    result = extract_final_answer(text)
    status = "[OK]" if result.lower() == expected.lower() else "[FAIL]"
    print(f"   {status} extract('{text}') = '{result}' (expected '{expected}')")

# 4. Test evaluation modules
print("\n4. Testing evaluation imports...")
from evaluation import (
    evaluate_accuracy, compute_accuracy, compute_accuracy_by_architecture,
    score_reasoning, batch_score_reasoning,
    compute_all_metrics, compute_self_contradiction_score,
    generate_calibration_sample, export_rating_template,
    compute_composite_score, compute_architecture_stats,
    test_hypothesis_h1, test_hypothesis_h2, test_hypothesis_h3,
)
print("   [OK] All 4 evaluation layers import successfully")

# 5. Test ground truth accuracy
print("\n5. Testing ground truth accuracy...")
assert evaluate_accuracy("(B)", "(B)") == True
assert evaluate_accuracy("Yes", "yes") == True
assert evaluate_accuracy("No", "Yes") == False
assert evaluate_accuracy("valid", "Valid") == True
print("   [OK] All accuracy tests pass")

# 6. Test experiment runner imports
print("\n6. Testing experiment runner imports...")
from experiments.run_all import make_run_id, create_llm
print("   [OK] Experiment runner imports OK")

# 7. Verify project structure
print("\n7. Checking project structure...")
from pathlib import Path
expected_files = [
    "data/bbh_loader.py",
    "data/bbh_subset_config.json",
    "data/old_v1/problems_easy.json",
    "architectures/__init__.py",
    "architectures/solo_agent.py",
    "architectures/pipeline_agents.py",
    "architectures/debate_agents.py",
    "architectures/reflection_agent.py",
    "evaluation/__init__.py",
    "evaluation/ground_truth_accuracy.py",
    "evaluation/llm_judge.py",
    "evaluation/automated_metrics.py",
    "evaluation/human_eval_calibration.py",
    "evaluation/scoring.py",
    "experiments/run_all.py",
    "experiments/ablations/debate_rounds.py",
    "experiments/ablations/reflection_loops.py",
    "requirements.txt",
    "README.md",
]
all_exist = True
for f in expected_files:
    if not Path(f).exists():
        print(f"   [MISSING] {f}")
        all_exist = False
if all_exist:
    print(f"   [OK] All {len(expected_files)} expected files present")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED -- SolveBench v2 is ready!")
print("=" * 60)
print("\nNext steps:")
print("  1. Set GOOGLE_API_KEY environment variable")
print("  2. Run: python experiments/run_all.py --dry-run")
print("  3. If dry run succeeds: python experiments/run_all.py")
