
import re
import json
from typing import Optional


# ── Prompt Template ───────────────────────────────────────────────────────────

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator assessing the REASONING QUALITY of a 
solution to a logic/reasoning problem. You are NOT evaluating the final answer — only the 
quality of the reasoning process.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROBLEM:
{problem}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOLUTION (reasoning trace):
{solution}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Score the reasoning on these 3 dimensions:

1. REASONING DEPTH (1–5):
   Is the causal chain from premise to conclusion tight and well-developed?
   1 = No reasoning shown, just states an answer
   3 = Basic reasoning with some logical steps
   5 = Deep, multi-step reasoning with clear logical connections

2. STEP COMPLETENESS (1–5):
   Are all logical steps present, or are there jumps in reasoning?
   1 = Major steps missing, conclusion not supported
   3 = Most steps present but some gaps
   5 = Every step is explicit, no logical leaps

3. SELF CONSISTENCY (1–5):
   Does the solution contradict itself at any point?
   1 = Multiple contradictions, reasoning conflicts with conclusion
   3 = Minor inconsistencies
   5 = Perfectly consistent throughout

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUCTIONS:
- Score each dimension independently.
- Be strict: a score of 5 should be rare.
- Output ONLY this JSON (no other text):

{{
  "reasoning_depth": <integer 1-5>,
  "step_completeness": <integer 1-5>,
  "self_consistency": <integer 1-5>,
  "brief_justification": "<one sentence explaining the scores>"
}}"""


# ── Response Parser ───────────────────────────────────────────────────────────

EXPECTED_DIMENSIONS = {"reasoning_depth", "step_completeness", "self_consistency"}


def parse_judge_response(response_text: str) -> Optional[dict]:
    # Convert list of dicts to string if using gemini-flash-latest
    if isinstance(response_text, list):
        response_text = "".join([t.get("text", "") if isinstance(t, dict) else str(t) for t in response_text])
    else:
        response_text = str(response_text)

    # Try to extract JSON from markdown code block if present
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
    json_str = code_block.group(1).strip() if code_block else response_text.strip()

    # Try to find the JSON object even if there's surrounding text
    json_match = re.search(r"\{[\s\S]*\}", json_str)
    if json_match:
        json_str = json_match.group(0)

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[llm_judge] JSONDecodeError: {e}")
        print(f"[llm_judge] Raw response was: {response_text}")
        return None

    # Validate all required dimension keys are present
    if not EXPECTED_DIMENSIONS.issubset(parsed.keys()):
        print(f"[llm_judge] Missing dimensions in JSON: {parsed.keys()}")
        return None

    # Validate scores are in valid range
    for dim in EXPECTED_DIMENSIONS:
        val = parsed.get(dim)
        if not isinstance(val, (int, float)) or not (1 <= val <= 5):
            print(f"[llm_judge] Invalid score for {dim}: {val}")
            return None

    return {
        "reasoning_depth"    : int(parsed["reasoning_depth"]),
        "step_completeness"  : int(parsed["step_completeness"]),
        "self_consistency"   : int(parsed["self_consistency"]),
        "brief_justification": str(parsed.get("brief_justification", "")),
    }


# ── Main Scoring Function ─────────────────────────────────────────────────────

def score_reasoning(
    problem     : str,
    solution    : str,
    judge_llm,
    max_retries : int = 3,
) -> dict:
    from langchain_core.messages import HumanMessage

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        problem=problem,
        solution=solution,
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = judge_llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content
            parsed = parse_judge_response(response_text)

            if parsed is not None:
                return {**parsed, "parse_success": True}

            print(f"[llm_judge] Parse failed on attempt {attempt}. Retrying...")

        except Exception as e:
            print(f"[llm_judge] API error on attempt {attempt}: {e}")

    # All retries exhausted — return failure sentinel
    print("[llm_judge] [WARN] All retries exhausted. Returning zero scores.")
    return {
        "reasoning_depth"    : 0,
        "step_completeness"  : 0,
        "self_consistency"   : 0,
        "brief_justification": "PARSE_FAILURE",
        "parse_success"      : False,
    }


# ── Batch Scoring ─────────────────────────────────────────────────────────────

def batch_score_reasoning(
    results   : list[dict],
    judge_llm,
    only_correct: bool = True,
) -> list[dict]:
    scored_results = []

    for i, result in enumerate(results):
        # Skip incorrect answers if only_correct is True
        if only_correct and not result.get("is_correct", False):
            scored_results.append({
                **result,
                "reasoning_scores": {
                    "reasoning_depth": 0,
                    "step_completeness": 0,
                    "self_consistency": 0,
                    "brief_justification": "SKIPPED — incorrect answer",
                    "parse_success": False,
                },
            })
            continue

        print(f"[llm_judge] Scoring {i+1}/{len(results)} — {result.get('problem_id', '?')}")

        scores = score_reasoning(
            problem=result["problem"],
            solution=result["solution"],
            judge_llm=judge_llm,
        )

        scored_results.append({**result, "reasoning_scores": scores})

    return scored_results


# ── Smoke Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test parser
    test_response = """
    ```json
    {
      "reasoning_depth": 4,
      "step_completeness": 3,
      "self_consistency": 5,
      "brief_justification": "Clear reasoning with some gaps in step 3"
    }
    ```
    """
    parsed = parse_judge_response(test_response)
    print("Parsed:", parsed)
