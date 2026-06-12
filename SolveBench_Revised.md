# SolveBench v2.0
### A Rigorous Multi-Agent Benchmarking Framework for Reasoning Tasks

> *"Four architectures. One standard dataset. Validated evaluation. Reproducible results."*

---

## What Changed From v1 and Why

The original design had three structural problems that would make results untrustworthy:

**Problem 1 — Custom dataset with no ground truth**
60 self-authored open-ended questions have no external validation. An LLM judge scoring them is an opinion, not a measurement. You cannot compare your results to any prior work.

**Problem 2 — Weak models showing no architectural variance**
Groq's Llama 3 8B and Gemma 2 9B lack sufficient reasoning capacity to differentiate between a solo agent and a 4-step pipeline. All architectures converge to similar outputs, making the benchmark meaningless.

**Problem 3 — No evaluation ground truth**
Without a validated judge or known answer baselines, your 4-dimension scores (Feasibility, Completeness, Creativity, Reasoning Depth) cannot be trusted. There is no way to prove the judge is measuring what it claims to measure.

**v2.0 fixes all three before adding complexity.**

---

## Revised Research Question

> When multiple LLM agents collaborate to solve reasoning-intensive tasks, does the architecture of collaboration (pipeline, debate, reflection, solo) produce measurably different outputs — and does this difference scale with task difficulty?

**Two sub-questions with testable hypotheses:**

- H1: Architectural complexity improves performance on hard reasoning tasks but not easy ones (complexity threshold hypothesis)
- H2: Debate architecture maximises answer diversity; Reflection architecture maximises self-consistency; Pipeline maximises completeness

---

## Dataset: BIG-Bench Hard (BBH)

**Why BBH:**
- Standard benchmark used in published LLM research — results are citable and comparable
- 23 task categories, all require multi-step reasoning
- Hard enough that architecture differences surface (GPT-4 scores ~50% on many tasks)
- Community baselines available for validation
- Free, no API cost to access the dataset
- Some tasks have exact-match ground truth; others have human evaluation scores

**Selected Subsets for SolveBench:**

| Subset | Task Type | Ground Truth | Reason for Selection |
|---|---|---|---|
| Logical Deduction (5 objects) | Deductive reasoning | Exact match | Clear right/wrong answer — architecture differences are measurable |
| Causal Judgement | Causal reasoning | Exact match | Tests reasoning chains, not recall |
| Web of Lies | Multi-step logic | Exact match | Requires tracking state across steps — pipeline should help |
| Formal Fallacies | Argument evaluation | Exact match | Debate architecture naturally fits |
| Navigate | Spatial reasoning | Exact match | Tests consistency — good for Reflection architecture |

**Problem count: 25 problems per subset × 5 subsets = 125 problems**

Reduced to **50 problems (10 per subset)** for the experiment — stratified random sample, reproducible via fixed seed.

**Why 50 and not 60:**
50 problems with 4 architectures × 1 model = 200 runs. Each run is validated against ground truth. 200 high-quality validated runs beat 480 unvalidated runs.

---

## Model Selection

**One model only: Gemini 1.5 Flash (Google AI Studio)**

**Why one model:**
- Original design used 2 models to show model-level differences — but with validated BBH tasks, architecture differences are the interesting variable, not model differences
- Adding a second model doubles token cost and complexity without adding to the core research question
- If results are strong, "tested on additional model" is a future work item, not a requirement

**Why Gemini 1.5 Flash specifically:**
- Free tier: 15 RPM, 1 million tokens/day — sufficient for 200 runs
- 1M context window — pipeline agents can see full conversation history
- Significantly stronger reasoning than Llama 3 8B — will show architectural variance
- Available via LangChain's `ChatGoogleGenerativeAI` — zero code change to existing agents

**Why not Groq:**
Groq's 8B/9B models collapse architectural differences. The existing A1 vs A2 data from v1 already demonstrates this (16.6 vs 16.6 on overall quality). Using the same models for v2 would produce the same problem.

---

## The 4 Architectures (Unchanged)

All four architectures remain identical to v1. The only change is the input and the model.

### A1 — Solo Agent (Baseline)
```
Problem → Agent → Solution
```

### A2 — Pipeline Agents
```
Problem → Researcher → Analyst → Solver → Critic → Solution
```

### A3 — Debate Agents
```
Problem → Agent A + Agent B → Argue → Judge → Solution
```

### A4 — Reflection Agent
```
Problem → Solve → Critique → Revise → Critique → Revise → Solution
```
*(3 reflection loops)*

---

## Evaluation Framework (Revised)

The evaluation now has two layers with clear separation of concerns.

### Layer 1 — Ground Truth Accuracy (Primary Metric)

For all 5 BBH subsets selected, answers have exact-match ground truth.

```
Accuracy = correct_answers / total_problems
```

This is the primary metric. It is objective, requires no judge, and is directly comparable to published BBH leaderboard scores.

**Why this is the anchor metric:**
No LLM judge subjectivity. No calibration needed. Immediately credible to anyone reviewing the project.

### Layer 2 — Reasoning Quality (Secondary Metric)

For problems where the answer is correct, *how* the agent got there still matters. A judge scores the reasoning trace, not the final answer.

**Judge model: Gemini 1.5 Flash (separate instance, not the solver)**

**3 dimensions only (down from 4):**

| Dimension | Description | Scale |
|---|---|---|
| Reasoning Depth | Is the causal chain from premise to conclusion tight? | 1–5 |
| Step Completeness | Are all logical steps present or are there jumps? | 1–5 |
| Self-Consistency | Does the solution contradict itself at any point? | 1–5 |

**Why 3 dimensions instead of 4:**
Feasibility and Creativity are subjective and hard to validate — they were the weakest dimensions in v1. The 3 remaining dimensions map directly to what multi-agent reasoning should improve and can be validated against human raters.

### Layer 3 — Automated Metrics (Objective)

| Metric | Description |
|---|---|
| Accuracy | Exact match against BBH ground truth |
| Latency | Wall-clock time per architecture per problem |
| Token cost | Total tokens consumed per run |
| Self-contradiction score | Automated: does output contradict itself (NLI model) |
| Answer stability | Run same problem 3 times — variance in final answer (for non-deterministic sampling) |

### Layer 4 — Judge Validation (One-Time Calibration)

To prove the LLM judge is trustworthy:
- Sample 20 problems from Layer 2 results
- Have 3 human raters score the same reasoning traces
- Compute Pearson correlation between LLM judge scores and human scores per dimension
- Report this as the judge reliability score in the paper/report

**Acceptance threshold: r > 0.70 on all 3 dimensions**
If any dimension falls below 0.70, that dimension is dropped from analysis.

---

## Experiment Matrix

**4 architectures × 1 model × 50 problems = 200 runs**

| Architecture | BBH (50 problems) |
|---|---|
| A1 Solo | 50 problems |
| A2 Pipeline | 50 problems |
| A3 Debate | 50 problems |
| A4 Reflection | 50 problems |

**Per-subset breakdown:**

| BBH Subset | Problems | Expected A1 Accuracy | Why Interesting |
|---|---|---|---|
| Logical Deduction | 10 | ~40% | Pipeline should help with structured steps |
| Causal Judgement | 10 | ~55% | Debate should surface causal disagreements |
| Web of Lies | 10 | ~45% | Reflection should catch state-tracking errors |
| Formal Fallacies | 10 | ~50% | Debate architecture maps directly to fallacy detection |
| Navigate | 10 | ~35% | Consistency metric most meaningful here |

Expected accuracy baselines sourced from the original BBH paper (Suzgun et al., 2022).

---

## What Happens to the v1 Data (60 Custom Questions, A1+A2)

The existing A1 vs A2 runs on 60 custom questions are **not thrown away**.

They are repositioned as a **supplementary analysis**:

> *"We additionally evaluated A1 and A2 on a set of 60 India-specific open-ended problems to assess performance in a domain not covered by standard benchmarks. Results showed [finding]. This suggests [implication]."*

This is honest — the data exists, it has limitations (no ground truth, unvalidated judge), and those limitations are stated explicitly. It adds to the project without overclaiming.

**What to remove from codebase:**
- `problems_easy.json`, `problems_medium.json`, `problems_hard.json` — archive, do not delete
- The LLM judge prompt tuned for open-ended India problems — replace with BBH-specific judge prompt
- Any hardcoded references to 60-problem count in `run_all.py`

---

## Hypotheses and Expected Findings

These are stated before running experiments — this is what separates a research project from a demo.

| Hypothesis | Expected Result | How to Test |
|---|---|---|
| H1: Complexity threshold | A3/A4 outperform A1 on hard tasks but not easy | Accuracy by task difficulty |
| H2: Pipeline improves completeness | A2 has higher Step Completeness score than A1 | Layer 2 dimension score |
| H3: Debate improves accuracy on argument tasks | A3 accuracy > A1 on Formal Fallacies subset | Subset-level accuracy |
| H4: Reflection reduces self-contradiction | A4 self-contradiction score < A1 | Automated NLI metric |
| H5: Architecture gains come at cost | A3/A4 latency and token cost 3–5x A1 | Layer 3 automated metrics |

**If results contradict a hypothesis, that is a valid and interesting finding — state it honestly.**

---

## Ablation Studies (2 only, not 4)

Reduced to 2 ablations that are directly testable and add real insight:

| Ablation | What It Tests | How |
|---|---|---|
| Debate: 2 rounds vs 3 rounds | Does an extra argument round improve accuracy or plateau? | Run A3 with 2 and 3 rounds on same 50 problems |
| Reflection: 2 loops vs 3 loops | At what loop count does self-critique stop improving accuracy? | Run A4 with 2 and 3 loops on same 50 problems |

---

## Tech Stack

| Component | Tool | Reason |
|---|---|---|
| Agent orchestration | LangGraph | Unchanged |
| LLM chaining | LangChain | Unchanged |
| LLM inference | Google AI Studio (Gemini 1.5 Flash) | Free, stronger reasoning than Groq 8B |
| Dataset | BIG-Bench Hard (HuggingFace datasets) | Standard, citable, ground truth available |
| Evaluation judge | Gemini 1.5 Flash (separate instance) | Same model, separate system prompt |
| NLI for contradiction | `cross-encoder/nli-deberta-v3-small` (HuggingFace) | Free, local, no API cost |
| Experiment tracking | MLflow (local) | Unchanged |
| Checkpointing | JSON per problem, per architecture | Never re-run completed problems |
| Demo UI | Streamlit | Unchanged |

---

## Revised Build Plan

| Week | Milestone |
|---|---|
| Week 1 | Swap model to Gemini 1.5 Flash. Load BBH dataset via HuggingFace. Test all 4 architectures on 5 problems each. Confirm variance exists across architectures. |
| Week 2 | Run full 200-problem experiment. Implement checkpointing so runs can be paused and resumed. Log all outputs to MLflow. |
| Week 3 | Build evaluation layer — ground truth accuracy, LLM judge for reasoning quality, NLI contradiction scorer. Run Layer 1 + 2 + 3 on all 200 outputs. |
| Week 4 | Human evaluation calibration (20 problems, 3 raters). Ablation studies (2 ablations × 50 problems = 100 additional runs). |
| Week 5 | Streamlit dashboard, result visualisation, write findings against hypotheses, GitHub cleanup. |

---

## Project Structure (Revised)

```
solvebench/
├── architectures/
│   ├── solo_agent.py
│   ├── pipeline_agents.py
│   ├── debate_agents.py
│   └── reflection_agent.py
├── evaluation/
│   ├── ground_truth_accuracy.py     # Layer 1 — exact match vs BBH answers
│   ├── llm_judge.py                 # Layer 2 — reasoning quality (3 dimensions)
│   ├── automated_metrics.py         # Layer 3 — latency, tokens, contradiction
│   └── human_eval_calibration.py    # Layer 4 — judge validation
├── data/
│   ├── bbh_loader.py                # Load BBH subsets via HuggingFace datasets
│   ├── bbh_subset_config.json       # Which subsets, how many problems, random seed
│   └── archive/                     # v1 custom 60 questions — kept, not used in main experiment
├── experiments/
│   ├── run_all.py                   # Main runner with checkpointing
│   ├── checkpoints/                 # JSON outputs per problem per architecture
│   └── ablations/
│       ├── debate_rounds.py
│       └── reflection_loops.py
├── dashboard/
│   └── app.py
├── mlflow_tracking/
├── results/
│   ├── accuracy_by_architecture.csv
│   ├── reasoning_scores.csv
│   └── automated_metrics.csv
└── README.md
```

---

## Resume Line (Revised)

> *"Built SolveBench v2 — a LangGraph-based framework benchmarking 4 multi-agent architectures (solo, pipeline, debate, reflection) on BIG-Bench Hard reasoning tasks. Evaluated 200 runs across 5 task categories using ground-truth accuracy, LLM-as-judge (validated against human raters at r=0.XX), and automated NLI-based contradiction scoring. Found that debate architecture improves accuracy by X% on formal reasoning tasks but shows no advantage on spatial reasoning, while reflection reduces self-contradiction rate by Y% at 3x the token cost."*

---

## What Makes This Project Defensible

1. **Standard dataset** — BBH is citable. Any interviewer can look it up.
2. **Ground truth primary metric** — accuracy is not an opinion.
3. **Validated judge** — human correlation score proves the LLM judge works.
4. **Pre-registered hypotheses** — stated before running, not after seeing results.
5. **Honest limitations** — one model, 50 problems, specific BBH subsets. Stated clearly.
6. **v1 data retained** — shows research continuity, not a dead end.

---

*SolveBench v2.0 — Phalak Mehta, IIITS*
