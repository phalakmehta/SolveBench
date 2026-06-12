# ⚡ SolveBench
### A Multi-Agent Benchmarking Framework for Open-Ended Problem Solving

> *"Same problem. Same model. Four different ways agents think. One winner."*

---

## Overview

SolveBench is a LangGraph-based benchmarking framework that pits four different multi-agent architectures against each other on open-ended real-world problems — and measures which collaboration pattern produces the best solutions.

---

## The Core Research Question

When multiple LLM agents collaborate to solve open-ended problems, **does the architecture of collaboration matter** — and if so, which pattern (pipeline, debate, reflection, or solo) produces solutions that are most feasible, complete, creative, and well-reasoned?

And does the answer change across different LLMs and problem difficulty levels?

---

## The 4 Architectures

All four architectures are built in **LangGraph**. Each receives the same problem and produces a solution independently.

### A1 — Solo Agent *(Baseline)*
```
Problem → Agent → Solution
```
Single LLM agent with RAG access. No collaboration. Fast and cheap but no cross-checking. Everything else is measured against this.

---

### A2 — Pipeline Agents
```
Problem → Researcher → Analyst → Solver → Critic → Solution
```
Four specialist agents in sequence. Each does one job:
- **Researcher** — gathers context via RAG
- **Analyst** — breaks the problem into sub-components
- **Solver** — proposes a structured solution
- **Critic** — identifies weaknesses and refines

---

### A3 — Debate Agents
```
Problem → Agent A + Agent B → Argue → Judge → Solution
```
Two agents independently propose solutions then argue against each other's weaknesses. A **Judge agent** reads both arguments and synthesises the strongest final answer.

---

### A4 — Reflection Agent
```
Problem → Solve → Critique → Revise → Critique → Revise → Solution
```
One agent solves, a separate **Critic agent** tears it apart, then the solver revises. Loops 3 times. Measures whether self-improvement actually works.

---

## Problem Dataset

**60 open-ended real-world problems** across 3 difficulty levels:

| Difficulty | Count | Example |
|---|---|---|
| Easy | 20 | "How would you reduce plastic use in a college canteen?" |
| Medium | 20 | "Design a system to reduce traffic congestion in a tier-2 Indian city." |
| Hard | 20 | "How would you redesign India's public healthcare system given budget constraints?" |

Problems are intentionally open-ended with no single correct answer — this forces evaluation of *reasoning quality* over factual correctness.

---

## Evaluation Framework

### Layer 1 — LLM-as-Judge
A separate Gemma 2 9B agent scores every solution on 4 dimensions:

| Dimension | Description | Scale |
|---|---|---|
| Feasibility | Is this actually doable in the real world? | 1–5 |
| Completeness | Does it address all parts of the problem? | 1–5 |
| Creativity | Does it propose something non-obvious? | 1–5 |
| Reasoning Depth | Are claims backed by logic, not just assertions? | 1–5 |

### Layer 2 — Automated Metrics
Objective measurements requiring no human or LLM:
- **Latency** — time taken per architecture
- **Token cost** — tokens consumed per solution
- **Self-contradiction score** — does the solution contradict itself?
- **Solution density** — information per 100 tokens

### Layer 3 — Human Evaluation
5 human evaluators rate a sample of 20 solutions to validate the LLM judge. Measures: **does the LLM judge agree with humans?** (Pearson correlation score per dimension)

---

## Experiment Matrix

**4 architectures × 2 models × 60 problems = 480 solutions evaluated**

| Architecture | Llama 3 8B (Groq) | Gemma 2 9B (Groq) |
|---|---|---|
| A1 Solo | 60 problems | 60 problems |
| A2 Pipeline | 60 problems | 60 problems |
| A3 Debate | 60 problems | 60 problems |
| A4 Reflection | 60 problems | 60 problems |

Enough runs for statistically significant results across all comparisons.

---

## Ablation Studies

| Ablation | What It Tests |
|---|---|
| RAG vs No RAG | Does retrieved context improve solution quality or just add noise? |
| Debate — 2 vs 3 rounds | Does one more argument round improve the final solution or plateau? |
| Reflection — 2 vs 3 loops | At what point does self-critique stop helping and start over-correcting? |
| LLM judge vs Human judge | How well does automated evaluation match human judgment? Where does it fail? |

---

## Expected Findings

> *"Pipeline and Debate architectures outperform Solo agent on hard problems by X% but show no significant advantage on easy problems — suggesting architectural complexity only pays off above a certain problem difficulty threshold."*

> *"Reflection architecture produces the highest Reasoning Depth scores but lowest Creativity scores — self-critique optimises for safety over novelty."*

> *"LLM-as-judge scores correlate with human evaluators at 0.78 Pearson on Feasibility but only 0.51 on Creativity — automated evaluation is unreliable for subjective dimensions."*

> *"RAG grounding improves Feasibility scores by ~18% but has no measurable effect on Creativity — retrieved context anchors solutions in reality but constrains imagination."*

---

## Tech Stack — 100% Free

| Component | Tool |
|---|---|
| Agent orchestration | LangGraph |
| LLM chaining | LangChain |
| LLM inference | Groq API (free tier) |
| Models | Llama 3 8B, Gemma 2 9B |
| Vector store | FAISS (local) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Evaluation | RAGAS |
| Experiment tracking | MLflow (local) |
| Pipeline orchestration | n8n |
| Demo UI | Streamlit |

---

## Build Plan

| Week | Milestone |
|---|---|
| Week 1 | Build all 4 LangGraph architectures, test on 5 problems each |
| Week 2 | Build evaluation layer — LLM judge, automated metrics, MLflow tracking |
| Week 3 | Full experiment run — 60 problems × 4 architectures × 2 models |
| Week 4 | Ablation studies + human evaluation on 20-solution sample |
| Week 5 | Streamlit dashboard, result visualisation, GitHub cleanup, write-up |

---

## Resume Line

> *"Built SolveBench — a LangGraph-based framework benchmarking 4 multi-agent architectures (solo, pipeline, debate, reflection) across 480 solutions on open-ended problems, evaluated with LLM-as-judge and human raters, finding that debate architecture outperforms solo agents by X% on complex problems but costs 3x more in latency."*

---

## Project Structure (Proposed)

```
solvebench/
├── architectures/
│   ├── solo_agent.py
│   ├── pipeline_agents.py
│   ├── debate_agents.py
│   └── reflection_agent.py
├── evaluation/
│   ├── llm_judge.py
│   ├── automated_metrics.py
│   └── human_eval_collector.py
├── data/
│   ├── problems_easy.json
│   ├── problems_medium.json
│   └── problems_hard.json
├── rag/
│   ├── indexer.py
│   └── retriever.py
├── experiments/
│   └── run_all.py
├── dashboard/
│   └── app.py
├── mlflow_tracking/
└── README.md
```

---

*SolveBench — built by Phalak Mehta, IIITS*
