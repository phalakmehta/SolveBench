# SolveBench ⚡

**SolveBench** is a benchmarking framework that tests whether multi-agent AI architectures genuinely outperform single-agent baselines on hard logic puzzles. 

We pitted 4 different architectures against 50 of the hardest logic problems from the **BIG-bench Hard (BBH)** dataset using the `open-mixtral-8x7b` model.

## The 4 Architectures Tested:
1. **A1 Solo Agent:** One model solving the problem end-to-end.
2. **A2 Pipeline Agent:** An assembly line (Draft → Review → Finalize).
3. **A3 Debate Agent:** Two models argue, a third acts as a judge.
4. **A4 Reflection Agent:** One model drafts, checks its own work, and corrects mistakes.

## The Surprising Results 🏆
More agents ≠ Better results!
- **Solo (A1)** and **Reflection (A4)** tied for 1st place with **70% accuracy**.
- **Debate (A3)** lagged behind at **62%**.
- **Pipeline (A2)** did the worst at **58%**.

Forcing identical models into a team created a "too many cooks" problem where they confused each other, while the Solo agent succeeded with a fraction of the token cost and latency.

## Quickstart

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your API key:
```bash
export MISTRAL_API_KEY="your-key-here"
```

3. Run the experiment:
```bash
python experiments/run_all.py
```

4. View the interactive dashboard (React):
```bash
cd bbh_dashboard
npm install
npm run dev
```

