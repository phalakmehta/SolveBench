
import re
import time
from typing import Optional

# ── Pricing Table (Gemini 1.5 Flash, USD per 1M tokens) ─────────────────────

TOKEN_PRICING = {
    "gemini-1.5-flash"        : {"prompt": 0.075, "completion": 0.30},
    "gemini-2.5-flash"        : {"prompt": 0.075, "completion": 0.30},
    "gemini-1.5-flash-latest" : {"prompt": 0.075, "completion": 0.30},
    "gemini-1.5-pro"          : {"prompt": 1.25,  "completion": 5.00},
    "models/gemini-1.5-flash" : {"prompt": 0.075, "completion": 0.30},
    # Legacy Groq models (for supplementary analysis)
    "llama-3.1-8b-instant"    : {"prompt": 0.05,  "completion": 0.08},
    "llama-3.3-70b-versatile" : {"prompt": 0.59,  "completion": 0.79},
    # Fallback
    "default"                 : {"prompt": 0.10,  "completion": 0.10},
}


# ── 1. Latency ────────────────────────────────────────────────────────────────

def compute_latency(start_time: float, end_time: Optional[float] = None) -> float:
    if end_time is None:
        end_time = time.time()
    return round(end_time - start_time, 2)


# ── 2. Token Cost ─────────────────────────────────────────────────────────────

def compute_token_cost(
    prompt_tokens    : int,
    completion_tokens: int,
    model            : str = "gemini-flash-latest",
) -> dict:
    pricing = TOKEN_PRICING.get(model, TOKEN_PRICING["default"])

    cost = (
        prompt_tokens     / 1_000_000 * pricing["prompt"] +
        completion_tokens / 1_000_000 * pricing["completion"]
    )

    return {
        "prompt_tokens"    : prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens"     : prompt_tokens + completion_tokens,
        "estimated_usd"    : round(cost, 8),
    }


# ── 3. NLI Contradiction Score ────────────────────────────────────────────────

_nli_model = None
_nli_available = None


def _get_nli_model():
    global _nli_model, _nli_available

    if _nli_available is False:
        return None

    if _nli_model is not None:
        return _nli_model

    try:
        from sentence_transformers import CrossEncoder
        print("[automated_metrics] Loading NLI model: cross-encoder/nli-deberta-v3-small ...")
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small", local_files_only=True)
        _nli_available = True
        print("[automated_metrics] NLI model loaded successfully.")
        return _nli_model
    except Exception as e:
        print(f"[automated_metrics] NLI model unavailable: {e}. Using heuristic fallback.")
        _nli_available = False
        return None


def compute_nli_contradiction_score(text: str) -> Optional[float]:
    model = _get_nli_model()
    if model is None:
        return None

    # Split into sentences (simple regex split)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 15]

    if len(sentences) < 2:
        return 0.0

    # Generate sentence pairs (not all pairs — sample for efficiency)
    pairs = []
    max_pairs = 5  # cap to avoid extremely slow CPU computation
    step = max(1, len(sentences) * (len(sentences) - 1) // 2 // max_pairs)

    count = 0
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            count += 1
            if count % step == 0:
                pairs.append((sentences[i], sentences[j]))
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break

    if not pairs:
        return 0.0

    # NLI classification: labels are ['contradiction', 'entailment', 'neutral']
    scores = model.predict(pairs)

    # scores shape: (n_pairs, 3) — [contradiction, entailment, neutral]
    contradiction_count = sum(1 for s in scores if s.argmax() == 0)

    return round(contradiction_count / len(pairs), 4)


# ── 4. Heuristic Contradiction Score (Fallback) ──────────────────────────────

NEGATION_PAIRS = [
    (r"\bis\b",      r"\bis not\b|\bisn't\b"),
    (r"\bwill\b",    r"\bwill not\b|\bwon't\b"),
    (r"\bshould\b",  r"\bshould not\b|\bshouldn't\b"),
    (r"\bcan\b",     r"\bcannot\b|\bcan't\b"),
    (r"\bmust\b",    r"\bmust not\b|\bmustn't\b"),
    (r"\balways\b",  r"\bnever\b|\brarely\b"),
    (r"\btrue\b",    r"\bfalse\b"),
    (r"\byes\b",     r"\bno\b"),
    (r"\bvalid\b",   r"\binvalid\b"),
]


def compute_heuristic_contradiction(text: str) -> float:
    text_lower = text.lower()
    triggered = 0

    for positive_pat, negative_pat in NEGATION_PAIRS:
        if re.search(positive_pat, text_lower) and re.search(negative_pat, text_lower):
            triggered += 1

    total_pairs = len(NEGATION_PAIRS)
    if total_pairs == 0:
        return 0.0

    return round(triggered / total_pairs, 4)


# ── 5. Self-Contradiction (unified interface) ─────────────────────────────────

def compute_self_contradiction_score(text: str) -> dict:
    nli_score = compute_nli_contradiction_score(text)
    if nli_score is not None:
        return {"score": nli_score, "method": "nli"}

    return {"score": compute_heuristic_contradiction(text), "method": "heuristic"}


# ── 6. All Metrics (convenience wrapper) ──────────────────────────────────────

def compute_all_metrics(state: dict) -> dict:
    meta = state.get("metadata", {})

    latency = meta.get("latency_seconds", 0.0)

    token_info = compute_token_cost(
        prompt_tokens    =meta.get("prompt_tokens", 0),
        completion_tokens=meta.get("completion_tokens", 0),
        model            =meta.get("model", "default"),
    )

    contradiction = compute_self_contradiction_score(state.get("solution", ""))

    return {
        "latency_seconds"     : latency,
        "prompt_tokens"       : token_info["prompt_tokens"],
        "completion_tokens"   : token_info["completion_tokens"],
        "total_tokens"        : token_info["total_tokens"],
        "estimated_usd"       : token_info["estimated_usd"],
        "self_contradiction"  : contradiction,
    }


# ── Smoke Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = """
    Step 1: We know that Alice is taller than Bob. This is true.
    Step 2: Bob is taller than Claire. This is also true.
    Step 3: Therefore Alice is taller than Claire. This is false.
    Wait, that contradicts step 1 and 2. Alice must be taller than Claire.
    The answer is Yes.
    """

    print("Contradiction:", compute_self_contradiction_score(sample))
    print("Token cost:", compute_token_cost(500, 800, "gemini-flash-latest"))
    print("Latency:", compute_latency(time.time() - 5.3))
