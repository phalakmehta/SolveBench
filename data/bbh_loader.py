
import json
import random
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "bbh_subset_config.json"


def load_bbh_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Main Loader ──────────────────────────────────────────────────────────────

def load_bbh_problems(
    n_per_subset: Optional[int] = None,
    seed: Optional[int] = None,
    config_path: Optional[str] = None,
) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The `datasets` library is required to load BBH data.\n"
            "Install it with: pip install datasets"
        )

    # Load config
    cfg_path = Path(config_path) if config_path else CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    rng_seed = seed if seed is not None else config["random_seed"]
    dataset_source = config["dataset_source"]

    all_problems = []

    for subset_cfg in config["subsets"]:
        subset_name = subset_cfg["name"]
        sample_count = n_per_subset if n_per_subset is not None else subset_cfg["sample_count"]
        display_name = subset_cfg["display_name"]
        task_type = subset_cfg["task_type"]

        print(f"[bbh_loader] Loading subset: {subset_name} ...")

        # Load the subset from HuggingFace
        # lukaemon/bbh has each task as a separate config
        dataset = load_dataset(dataset_source, subset_name, split="test")

        # Stratified random sample with fixed seed
        rng = random.Random(rng_seed)
        indices = list(range(len(dataset)))

        if sample_count >= len(indices):
            sampled_indices = indices
        else:
            sampled_indices = sorted(rng.sample(indices, sample_count))

        for idx in sampled_indices:
            row = dataset[idx]
            problem_id = f"{subset_name}_{idx:03d}"

            all_problems.append({
                "id":             problem_id,
                "subset":         subset_name,
                "subset_display": display_name,
                "task_type":      task_type,
                "input":          row["input"],
                "target":         row["target"],
                "index":          idx,
            })

        print(f"[bbh_loader]   -> Sampled {len(sampled_indices)} / {len(dataset)} problems")

    print(f"[bbh_loader] Total problems loaded: {len(all_problems)}")
    return all_problems


# ── Utility ──────────────────────────────────────────────────────────────────

def get_subset_names() -> list[str]:
    config = load_bbh_config()
    return [s["name"] for s in config["subsets"]]


def get_subset_display_map() -> dict[str, str]:
    config = load_bbh_config()
    return {s["name"]: s["display_name"] for s in config["subsets"]}


# ── Smoke Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    problems = load_bbh_problems(n_per_subset=2)  # small sample for testing
    for p in problems:
        print(f"\n{'─'*60}")
        print(f"ID:     {p['id']}")
        print(f"Subset: {p['subset_display']}")
        print(f"Input:  {p['input'][:120]}...")
        print(f"Target: {p['target']}")
