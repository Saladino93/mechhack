"""Dataset loading utilities for activation extraction.

Handles both cyber probes and refusal probes with a uniform interface.
Each sample is returned as a dict with at least:
  - sample_id: str
  - prompt: str (the text to feed to the model)
  - label: int (task-dependent)
  - split: str ("train" or "test")
  - metadata: dict (everything else)
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS = REPO_ROOT / "datasets"

# ── Dataset registry ──────────────────────────────────────────

DATASET_CONFIGS = {
    "cyber": {
        "path": DATASETS / "cyber_probes",
        "files": {"train": "train.jsonl", "test": "test.jsonl"},
        "prompt_field": "prompt",
        "label_field": "category",
        "description": "7259 cyber-harm rollouts, 4 categories",
    },
    "refusal_gemma4_31b": {
        "path": DATASETS / "refusal_probes" / "gemma4_31b",
        "files": {"full": "attacks_full.jsonl", "train": "train_split.jsonl",
                  "test": "test_split.jsonl", "eval": "attribution_eval.jsonl"},
        "prompt_field": "attack_prompt",
        "label_field": "is_refusal",
        "description": "878 Gemma refusal rollouts",
    },
    "refusal_qwen36": {
        "path": DATASETS / "refusal_probes" / "qwen36",
        "files": {"full": "attacks_full.jsonl", "train": "train_split.jsonl",
                  "test": "test_split.jsonl", "eval": "attribution_eval.jsonl"},
        "prompt_field": "attack_prompt",
        "label_field": "is_refusal",
        "description": "878 Qwen refusal rollouts",
    },
}


def list_datasets():
    """Return available dataset names and descriptions."""
    return {k: v["description"] for k, v in DATASET_CONFIGS.items()}


def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_dataset(name, split="full"):
    """Load a dataset by name and split.

    Args:
        name: One of 'cyber', 'refusal_gemma4_31b', 'refusal_qwen36'
        split: 'train', 'test', 'full', or 'eval' (refusal only)
               For cyber, 'full' = train + test combined.

    Returns:
        List of dicts with unified schema:
        {sample_id, prompt, label, split, metadata}
    """
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_CONFIGS)}")

    cfg = DATASET_CONFIGS[name]
    prompt_field = cfg["prompt_field"]
    label_field = cfg["label_field"]

    # Load raw data
    if split == "full":
        if "full" in cfg["files"]:
            raw = load_jsonl(cfg["path"] / cfg["files"]["full"])
        else:
            # Combine train + test
            raw = (load_jsonl(cfg["path"] / cfg["files"]["train"]) +
                   load_jsonl(cfg["path"] / cfg["files"]["test"]))
    elif split in cfg["files"]:
        raw = load_jsonl(cfg["path"] / cfg["files"][split])
    else:
        raise ValueError(f"Split '{split}' not available for '{name}'. "
                         f"Available: {list(cfg['files'])}")

    # Normalize to uniform schema
    samples = []
    for row in raw:
        sample = {
            "sample_id": row["sample_id"],
            "prompt": row[prompt_field],
            "label": row[label_field],
            "split": row.get("split", split),
            "metadata": {k: v for k, v in row.items()
                         if k not in (prompt_field, "sample_id", "split")},
        }
        samples.append(sample)

    return samples


def get_label_for_task(sample, task):
    """Convert a sample's raw label to a binary int for a specific probe task.

    Args:
        sample: Dict with 'label' field (category str or bool)
        task: One of 'cyber_1', 'cyber_2', 'cyber_3',
              'refusal_gemma4_31b', 'refusal_qwen36'

    Returns:
        1 (positive), 0 (negative), or None (not in this task's scope)
    """
    label = sample["label"]

    if task == "cyber_1":
        if label == "dual_use": return 1
        if label == "benign": return 0
        return None
    elif task == "cyber_2":
        if label == "high_risk_dual_use": return 1
        if label in ("dual_use", "benign"): return 0
        return None
    elif task == "cyber_3":
        if label == "prohibited": return 1
        if label in ("high_risk_dual_use", "dual_use", "benign"): return 0
        return None
    elif task.startswith("refusal"):
        return int(bool(label))
    else:
        raise ValueError(f"Unknown task '{task}'")


# ── Convenience ───────────────────────────────────────────────

def get_tasks_for_dataset(name):
    """Return the probe task names relevant to a dataset."""
    if name == "cyber":
        return ["cyber_1", "cyber_2", "cyber_3"]
    elif name == "refusal_gemma4_31b":
        return ["refusal_gemma4_31b"]
    elif name == "refusal_qwen36":
        return ["refusal_qwen36"]
    else:
        raise ValueError(f"Unknown dataset '{name}'")
