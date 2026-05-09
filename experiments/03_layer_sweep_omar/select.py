"""Selects a 50/50 balanced cyber_1 train subset and writes it to selection.json.

500 random dual_use + 500 random benign, seeded for reproducibility.
Both extract.py and train_probes.py read this file so they stay in sync.
"""
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments" / "02_extract_activations"))
from data import load_dataset

N_PER_CLASS = 500
SEED = 42
HERE = Path(__file__).parent
OUT = HERE / "selection.json"


def main():
    samples = load_dataset("cyber", split="train")
    by_cls = {"dual_use": [s for s in samples if s["label"] == "dual_use"],
              "benign":   [s for s in samples if s["label"] == "benign"]}
    print(f"pool: dual_use={len(by_cls['dual_use'])}, benign={len(by_cls['benign'])}")

    rng = random.Random(SEED)
    chosen = []
    for cls in ("dual_use", "benign"):
        pool = by_cls[cls]
        if len(pool) < N_PER_CLASS:
            raise SystemExit(f"only {len(pool)} {cls} samples available, need {N_PER_CLASS}")
        picked = rng.sample(pool, N_PER_CLASS)
        chosen.extend({"sample_id": s["sample_id"], "label": s["label"]} for s in picked)

    rng.shuffle(chosen)  # interleave classes
    OUT.write_text(json.dumps({
        "n_per_class": N_PER_CLASS,
        "seed": SEED,
        "task": "cyber_1",
        "samples": chosen,
    }, indent=2))
    print(f"wrote {OUT} with {len(chosen)} samples ({N_PER_CLASS} per class)")


if __name__ == "__main__":
    main()
