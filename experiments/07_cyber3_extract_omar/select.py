"""Select a 50/50 balanced cyber_3 train subset and write to selection.json.

cyber_3 = prohibited (positive) vs {high_risk_dual_use, dual_use, benign} (negative).

We pick 500 random prohibited as positives (all are NEW extracts).

For negatives we draw 500 from the union {high_risk_dual_use, dual_use, benign},
preferring sample_ids that already have extracts in /home/ubuntu/extracts/cyber_all_omar/.
This dir is populated with:
  - 1406 dual_use+benign extracts from exp 03 (symlinks)
  - 500 high_risk_dual_use extracts from exp 06 (after cyber_2 runs)
So if exp 06 has finished, we can pull from all 3 negative classes "for free".

Strategy for negatives:
  - Shuffle the union with rng.Random(42).
  - Take the first 500 that are already extracted.
  - If exp 06 has not yet run, this list will be entirely dual_use+benign and
    will *not* contain any high_risk_dual_use. Re-run select.py after exp 06
    finishes to get a 3-class negative mix. (Documented in notes.md.)

Seed: random.Random(42).
"""
import json
import os
import random
import sys
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments" / "02_extract_activations"))
from data import load_dataset

N_PER_CLASS = 500
SEED = 42
HERE = Path(__file__).parent
OUT = HERE / "selection.json"
SHARED_EXTRACTS = Path("/home/ubuntu/extracts/cyber_all_omar")
TASK = "cyber_3"


def main():
    samples = load_dataset("cyber", split="train")
    by_cls = {
        "prohibited":         [s for s in samples if s["label"] == "prohibited"],
        "high_risk_dual_use": [s for s in samples if s["label"] == "high_risk_dual_use"],
        "dual_use":           [s for s in samples if s["label"] == "dual_use"],
        "benign":             [s for s in samples if s["label"] == "benign"],
    }
    print(f"pool: prohibited={len(by_cls['prohibited'])}, "
          f"high_risk_dual_use={len(by_cls['high_risk_dual_use'])}, "
          f"dual_use={len(by_cls['dual_use'])}, benign={len(by_cls['benign'])}")

    already_extracted = {f[:-3] for f in os.listdir(SHARED_EXTRACTS) if f.endswith(".pt")}
    print(f"already extracted in {SHARED_EXTRACTS}: {len(already_extracted)}")

    rng = random.Random(SEED)

    # Positives: 500 random prohibited
    pos_pool = by_cls["prohibited"]
    if len(pos_pool) < N_PER_CLASS:
        raise SystemExit(f"only {len(pos_pool)} prohibited, need {N_PER_CLASS}")
    pos = rng.sample(pos_pool, N_PER_CLASS)

    # Negatives: shuffle union of all 3 negative classes, prefer already-extracted
    neg_pool = by_cls["high_risk_dual_use"] + by_cls["dual_use"] + by_cls["benign"]
    rng.shuffle(neg_pool)
    extracted_neg = [s for s in neg_pool if s["sample_id"] in already_extracted]
    fresh_neg     = [s for s in neg_pool if s["sample_id"] not in already_extracted]
    print(f"negative pool: {len(extracted_neg)} already-extracted, {len(fresh_neg)} fresh")
    if len(extracted_neg) >= N_PER_CLASS:
        neg = extracted_neg[:N_PER_CLASS]
        print(f"-> using {N_PER_CLASS} already-extracted negatives, 0 fresh extractions")
    else:
        neg = extracted_neg + fresh_neg[: N_PER_CLASS - len(extracted_neg)]
        print(f"-> using {len(extracted_neg)} already-extracted + "
              f"{N_PER_CLASS - len(extracted_neg)} fresh negatives")
    neg_label_counts = Counter(s["label"] for s in neg)
    print(f"negative label breakdown: {dict(neg_label_counts)}")

    chosen = (
        [{"sample_id": s["sample_id"], "label": s["label"]} for s in pos]
      + [{"sample_id": s["sample_id"], "label": s["label"]} for s in neg]
    )
    rng.shuffle(chosen)

    OUT.write_text(json.dumps({
        "n_per_class": N_PER_CLASS,
        "seed": SEED,
        "task": TASK,
        "positive_label": "prohibited",
        "negative_labels": ["high_risk_dual_use", "dual_use", "benign"],
        "negative_label_counts": dict(neg_label_counts),
        "samples": chosen,
    }, indent=2))
    print(f"wrote {OUT} with {len(chosen)} samples ({N_PER_CLASS} per class)")


if __name__ == "__main__":
    main()
