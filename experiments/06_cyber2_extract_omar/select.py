"""Select a 50/50 balanced cyber_2 train subset and write to selection.json.

cyber_2 = high_risk_dual_use (positive) vs {dual_use, benign} (negative).

We pick 500 random high_risk_dual_use as positives (all are NEW extracts, since
exp 03 only covered dual_use+benign).

For negatives we draw 500 from the union {dual_use, benign}, but PREFER
sample_ids that already have extracts in /home/ubuntu/extracts/cyber_all_omar/
(symlinks to /home/ubuntu/extracts/03_layer_sweep_omar/). This saves ~5-15 min
of extraction time.

Strategy for negatives:
  - 1406 dual_use+benign already extracted (685 dual_use + 721 benign).
  - We do a RNG shuffle of the union, then pick the first 500 that are already
    extracted. This is enough — we only need 500 and we have 1406 available.
  - This biases the negative pool toward exp 03's selection, but exp 03's
    selection was itself rng.Random(42)-shuffled within {dual_use, benign},
    so the resulting negatives are still a random subsample of cyber-train's
    {dual_use, benign}. The proportion of dual_use vs benign in the chosen 500
    is documented below.

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
TASK = "cyber_2"


def main():
    samples = load_dataset("cyber", split="train")
    by_cls = {
        "high_risk_dual_use": [s for s in samples if s["label"] == "high_risk_dual_use"],
        "dual_use":           [s for s in samples if s["label"] == "dual_use"],
        "benign":             [s for s in samples if s["label"] == "benign"],
    }
    print(f"pool: high_risk_dual_use={len(by_cls['high_risk_dual_use'])}, "
          f"dual_use={len(by_cls['dual_use'])}, benign={len(by_cls['benign'])}")

    already_extracted = {f[:-3] for f in os.listdir(SHARED_EXTRACTS) if f.endswith(".pt")}
    print(f"already extracted in {SHARED_EXTRACTS}: {len(already_extracted)}")

    rng = random.Random(SEED)

    # Positives: 500 random high_risk_dual_use
    pos_pool = by_cls["high_risk_dual_use"]
    if len(pos_pool) < N_PER_CLASS:
        raise SystemExit(f"only {len(pos_pool)} high_risk_dual_use, need {N_PER_CLASS}")
    pos = rng.sample(pos_pool, N_PER_CLASS)

    # Negatives: shuffle union of {dual_use, benign}, prefer already-extracted IDs
    neg_pool = by_cls["dual_use"] + by_cls["benign"]
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
        "positive_label": "high_risk_dual_use",
        "negative_labels": ["dual_use", "benign"],
        "negative_label_counts": dict(neg_label_counts),
        "samples": chosen,
    }, indent=2))
    print(f"wrote {OUT} with {len(chosen)} samples ({N_PER_CLASS} per class)")


if __name__ == "__main__":
    main()
