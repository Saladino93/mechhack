"""Select ALL 3190 cyber_1-eligible samples (train + test) and write to selection.json.

This is a scale-up of experiment 03 — exp 03 used a balanced 999-sample
sub-selection of cyber-train and reported test AUC ~0.98 across 5 folds.
The user's working hypothesis is that ~0.98 may be hiding overfitting; this
experiment runs the same probe on the FULL cyber_1 pool and includes a proper
held-out evaluation on the official cyber-test split.

Pool sizes (from `data.load_dataset('cyber', split=...)`):
  train: 685 dual_use + 1582 benign = 2267
  test:  267 dual_use +  656 benign =  923
  total: 952 dual_use + 2238 benign = 3190

Each entry stores: sample_id, label (raw category string), split.

OOM handling: we keep ALL 3190 IDs in selection.json. extract.py wraps each
forward pass in try/except — samples that OOM (e.g. cyber_6469's 25K-char
binary blob in exp 03) are logged and skipped, leaving no .pt on disk.
train_probes.py automatically excludes any selection-id whose .pt is missing.

We do flag prompts >20000 chars as "long" in selection.json so the OOM list
is predictable, but we still attempt to extract them (most go through fine).

Output is sorted by sample_id for reproducibility (no rng involved here —
we want every cyber_1-eligible sample).
"""
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments" / "02_extract_activations"))
from data import load_dataset

HERE = Path(__file__).parent
OUT = HERE / "selection.json"
LONG_PROMPT_CHARS = 20000  # warn-threshold; samples are still attempted


def main():
    train = load_dataset("cyber", split="train")
    test = load_dataset("cyber", split="test")

    chosen = []
    long_prompts = []
    for split_name, samples in [("train", train), ("test", test)]:
        for s in samples:
            if s["label"] not in ("dual_use", "benign"):
                continue
            row = {"sample_id": s["sample_id"],
                   "label": s["label"],
                   "split": split_name}
            chosen.append(row)
            if len(s["prompt"]) > LONG_PROMPT_CHARS:
                long_prompts.append({"sample_id": s["sample_id"],
                                     "label": s["label"],
                                     "split": split_name,
                                     "prompt_chars": len(s["prompt"])})

    # Sort by sample_id for reproducibility
    chosen.sort(key=lambda r: r["sample_id"])

    by_split = Counter(r["split"] for r in chosen)
    by_label = Counter(r["label"] for r in chosen)
    by_split_label = Counter((r["split"], r["label"]) for r in chosen)

    print(f"Pool sizes (cyber_1):")
    for sp in ("train", "test"):
        n_du = by_split_label[(sp, "dual_use")]
        n_be = by_split_label[(sp, "benign")]
        print(f"  {sp}: dual_use={n_du}, benign={n_be}, total={n_du+n_be}")
    print(f"  TOTAL: {len(chosen)}")
    print(f"  Long prompts (>{LONG_PROMPT_CHARS} chars, may OOM): {len(long_prompts)}")

    OUT.write_text(json.dumps({
        "task": "cyber_1",
        "long_prompt_chars": LONG_PROMPT_CHARS,
        "by_split": dict(by_split),
        "by_label": dict(by_label),
        "by_split_label": {f"{sp}/{lbl}": n
                           for (sp, lbl), n in by_split_label.items()},
        "n_long_prompts": len(long_prompts),
        "long_prompts": long_prompts,
        "samples": chosen,
    }, indent=2))
    print(f"wrote {OUT} with {len(chosen)} samples")


if __name__ == "__main__":
    main()
