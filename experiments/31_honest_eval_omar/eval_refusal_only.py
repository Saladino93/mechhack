"""Refusal-only honest eval (parallel companion to eval_test_split.py).

Runs the same multi-layer probe set as eval_test_split.py but only on
refusal_gemma. Refusal extracts are in a separate directory from cyber, so
this doesn't compete with Q4's disk reads on cyber_all_omar/.

Output: refusal_only_results.json + refusal_only_table.md
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from eval_test_split import load_refusal_features, eval_task, LAYERS  # noqa


def main():
    print("[refusal-only honest eval]", flush=True)
    print("loading refusal train+test...", flush=True)
    t0 = time.time()
    X_tr_mean, X_tr_last, y_tr, _ = load_refusal_features("train")
    X_te_mean, X_te_last, y_te, _ = load_refusal_features("test")
    print(f"  train={X_tr_mean.shape} test={X_te_mean.shape} ({time.time()-t0:.0f}s)", flush=True)

    out = {"refusal_gemma": eval_task("refusal_gemma", X_tr_mean, X_tr_last, y_tr,
                                       X_te_mean, X_te_last, y_te)}
    (HERE / "refusal_only_results.json").write_text(json.dumps(out, indent=2))

    # Compact table
    md = ["# Refusal-only honest eval (test-split AUC, sorted desc)\n"]
    rows = sorted([(v["test_auc"], k, v) for k, v in out["refusal_gemma"].items()], reverse=True)
    md.append("| Probe | train | **test** |")
    md.append("|---|---:|---:|")
    for _, k, v in rows[:25]:
        md.append(f"| {k} | {v['train_auc']:.4f} | **{v['test_auc']:.4f}** |")
    (HERE / "refusal_only_table.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {HERE/'refusal_only_results.json'} and refusal_only_table.md", flush=True)
    print(f"\nTOP 8 probes on refusal test set:")
    for _, k, v in rows[:8]:
        print(f"  {k:>22}  test_AUC={v['test_auc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
