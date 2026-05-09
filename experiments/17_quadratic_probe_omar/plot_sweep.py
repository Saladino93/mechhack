"""Plot Pleshkov 13-layer sweep — AUC vs layer for linear / linear-on-PCs / quad.

Reads results/<task>_d<d>_sweep.json and writes results/<task>_d<d>_sweep.png.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--d_pca", type=int, required=True)
    args = ap.parse_args()

    p = RESULTS / f"{args.task}_d{args.d_pca}_sweep.json"
    d = json.loads(p.read_text())
    layers = [int(l) for l in d["layer_idxs"] if str(l) in d["by_layer"]]

    def col(k):
        return [d["by_layer"][str(l)][k]["auc_mean"] for l in layers]
    def err(k):
        return [d["by_layer"][str(l)][k]["auc_std"] for l in layers]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(layers, col("linear"),        yerr=err("linear"),
                marker="o", label="linear (raw 5376-d)", capsize=3)
    ax.errorbar(layers, col("linear_on_pcs"), yerr=err("linear_on_pcs"),
                marker="s", label=f"linear on {args.d_pca} PCs", capsize=3)
    ax.errorbar(layers, col("quadratic"),     yerr=err("quadratic"),
                marker="^", label=f"quadratic (PCA d={args.d_pca})", capsize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUC (5-fold CV mean ± std)")
    ax.set_ylim(0.5, 1.0)
    ax.set_title(f"{args.task}: Pleshkov sweep (d_pca={args.d_pca}, "
                 f"n={d['n_samples']}, n_quad_features={d['n_quadratic_features']})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = RESULTS / f"{args.task}_d{args.d_pca}_sweep.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
