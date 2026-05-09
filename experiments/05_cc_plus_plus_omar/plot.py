"""Plot probe-head comparison for experiment 05.

Reads results.json and swim_traces.npz produced by train_heads.py and writes:
  - head_comparison_auc.png
  - head_comparison_acc.png
  - swim_token_logits_example.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.json"
TRACES = HERE / "swim_traces.npz"


def gather(results):
    """Return ordered (label, mean, lo, hi, std) tuples for AUC and accuracy."""
    heads = results["heads"]

    def from_single(d, m):
        # sklearn heads have flat dict; std is None
        if m == "auc":
            return d["auc"], d.get("auc_ci_lo", d["auc"]), d.get("auc_ci_hi", d["auc"]), 0.0
        if m == "acc":
            return d["acc"], d["acc"], d["acc"], 0.0

    def from_summary(d, m):
        s = d["summary"]
        return (
            s[f"{m}_mean"],
            s[f"{m}_mean"] - s[f"{m}_std"],
            s[f"{m}_mean"] + s[f"{m}_std"],
            s[f"{m}_std"],
        )

    rows_auc = [
        ("D: baseline\n(LR mean@40)",            *from_single(heads["D_baseline"],          "auc")),
        ("A: concat\n(LR all-layers)",           *from_single(heads["A_concat"],            "auc")),
        ("B: SWiM\n(per-tok, mean BCE)",         *from_summary(heads["B_swim"],             "auc")),
        ("C: softw-BCE\n(SWiM agg)",             *from_summary(heads["C_softbce_swim_max"], "auc")),
        ("C': softw-BCE\n(max agg)",             *from_summary(heads["C_softbce_max_only"], "auc")),
    ]
    rows_acc = [
        ("D: baseline\n(LR mean@40)",            *from_single(heads["D_baseline"],          "acc")),
        ("A: concat\n(LR all-layers)",           *from_single(heads["A_concat"],            "acc")),
        ("B: SWiM\n(per-tok, mean BCE)",         *from_summary(heads["B_swim"],             "acc")),
        ("C: softw-BCE\n(SWiM agg)",             *from_summary(heads["C_softbce_swim_max"], "acc")),
        ("C': softw-BCE\n(max agg)",             *from_summary(heads["C_softbce_max_only"], "acc")),
    ]
    return rows_auc, rows_acc


def bar_plot(rows, title, ylabel, out_path):
    labels = [r[0] for r in rows]
    means = [r[1] for r in rows]
    los = [max(0, r[1] - r[2]) for r in rows]
    his = [max(0, r[3] - r[1]) for r in rows]
    stds = [r[4] for r in rows]
    colors = ["#888", "#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, means, yerr=[los, his], capsize=5, color=colors,
                  edgecolor="black", linewidth=0.5)
    for bar, m, s in zip(bars, means, stds):
        h = bar.get_height()
        ann = f"{m:.3f}"
        if s > 0:
            ann += f"\n±{s:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, ann,
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.5, 1.02)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.7)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"  wrote {out_path}")


def trace_plot(out_path):
    if not TRACES.exists():
        print("  swim_traces.npz missing -- skipping trace plot")
        return
    z = np.load(TRACES)
    keys = list(z.files)
    pairs = []
    for label_name, kind in [("dual_use (positive)", "pos"), ("benign (negative)", "neg")]:
        if f"{kind}_raw" not in keys:
            continue
        pairs.append((label_name, z[f"{kind}_raw"], z[f"{kind}_smooth"]))
    if not pairs:
        print("  no traces in npz -- skipping")
        return
    fig, axes = plt.subplots(len(pairs), 1, figsize=(10, 3 * len(pairs)),
                             sharex=False)
    if len(pairs) == 1:
        axes = [axes]
    for ax, (name, raw, smooth) in zip(axes, pairs):
        x = np.arange(len(raw))
        ax.plot(x, raw, color="#888", linewidth=0.7, alpha=0.7,
                label="raw per-token logit")
        ax.plot(x, smooth, color="#d62728", linewidth=1.6,
                label=f"SWiM (window=16) smoothed")
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.6)
        ax.set_title(name)
        ax.set_xlabel("token position")
        ax.set_ylabel("probe logit")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"  wrote {out_path}")


def main():
    results = json.loads(RESULTS.read_text())
    rows_auc, rows_acc = gather(results)
    bar_plot(rows_auc, "Probe-head AUC on cyber_1 (test)",
             "ROC-AUC (test)", HERE / "head_comparison_auc.png")
    bar_plot(rows_acc, "Probe-head accuracy on cyber_1 (test)",
             "Accuracy (test)", HERE / "head_comparison_acc.png")
    trace_plot(HERE / "swim_token_logits_example.png")


if __name__ == "__main__":
    main()
