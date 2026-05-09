"""Plot 5-fold CV AUC and accuracy vs layer for each pooling.

Each metric is mean ± 1σ across 5 stratified folds. Train AUC is overlaid as a
faint dashed line so the train-vs-test gap (overfitting indicator) is visible.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RESULTS = HERE / "results.json"

POOLING_STYLE = {
    "mean":       {"label": "mean-pool",  "color": "tab:blue",   "marker": "o"},
    "last_token": {"label": "last-token", "color": "tab:orange", "marker": "s"},
}


def _plot_metric(ax, data, mean_key, std_key, ylabel, title,
                  train_key=None, chance_line=None, ymin=0.4):
    plotted = False
    for task, t in data["tasks"].items():
        if t.get("status") != "ok":
            continue
        for pname, rows in (t.get("per_pooling") or {}).items():
            style = POOLING_STYLE.get(pname, {})
            layers = [r["layer"] for r in rows]
            ys = [r[mean_key] for r in rows]
            ss = [r[std_key]  for r in rows]
            line = ax.plot(layers, ys,
                           marker=style.get("marker", "o"),
                           color=style.get("color"),
                           label=f"{task} · {style.get('label', pname)} (test)")[0]
            ax.fill_between(layers,
                            [y - s for y, s in zip(ys, ss)],
                            [y + s for y, s in zip(ys, ss)],
                            alpha=0.2, color=line.get_color())
            if train_key is not None:
                ts = [r[train_key] for r in rows]
                ax.plot(layers, ts, ls="--", lw=1, color=line.get_color(), alpha=0.7,
                        label=f"{task} · {style.get('label', pname)} (train)")
            plotted = True
    if chance_line is not None:
        ax.axhline(chance_line, color="grey", ls="--", lw=0.8, label="chance")
    ax.set_xlabel("Residual-stream layer index (0 = embedding)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(ymin, 1.02)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="lower right", fontsize=8)


def main():
    if not RESULTS.exists():
        raise SystemExit(f"{RESULTS} not found. Run train_probes.py first.")
    data = json.loads(RESULTS.read_text())
    n_samples = data["n_samples"]
    n_folds = next((r["n_folds"] for t in data["tasks"].values() if t.get("status") == "ok"
                    for rows in t["per_pooling"].values() for r in rows
                    if "n_folds" in r), "?")

    task_names = [name for name, t in data["tasks"].items() if t.get("status") == "ok"]
    task_label = ", ".join(task_names) if task_names else "?"
    primary = data["tasks"].get(task_names[0], {}) if task_names else {}
    if primary.get("status") == "ok":
        chance_acc = max(primary["n_pos"], primary["n_neg"]) / (primary["n_pos"] + primary["n_neg"])
    else:
        chance_acc = 0.5

    common_subtitle = (f"balanced {task_label} ({n_samples} samples), "
                       f"{n_folds}-fold stratified CV — band = ±1σ across folds")

    # AUC plot (with train overlay)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _plot_metric(ax, data, "auc_mean", "auc_std",
                 "ROC-AUC", f"Gemma 4-31B-it {task_label} probe\n{common_subtitle}",
                 train_key="train_auc_mean", chance_line=0.5)
    fig.tight_layout()
    fig.savefig(HERE / "auc_vs_layer.png", dpi=150); plt.close(fig)
    print(f"Wrote {HERE / 'auc_vs_layer.png'}")

    # Accuracy plot (with train overlay)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _plot_metric(ax, data, "acc_mean", "acc_std",
                 "Accuracy", f"Gemma 4-31B-it {task_label} probe\n{common_subtitle}",
                 train_key="train_acc_mean", chance_line=chance_acc)
    fig.tight_layout()
    fig.savefig(HERE / "acc_vs_layer.png", dpi=150); plt.close(fig)
    print(f"Wrote {HERE / 'acc_vs_layer.png'}")

    # Combined two-panel
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)
    _plot_metric(ax1, data, "auc_mean", "auc_std",
                 "ROC-AUC", "AUC vs layer",
                 train_key="train_auc_mean", chance_line=0.5)
    _plot_metric(ax2, data, "acc_mean", "acc_std",
                 "Accuracy", "Accuracy vs layer",
                 train_key="train_acc_mean", chance_line=chance_acc)
    fig.suptitle(f"Gemma 4-31B-it {task_label} layer sweep — {common_subtitle}")
    fig.tight_layout()
    fig.savefig(HERE / "metrics_vs_layer.png", dpi=150); plt.close(fig)
    print(f"Wrote {HERE / 'metrics_vs_layer.png'}")


if __name__ == "__main__":
    main()
