"""Plot CV vs held-out test AUC/accuracy across layers and poolings.

Three plots:
  - auc_vs_layer.png  — per pooling: CV mean ±σ (band), held-out test AUC,
                        train AUC (dashed) for overfitting diagnostic.
  - acc_vs_layer.png  — same, for accuracy.
  - metrics_vs_layer.png — both side by side.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RESULTS = HERE / "results.json"
EXP03_RESULTS = HERE.parent / "03_layer_sweep_omar" / "results.json"

POOLING_STYLE = {
    "mean":       {"label": "mean-pool",  "color": "tab:blue",   "marker": "o"},
    "last_token": {"label": "last-token", "color": "tab:orange", "marker": "s"},
}


def _overlay_exp03(ax, metric_key="auc"):
    """Overlay exp 03's 999-sample CV curve as dotted lines for comparison."""
    if not EXP03_RESULTS.exists():
        return
    try:
        d03 = json.loads(EXP03_RESULTS.read_text())
        for pname, rows in d03["tasks"]["cyber_1"]["per_pooling"].items():
            style = POOLING_STYLE.get(pname, {})
            color = style.get("color")
            layers = [r["layer"] for r in rows]
            if metric_key == "auc":
                vals = [r["auc_mean"] for r in rows]
            else:
                vals = [r["acc_mean"] for r in rows]
            ax.plot(layers, vals, ls=":", lw=1.4, color=color, alpha=0.85,
                    marker=style.get("marker", "o"), markersize=4,
                    markerfacecolor="white", markeredgewidth=1,
                    label=f"{style.get('label', pname)} (exp 03 N=999 CV)")
    except Exception as e:
        print(f"  [warn] could not overlay exp 03: {e}")


def _plot_metric(ax, data, *, cv_mean_key, cv_std_key, ho_key,
                 ylabel, title, train_key=None, chance_line=None, ymin=0.4):
    plotted = False
    cv = data["cv"]["per_pooling"]
    ho = data["held_out_test"]["per_pooling"]
    for pname, cv_rows in cv.items():
        ho_rows = ho.get(pname, [])
        style = POOLING_STYLE.get(pname, {})
        layers = [r["layer"] for r in cv_rows]
        cv_means = [r[cv_mean_key] for r in cv_rows]
        cv_stds  = [r[cv_std_key]  for r in cv_rows]
        # CV line + band
        line = ax.plot(layers, cv_means,
                       marker=style.get("marker", "o"),
                       color=style.get("color"),
                       label=f"{style.get('label', pname)} (CV mean ±1σ)")[0]
        ax.fill_between(layers,
                        [m - s for m, s in zip(cv_means, cv_stds)],
                        [m + s for m, s in zip(cv_means, cv_stds)],
                        alpha=0.2, color=line.get_color())
        # Held-out test line (solid, same color, different marker)
        ho_layers = [r["layer"] for r in ho_rows]
        ho_vals = [r[ho_key] for r in ho_rows]
        ax.plot(ho_layers, ho_vals,
                marker=style.get("marker", "o"),
                color=line.get_color(), ls="-", lw=2.2, alpha=0.95,
                markerfacecolor="none", markeredgewidth=1.5,
                label=f"{style.get('label', pname)} (held-out test)")
        # Train (dashed, faint)
        if train_key is not None:
            t_vals = [r[train_key] for r in cv_rows]
            ax.plot(layers, t_vals, ls="--", lw=1, color=line.get_color(), alpha=0.6,
                    label=f"{style.get('label', pname)} (train)")
        plotted = True
    if chance_line is not None:
        ax.axhline(chance_line, color="grey", ls="--", lw=0.8, label="chance")
    ax.set_xlabel("Residual-stream layer index (0 = embedding)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(ymin, 1.02)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="lower right", fontsize=7)


def main():
    if not RESULTS.exists():
        raise SystemExit(f"{RESULTS} not found. Run train_probes.py first.")
    data = json.loads(RESULTS.read_text())

    n_train = data["n_train"]
    n_test = data["n_test"]
    n_folds = data.get("n_folds", "?")

    # Chance for accuracy = majority-class fraction on the train set
    pos = data["train_label_counts"]["pos"]
    neg = data["train_label_counts"]["neg"]
    chance_acc_train = max(pos, neg) / max(pos + neg, 1)
    pos_t = data["test_label_counts"]["pos"]
    neg_t = data["test_label_counts"]["neg"]
    chance_acc_test = max(pos_t, neg_t) / max(pos_t + neg_t, 1)
    chance_acc = max(chance_acc_train, chance_acc_test)

    common_subtitle = (f"Gemma 4-31B-it cyber_1 | "
                       f"train n={n_train} ({pos}+/{neg}-), test n={n_test} "
                       f"({pos_t}+/{neg_t}-) | {n_folds}-fold CV ±1σ band")

    # AUC plot
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_metric(ax, data,
                 cv_mean_key="auc_mean", cv_std_key="auc_std", ho_key="auc",
                 ylabel="ROC-AUC",
                 title=f"AUC vs layer\n{common_subtitle}",
                 train_key="train_auc_mean", chance_line=0.5)
    _overlay_exp03(ax, metric_key="auc")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "auc_vs_layer.png", dpi=150); plt.close(fig)
    print(f"Wrote {HERE / 'auc_vs_layer.png'}")

    # Accuracy plot
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_metric(ax, data,
                 cv_mean_key="acc_mean", cv_std_key="acc_std", ho_key="acc",
                 ylabel="Accuracy",
                 title=f"Accuracy vs layer\n{common_subtitle}",
                 train_key="train_acc_mean", chance_line=chance_acc)
    _overlay_exp03(ax, metric_key="acc")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "acc_vs_layer.png", dpi=150); plt.close(fig)
    print(f"Wrote {HERE / 'acc_vs_layer.png'}")

    # Combined two-panel
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    _plot_metric(ax1, data,
                 cv_mean_key="auc_mean", cv_std_key="auc_std", ho_key="auc",
                 ylabel="ROC-AUC", title="AUC vs layer",
                 train_key="train_auc_mean", chance_line=0.5)
    _overlay_exp03(ax1, metric_key="auc")
    ax1.legend(loc="lower right", fontsize=7)
    _plot_metric(ax2, data,
                 cv_mean_key="acc_mean", cv_std_key="acc_std", ho_key="acc",
                 ylabel="Accuracy", title="Accuracy vs layer",
                 train_key="train_acc_mean", chance_line=chance_acc)
    _overlay_exp03(ax2, metric_key="acc")
    ax2.legend(loc="lower right", fontsize=7)
    fig.suptitle(common_subtitle)
    fig.tight_layout()
    fig.savefig(HERE / "metrics_vs_layer.png", dpi=150); plt.close(fig)
    print(f"Wrote {HERE / 'metrics_vs_layer.png'}")


if __name__ == "__main__":
    main()
