"""Length-stratified AUC + Kramár FNR-at-strict-FPR.

Tertile-bins the 281 test prompts by char length of `attack_prompt`.
For each per-layer probe in `test_predictions.npz`, computes:
  - bucket-wise AUC + 1000-bootstrap 95% CI
  - bucket-wise FNR @ {1, 5}% FPR using a threshold *locked on the short
    bucket* (matches exp 15: this is the threshold a production guardrail
    would deploy)

Outputs:
  - length_strat_results.json
  - length_strat_auc.png
  - length_strat_fnr.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

OUT_DIR = Path(__file__).parent
SAMPLES_FILE = (OUT_DIR.parent.parent /
                "datasets/refusal_probes/qwen36/attacks_full.jsonl")

# Load test predictions
preds = np.load(OUT_DIR / "test_predictions.npz")
meta = json.loads((OUT_DIR / "test_predictions_meta.json").read_text())
test_ids = meta["test_sample_ids"]
y_test = np.array(meta["y_test"], dtype=int)

# Load full samples for char length
all_samples = {json.loads(l)["sample_id"]: json.loads(l)
               for l in open(SAMPLES_FILE)}
lens = np.array([len(all_samples[sid]["attack_prompt"]) for sid in test_ids])

# Tertile bins: short / medium / long
q33, q66 = np.percentile(lens, [33.3333, 66.6666])
bucket = np.where(lens <= q33, "short",
            np.where(lens <= q66, "medium", "long"))
print(f"Length tertiles: short ≤{int(q33)}, medium ≤{int(q66)}, long >{int(q66)}")
for b in ["short", "medium", "long"]:
    m = bucket == b
    print(f"  {b:6s}: n={m.sum()}, n_pos={int(y_test[m].sum())} "
          f"({100*y_test[m].mean():.1f}%), len p50={int(np.median(lens[m]))}")


def auc_with_ci(y_true, y_score, n_boot=1000, seed=0):
    auc = float(roc_auc_score(y_true, y_score))
    rng = np.random.default_rng(seed)
    n = len(y_true); aucs = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, size=n)
        if len(set(y_true[ix].tolist())) < 2: continue
        aucs.append(roc_auc_score(y_true[ix], y_score[ix]))
    return auc, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def fnr_at_fpr(y_true, y_score, target_fpr):
    """Threshold chosen so that FPR ≤ target_fpr; returns (threshold, fnr)."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    # find smallest threshold that achieves FPR <= target
    ok = fpr <= target_fpr
    if not ok.any():
        return float("inf"), 1.0
    t_idx = np.where(ok)[0][-1]   # last index satisfying constraint
    return float(thr[t_idx]), float(1.0 - tpr[t_idx])


def fnr_at_threshold(y_true, y_score, threshold):
    pos = y_true == 1
    if pos.sum() == 0: return float("nan"), 0
    pred_pos = y_score >= threshold
    tn = pred_pos[pos].sum()  # true positives at threshold
    return float(1.0 - tn / pos.sum()), int(pos.sum())


# Run the analysis for each (arch, layer)
records = []
arch_layers = [k for k in preds.keys()]
for k in arch_layers:
    arch_L = k  # e.g. "linear_sklearn_last_L60"
    arch, _, L = arch_L.rpartition("_L")
    L = int(L)
    p = preds[k]
    rec = {"key": arch_L, "arch": arch, "layer": L, "buckets": {}}
    # Lock thresholds on the SHORT bucket
    short_m = bucket == "short"
    thr_1pct, fnr_short_1 = fnr_at_fpr(y_test[short_m], p[short_m], 0.01)
    thr_5pct, fnr_short_5 = fnr_at_fpr(y_test[short_m], p[short_m], 0.05)
    rec["short_thr_1pct"] = thr_1pct; rec["short_thr_5pct"] = thr_5pct
    for b in ["short", "medium", "long"]:
        m = bucket == b
        if m.sum() < 5 or len(set(y_test[m].tolist())) < 2:
            continue
        a, lo, hi = auc_with_ci(y_test[m], p[m], seed=L)
        f1, n_pos = fnr_at_threshold(y_test[m], p[m], thr_1pct)
        f5, _    = fnr_at_threshold(y_test[m], p[m], thr_5pct)
        rec["buckets"][b] = {"n": int(m.sum()), "n_pos": n_pos,
                              "auc": a, "ci95": [lo, hi],
                              "fnr_1pct": f1, "fnr_5pct": f5}
    records.append(rec)

# Save
out_json = {"q33": float(q33), "q66": float(q66),
            "bucket_counts": {b: int((bucket==b).sum()) for b in ["short","medium","long"]},
            "records": records}
(OUT_DIR / "length_strat_results.json").write_text(json.dumps(out_json, indent=2))


# Plot — focus on the winner (highest test AUC overall)
def overall_auc(rec):
    aucs = [b["auc"] for b in rec["buckets"].values()]
    return float(np.mean(aucs)) if aucs else 0.0

records.sort(key=overall_auc, reverse=True)
TOP = records[:6]
print(f"\nTop 6 (arch, L, mean-bucket AUC):")
for r in TOP:
    print(f"  {r['key']}: {overall_auc(r):.4f}")

# Plot AUC vs bucket for top 6
fig, ax = plt.subplots(figsize=(8, 5))
buckets = ["short", "medium", "long"]
COLORS = plt.cm.tab10(np.linspace(0, 1, len(TOP)))
for r, c in zip(TOP, COLORS):
    aucs, los, his = [], [], []
    for b in buckets:
        bd = r["buckets"].get(b, {})
        aucs.append(bd.get("auc", float("nan")))
        los.append(bd.get("ci95", [float("nan"), float("nan")])[0])
        his.append(bd.get("ci95", [float("nan"), float("nan")])[1])
    aucs = np.array(aucs); los = np.array(los); his = np.array(his)
    ax.errorbar(range(3), aucs, yerr=[aucs-los, his-aucs], capsize=3,
                marker="o", label=r["key"], color=c)
ax.set_xticks(range(3))
ax.set_xticklabels([f"{b}\n(n={int((bucket==b).sum())})" for b in buckets])
ax.set_ylabel("Test AUC")
ax.set_title(f"Qwen Refusal — AUC by prompt-length tertile")
ax.set_ylim(0.5, 1.0)
ax.axhline(0.5, color="black", lw=0.5, alpha=0.4)
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "length_strat_auc.png", dpi=140)
print(f"Saved {OUT_DIR / 'length_strat_auc.png'}")

# Plot FNR @ 1% FPR vs bucket for top 6
fig, ax = plt.subplots(figsize=(8, 5))
for r, c in zip(TOP, COLORS):
    fnrs = [r["buckets"].get(b, {}).get("fnr_1pct", float("nan"))
            for b in buckets]
    ax.plot(range(3), fnrs, marker="o", label=r["key"], color=c)
ax.set_xticks(range(3))
ax.set_xticklabels([f"{b}\n(n_pos={r['buckets'].get(b, {}).get('n_pos', '?')})"
                    for b in buckets])
ax.set_ylabel("FNR @ 1% FPR (lower better)")
ax.set_title(f"Qwen Refusal — Kramár FNR (threshold locked on SHORT)")
ax.set_ylim(0.0, 1.05)
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "length_strat_fnr.png", dpi=140)
print(f"Saved {OUT_DIR / 'length_strat_fnr.png'}")
