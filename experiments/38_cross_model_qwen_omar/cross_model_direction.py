"""Gemma <-> Qwen refusal-direction agreement on the shared 878-prompt set.

Different d_model, so direct cosine of the two direction vectors is undefined.
But the same 878 prompts appear in both refusal datasets (just different
labels — `is_refusal_gemma` vs `is_refusal_qwen` reflect the rolled-out
behavior of each model). For each prompt p we compute:

  s_G(p) = arditi_G_dir . mean_pool_residual_G_at_L(p)
  s_Q(p) = arditi_Q_dir . mean_pool_residual_Q_at_L(p)

The score vectors {s_G(p)} and {s_Q(p)} live in the same 878-d 'prompt
space', so:
  - cosine similarity is meaningful
  - Pearson correlation is meaningful
  - cross-model AUC is meaningful (use Gemma score as predictor of
    is_refusal_qwen and vice versa)

Outputs:
  cross_model_direction.png — scatter of Gemma vs Qwen scores, coloured by
                              joint refusal label (both/only_G/only_Q/neither).
  cross_model_direction.json — per-layer cosine, Pearson, and cross-model AUC.
"""
import json
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

OUT_DIR = Path(__file__).parent
LAYER_IDXS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
TARGET_LAYER = 40   # report headline at L40 for both

GEMMA_EXTRACTS = Path("/home/ubuntu/extracts/gemma_refusal")
QWEN_EXTRACTS  = Path("/home/ubuntu/extracts/qwen36")
SAMPLES = {
    "gemma": (Path("/lambda/nfs/jsWnew/mechhack/datasets/refusal_probes/gemma4_31b/attacks_full.jsonl"),
              GEMMA_EXTRACTS),
    "qwen":  (Path("/lambda/nfs/jsWnew/mechhack/datasets/refusal_probes/qwen36/attacks_full.jsonl"),
              QWEN_EXTRACTS),
}


def load_pool_cache(samples_file, extracts_dir, model_name):
    """Build (N, n_layers, d) mean-pool cache. Returns features + sample_ids + labels + splits."""
    cache_path = OUT_DIR / f"pool_cache_{model_name}_refusal.npz"
    if cache_path.exists():
        z = np.load(cache_path)
        meta = json.loads((OUT_DIR / f"pool_cache_{model_name}_refusal_meta.json").read_text())
        return z["mean"], meta["sample_ids"], meta["is_refusal"], meta["splits"]
    print(f"  Caching {model_name} pool features...")
    samples = {}
    for line in open(samples_file):
        s = json.loads(line)
        samples[s["sample_id"]] = s
    mean_feats, ids_ok, refusal, splits = [], [], [], []
    skipped = 0
    t0 = time.time()
    for sid, s in samples.items():
        p = extracts_dir / f"{sid}.pt"
        if not p.exists():
            skipped += 1; continue
        ex = torch.load(str(p), weights_only=False, map_location="cpu")
        r = ex["residuals"].float()       # (n_layers, N, d)
        m = ex["attention_mask"].bool()
        if not m.any(): skipped += 1; continue
        m_f = m.float().unsqueeze(0).unsqueeze(-1)
        mean = (r * m_f).sum(1) / m_f.sum(1).clamp(min=1)
        mean_feats.append(mean.numpy().astype(np.float32))
        ids_ok.append(sid)
        refusal.append(s.get("is_refusal"))   # may be None for some Qwen rows
        splits.append(s.get("split"))
    arr = np.stack(mean_feats, axis=0)
    np.savez_compressed(cache_path, mean=arr)
    (OUT_DIR / f"pool_cache_{model_name}_refusal_meta.json").write_text(json.dumps(
        {"sample_ids": ids_ok, "is_refusal": refusal, "splits": splits},
        indent=2))
    print(f"  {model_name}: {arr.shape}, skipped {skipped}, {time.time()-t0:.1f}s")
    return arr, ids_ok, refusal, splits


def arditi_direction(features, labels, splits, layer_idx):
    """At one layer, on the train split, return the unit-norm
    (mean_refused - mean_complied) direction."""
    train_mask = np.array([s == "train" and r is not None for s, r in zip(splits, labels)])
    pos = np.array([r == True for r in labels])
    neg = np.array([r == False for r in labels])
    X_train_pos = features[train_mask & pos, layer_idx, :]
    X_train_neg = features[train_mask & neg, layer_idx, :]
    d = X_train_pos.mean(axis=0) - X_train_neg.mean(axis=0)
    return d / (np.linalg.norm(d) + 1e-8)


# Step 1: load pool caches for both models
print("=== loading pool caches ===")
g_feats, g_ids, g_lab, g_split = load_pool_cache(*SAMPLES["gemma"], "gemma")
q_feats, q_ids, q_lab, q_split = load_pool_cache(*SAMPLES["qwen"], "qwen")
common = sorted(set(g_ids) & set(q_ids))
print(f"  shared sample_ids: {len(common)}")

g_idx = {sid: i for i, sid in enumerate(g_ids)}
q_idx = {sid: i for i, sid in enumerate(q_ids)}

# Re-index to the shared set
g_perm = [g_idx[sid] for sid in common]
q_perm = [q_idx[sid] for sid in common]
g_feats = g_feats[g_perm]
q_feats = q_feats[q_perm]
g_lab = [g_lab[i] for i in g_perm]
q_lab = [q_lab[i] for i in q_perm]
splits = [g_split[g_idx[sid]] for sid in common]
print(f"  re-indexed: gemma {g_feats.shape}, qwen {q_feats.shape}")

# Step 2: compute Arditi directions per layer on train split, project on test
test_mask = np.array([s == "test" for s in splits])
train_mask = np.array([s == "train" for s in splits])
print(f"  test n={test_mask.sum()}  train n={train_mask.sum()}")

# Filter to prompts where BOTH models have a valid label (not None)
both_labeled_test = test_mask & np.array([
    g is not None and q is not None for g, q in zip(g_lab, q_lab)
])
print(f"  test with both labels: {both_labeled_test.sum()}")
test_idx = np.where(both_labeled_test)[0]

results = {"layers": LAYER_IDXS, "n_shared": len(common),
           "n_test_both_labeled": int(both_labeled_test.sum()),
           "per_layer": {}}

for L in LAYER_IDXS:
    li = LAYER_IDXS.index(L)
    g_dir = arditi_direction(g_feats, g_lab, splits, li)
    q_dir = arditi_direction(q_feats, q_lab, splits, li)
    s_G = g_feats[test_idx, li, :] @ g_dir
    s_Q = q_feats[test_idx, li, :] @ q_dir
    # cosine of the score-vectors over the 281 test prompts
    cos = float(s_G @ s_Q / (np.linalg.norm(s_G) * np.linalg.norm(s_Q) + 1e-12))
    pear = float(np.corrcoef(s_G, s_Q)[0, 1])
    # cross-model AUC: predict Qwen labels using Gemma score (and vice versa)
    yg = np.array([1 if g_lab[i] else 0 for i in test_idx], dtype=int)
    yq = np.array([1 if q_lab[i] else 0 for i in test_idx], dtype=int)
    try:
        auc_G_to_G = float(roc_auc_score(yg, s_G))
        auc_G_to_Q = float(roc_auc_score(yq, s_G))
        auc_Q_to_G = float(roc_auc_score(yg, s_Q))
        auc_Q_to_Q = float(roc_auc_score(yq, s_Q))
    except Exception:
        auc_G_to_G = auc_G_to_Q = auc_Q_to_G = auc_Q_to_Q = float("nan")
    results["per_layer"][str(L)] = {
        "cosine_score_vectors": cos, "pearson_score_vectors": pear,
        "auc_GdirOnGlabels": auc_G_to_G,
        "auc_GdirOnQlabels": auc_G_to_Q,
        "auc_QdirOnGlabels": auc_Q_to_G,
        "auc_QdirOnQlabels": auc_Q_to_Q,
    }
    print(f"  L{L:02d}: cos {cos:+.3f}  pear {pear:+.3f}  "
          f"AUC G->G={auc_G_to_G:.3f} G->Q={auc_G_to_Q:.3f}  "
          f"Q->G={auc_Q_to_G:.3f} Q->Q={auc_Q_to_Q:.3f}")

(OUT_DIR / "cross_model_direction.json").write_text(json.dumps(results, indent=2))
print(f"Saved {OUT_DIR / 'cross_model_direction.json'}")

# Step 3: scatter + per-layer cosine plot at TARGET_LAYER
li = LAYER_IDXS.index(TARGET_LAYER)
g_dir = arditi_direction(g_feats, g_lab, splits, li)
q_dir = arditi_direction(q_feats, q_lab, splits, li)
s_G = g_feats[test_idx, li, :] @ g_dir
s_Q = q_feats[test_idx, li, :] @ q_dir
yg = np.array([1 if g_lab[i] else 0 for i in test_idx], dtype=int)
yq = np.array([1 if q_lab[i] else 0 for i in test_idx], dtype=int)
agree = (yg == yq)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Scatter at L40
both_ref = (yg == 1) & (yq == 1)
both_comp = (yg == 0) & (yq == 0)
g_only = (yg == 1) & (yq == 0)
q_only = (yg == 0) & (yq == 1)

for mask, c, lbl in [(both_ref, "#dc2626", f"both refused (n={both_ref.sum()})"),
                       (both_comp, "#16a34a", f"both complied (n={both_comp.sum()})"),
                       (g_only,   "#7c3aed", f"only Gemma refused (n={g_only.sum()})"),
                       (q_only,   "#f59e0b", f"only Qwen refused (n={q_only.sum()})")]:
    ax1.scatter(s_G[mask], s_Q[mask], c=c, s=18, alpha=0.7, label=lbl)
ax1.axhline(0, color="black", lw=0.5, alpha=0.3)
ax1.axvline(0, color="black", lw=0.5, alpha=0.3)
ax1.set_xlabel(f"Gemma Arditi-projection score @ L{TARGET_LAYER}")
ax1.set_ylabel(f"Qwen Arditi-projection score @ L{TARGET_LAYER}")
cos = results["per_layer"][str(TARGET_LAYER)]["cosine_score_vectors"]
pear = results["per_layer"][str(TARGET_LAYER)]["pearson_score_vectors"]
ax1.set_title(f"L{TARGET_LAYER} per-prompt agreement\n"
              f"cos(s_G, s_Q)={cos:+.3f}   Pearson={pear:+.3f}   n={len(s_G)}")
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(True, alpha=0.3)

# Per-layer cosine + Pearson + cross-AUC
Ls = LAYER_IDXS
cosines = [results["per_layer"][str(L)]["cosine_score_vectors"] for L in Ls]
pears   = [results["per_layer"][str(L)]["pearson_score_vectors"] for L in Ls]
g2q     = [results["per_layer"][str(L)]["auc_GdirOnQlabels"]      for L in Ls]
q2g     = [results["per_layer"][str(L)]["auc_QdirOnGlabels"]      for L in Ls]

ax2.plot(Ls, cosines, "-o", label="cosine of score vectors", color="#1d4ed8")
ax2.plot(Ls, pears,    "-s", label="Pearson r", color="#7c3aed")
ax2.plot(Ls, g2q,      "--^", label="AUC: Gemma dir -> Qwen labels", color="#dc2626")
ax2.plot(Ls, q2g,      "--v", label="AUC: Qwen dir -> Gemma labels", color="#f59e0b")
ax2.axhline(0.5, color="black", lw=0.5, alpha=0.3)
ax2.set_xticks(Ls)
ax2.set_xlabel("Layer index")
ax2.set_ylabel("similarity / AUC")
ax2.set_title("Cross-model agreement vs layer")
ax2.set_ylim(-0.2, 1.0)
ax2.legend(loc="lower right", fontsize=8)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / "cross_model_direction.png", dpi=140, bbox_inches="tight")
print(f"Saved {OUT_DIR / 'cross_model_direction.png'}")
