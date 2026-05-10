"""Combined-probe (cyber_3 train ∪ refusal train) causality on rewrites_k7.

Train a single LR on cyber_3-positive (prohibited) + refusal-positive
samples = "harmful intent" direction.

Apply to all 648 rewrite features (already extracted in
experiments/30_rewrites_causality_omar/features_rewrites.npz).

Match to existing rollouts.jsonl + judgements.jsonl (211 rollouts: original,
lr_best, arditi_best). Compute Pr(f|edit) and Pr(model|f flipped) for the
combined probe vs the refusal-only probe.

Tests two layers per pooling: L40 mean (refusal best mean) and L45 last-tok
(refusal best last). For each:
  - refusal-only probe   (train on refusal train split)
  - cyber_3-only probe   (train on cyber_3 train split)
  - combined probe       (train on UNION)

Output:
  combined_causality.json
  combined_causality.md
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
HERE.mkdir(parents=True, exist_ok=True)

REWRITES_FEATURES = REPO_ROOT / "experiments" / "30_rewrites_causality_omar" / "features_rewrites.npz"
ROLLOUTS = REPO_ROOT / "experiments" / "13_pre_rewrites_omar" / "rollouts.jsonl"
JUDGE = REPO_ROOT / "experiments" / "13_pre_rewrites_omar" / "judgements.jsonl"
ATTRS_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"

CYBER_EXTRACTS = Path("/home/ubuntu/extracts/cyber_all_omar")
REFUSAL_EXTRACTS = Path("/home/ubuntu/extracts/gemma_refusal_13layers/gemma4_31b")
CYBER_TRAIN = REPO_ROOT / "datasets" / "cyber_probes" / "train.jsonl"
REFUSAL_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"

LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]


def cyber_3_label(category):
    if category == "prohibited": return 1
    return 0  # everything else is negative for cyber_3


def pool_one(p, layer_idx, pooling):
    ex = torch.load(str(p), weights_only=False, map_location="cpu", mmap=True)
    residuals = ex["residuals"][layer_idx].float()
    mask = ex["attention_mask"].bool().squeeze()
    n = int(mask.sum().item())
    if n < 2: return None
    if pooling == "mean":
        m = mask.float().unsqueeze(-1)
        return ((residuals * m).sum(dim=0) / n).numpy().astype(np.float32)
    else:  # last
        last_idx = int(mask.nonzero().max().item())
        return residuals[last_idx, :].numpy().astype(np.float32)


def load_cyber_3_train(layer, pooling):
    rows = []
    with CYBER_TRAIN.open() as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["sample_id"], cyber_3_label(r["category"])))
    Xs, ys = [], []
    li = LAYERS.index(layer)
    for sid, lbl in rows:
        p = CYBER_EXTRACTS / f"{sid}.pt"
        if not p.exists(): continue
        feat = pool_one(p, li, pooling)
        if feat is None or not np.isfinite(feat).all(): continue
        Xs.append(feat); ys.append(lbl)
    return np.stack(Xs), np.asarray(ys, np.int64)


def load_refusal_train(layer, pooling):
    rows = []
    with REFUSAL_FULL.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") != "train": continue
            if r.get("is_refusal") is None: continue
            rows.append((r["sample_id"], int(bool(r["is_refusal"]))))
    Xs, ys = [], []
    li = LAYERS.index(layer)
    for sid, lbl in rows:
        p = REFUSAL_EXTRACTS / f"{sid}.pt"
        if not p.exists(): continue
        feat = pool_one(p, li, pooling)
        if feat is None or not np.isfinite(feat).all(): continue
        Xs.append(feat); ys.append(lbl)
    return np.stack(Xs), np.asarray(ys, np.int64)


def main():
    print("[combined-probe causality on rewrites_k7]", flush=True)
    # Load rewrite features
    print("  loading rewrite features...", flush=True)
    z = np.load(REWRITES_FEATURES, allow_pickle=True)
    sids = list(z["sample_ids"])
    ridxs = list(z["rewrite_idxs"])
    means = z["mean"]   # (N, 13, d)
    lasts = z["last"]   # (N, 13, d)
    N = len(sids)
    print(f"    {N} entries", flush=True)

    # Load orig labels (refusal-or-not on the original prompt)
    orig_label = {}
    with ATTRS_FULL.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("is_refusal") is not None:
                orig_label[r["sample_id"]] = int(bool(r["is_refusal"]))

    # Load rollouts/judges (only have lr_best, arditi_best, original variants)
    rollout_judge = {}  # (sample_id, variant) -> is_refusal_judge bool
    with JUDGE.open() as f:
        for line in f:
            r = json.loads(line)
            v = r.get("is_refusal_judge")
            if v is None: continue
            rollout_judge[(r["sample_id"], r["variant"])] = v
    print(f"    judgements: {len(rollout_judge)}", flush=True)
    # Map lr_best/arditi_best to which rewrite_idx — load from rollouts
    rollout_label_to_ridx = {}  # (sample_id, variant) -> rewrite_idx (which of the 7 was best)
    with ROLLOUTS.open() as f:
        for line in f:
            r = json.loads(line)
            if r["variant"] in ("lr_best", "arditi_best"):
                # We need to recover which rewrite_idx this was. Not directly stored;
                # match by rewrite text equality if possible.
                rollout_label_to_ridx[(r["sample_id"], r["variant"])] = (r.get("variant_idx"), r.get("prompt_chars"))

    # Train probes and score rewrites
    out = {}
    for layer, pooling in [(40, "mean"), (45, "last"), (40, "last"), (35, "mean"), (45, "mean")]:
        key = f"L{layer}_{pooling}"
        print(f"\n  --- {key} ---", flush=True)
        t0 = time.time()
        Xc, yc = load_cyber_3_train(layer, pooling)
        Xr, yr = load_refusal_train(layer, pooling)
        print(f"    cyber_3 train: {Xc.shape} ({yc.sum()} pos / {(yc==0).sum()} neg)", flush=True)
        print(f"    refusal train: {Xr.shape} ({yr.sum()} pos / {(yr==0).sum()} neg)", flush=True)
        print(f"    loaded in {time.time()-t0:.0f}s", flush=True)

        # Three probes
        probes = {}
        probes["refusal_only"] = LogisticRegression(C=1.0, max_iter=2000).fit(Xr, yr)
        probes["cyber_3_only"] = LogisticRegression(C=1.0, max_iter=2000).fit(Xc, yc)
        X_combo = np.concatenate([Xc, Xr], axis=0)
        y_combo = np.concatenate([yc, yr], axis=0)
        probes["combined"] = LogisticRegression(C=1.0, max_iter=2000).fit(X_combo, y_combo)

        # Score every rewrite under each probe
        li = LAYERS.index(layer)
        feat_block = means[:, li, :] if pooling == "mean" else lasts[:, li, :]
        scores = {pname: p.predict_proba(feat_block)[:, 1] for pname, p in probes.items()}

        # For causality: only have rollouts for original, lr_best, arditi_best.
        # Check: among the 81 prompts where original was refusal AND we rolled lr_best/arditi_best,
        # what is the probe score on the original vs the rolled variant? Did model flip?
        results = {}
        for pname, sc in scores.items():
            # Build per-sample probe score on original (rewrite_idx=-1) and on each rewrite
            # idx 0..6 (we have features for all 7 rewrites, but rollouts only for one of them).
            n_total = 0
            n_f_flip = 0
            n_m_flip = 0
            n_concord_denom = 0
            n_concord_num = 0
            for i, sid in enumerate(sids):
                if ridxs[i] != -1: continue   # only count originals here as the "anchor"
                if orig_label.get(sid) != 1: continue  # only orig=refusal
                # Probe score on original
                orig_score = sc[i]
                # Find the lr_best variant rollout for this sid
                judge_lrbest = rollout_judge.get((sid, "lr_best"))
                if judge_lrbest is None: continue
                # We need probe score on the lr_best rewrite. We have features for all 7, but
                # we don't know which rewrite_idx was selected as lr_best at the time. Use
                # MIN-score rewrite over the 7 as a proxy for "probe-best flipped".
                rewrite_scores = []
                for j in range(N):
                    if sids[j] == sid and ridxs[j] >= 0:
                        rewrite_scores.append((ridxs[j], sc[j]))
                if not rewrite_scores: continue
                # Best-flipped = lowest probe score
                best_idx, best_score = min(rewrite_scores, key=lambda x: x[1])
                n_total += 1
                f_orig = orig_score >= 0.5
                f_now = best_score < 0.5
                if not f_orig:
                    continue  # skip if probe wasn't flagging the original
                if f_now:
                    n_f_flip += 1
                    n_concord_denom += 1
                    if judge_lrbest is False:
                        n_concord_num += 1
                if judge_lrbest is False:
                    n_m_flip += 1

            from math import sqrt
            def w(k, n, z=1.96):
                if n == 0: return float("nan"), float("nan"), float("nan")
                p = k / n
                den = 1 + z*z/n
                c = (p + z*z/(2*n))/den
                s = z/den * sqrt(p*(1-p)/n + z*z/(4*n*n))
                return p, max(0.0, c-s), min(1.0, c+s)
            pf = w(n_f_flip, n_total)
            pm = w(n_m_flip, n_total)
            pc = w(n_concord_num, n_concord_denom)
            results[pname] = {
                "n_orig_refusal": n_total,
                "Pr_f_given_edit":     {"k": n_f_flip, "n": n_total, "point": pf[0], "ci": [pf[1], pf[2]]},
                "Pr_model_given_edit": {"k": n_m_flip, "n": n_total, "point": pm[0], "ci": [pm[1], pm[2]]},
                "Pr_model_given_f":    {"k": n_concord_num, "n": n_concord_denom, "point": pc[0], "ci": [pc[1], pc[2]]},
            }
            print(f"    {pname:>15}: Pr(f|e)={pf[0]:.3f}  Pr(m|e)={pm[0]:.3f}  "
                  f"Pr(m|f)={pc[0]:.3f}  ({n_concord_num}/{n_concord_denom})", flush=True)
        out[key] = results

    OUT = HERE / "combined_causality.json"
    OUT.write_text(json.dumps(out, indent=2))

    # Pretty md
    md = ["# Combined-probe (cyber_3 ∪ refusal) causality on rewrites_k7\n"]
    md.append("Probe trained on: refusal-only / cyber_3-only / combined.")
    md.append("Edits: 81 prompts × 7 rewrites = 567. Per prompt, 'flipped' edit = the")
    md.append("lowest-scoring rewrite. Match to lr_best rollout (existing). Pr(model|f)")
    md.append("counts (model flipped) AND (probe says flipped).\n")
    for key, results in out.items():
        md.append(f"\n## {key}\n")
        md.append("| Probe | Pr(f|edit) | Pr(model|edit) | Pr(model|f) | n (probe-flips) |")
        md.append("|---|---:|---:|---:|---:|")
        for pname, r in results.items():
            pf = r["Pr_f_given_edit"]; pm = r["Pr_model_given_edit"]; pc = r["Pr_model_given_f"]
            pf_str = f'{pf["k"]}/{pf["n"]} = {pf["point"]:.3f}'
            pm_str = f'{pm["k"]}/{pm["n"]} = {pm["point"]:.3f}'
            pc_str = f'{pc["k"]}/{pc["n"]} = {pc["point"]:.3f}'
            md.append(f"| **{pname}** | {pf_str} | {pm_str} | {pc_str} | {pc['n']} |")
    (HERE / "combined_causality.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {OUT} and {HERE/'combined_causality.md'}", flush=True)


if __name__ == "__main__":
    main()
