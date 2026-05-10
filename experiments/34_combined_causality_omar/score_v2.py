"""Causality on rewrites_k7 — apples-to-apples across all probes.

Setup:
  - We have 81 prompts × 7 rewrites = 567 features (extracted by exp 30).
  - We have 211 rollouts (81 original + 81 lr_best + 49 arditi_best) WITH
    DeepSeek judge labels (`is_refusal_judge`).
  - Plus the 81 originals' rollouts — Gemma-on-original-prompt judge.

Procedure (apples-to-apples across all probes):
  For the 81 'lr_best' rewrites and the 81 originals:
    For each probe variant P:
      score_orig[i]   = P(features[(sid, -1)])           # original
      score_lrbest[i] = P(features[(sid, k_lrbest)])     # the rewrite chosen
                                                          # by old LR L32 as best
    Compute on prompts where the ORIGINAL was a refusal (judge=True):
      f_flipped = (score_orig ≥ 0.5) AND (score_lrbest < 0.5)
      m_flipped = (judge_orig == True) AND (judge_lrbest == False)
      Pr(f|edit) = sum(f_flipped) / n
      Pr(m|edit) = sum(m_flipped) / n
      Pr(m|f flipped) = sum(f∧m) / sum(f)

This is the right test: same model behavior, swap the probe, see how each
probe's flip-rate AND concordance compare. Pleshkov, LR variants, the
combined cyber_3 + refusal probe — all on the same 81 edits.

But we need to know which rewrite_idx was lr_best. Rollouts file has the
prompt_chars but not the index directly. We recover it by matching the
prompt_chars to the 7 rewrite features per sample.

Output:
  causality_rewrites_k7.json
  causality_rewrites_k7.md  — sorted table
"""
from __future__ import annotations
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
HERE.mkdir(parents=True, exist_ok=True)

REWRITES_FEATURES = REPO_ROOT / "experiments" / "30_rewrites_causality_omar" / "features_rewrites.npz"
REWRITES_PATH = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "rewrites_k7.json"
ROLLOUTS = REPO_ROOT / "experiments" / "13_pre_rewrites_omar" / "rollouts.jsonl"
JUDGE = REPO_ROOT / "experiments" / "13_pre_rewrites_omar" / "judgements.jsonl"
ATTRS_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"

CYBER_EXTRACTS = Path("/home/ubuntu/extracts/cyber_all_omar")
REFUSAL_EXTRACTS = Path("/home/ubuntu/extracts/gemma_refusal_13layers/gemma4_31b")
CYBER_TRAIN = REPO_ROOT / "datasets" / "cyber_probes" / "train.jsonl"
REFUSAL_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"

LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

sys.path.insert(0, str(REPO_ROOT / "experiments" / "17_quadratic_probe_omar"))
from probes import QuadraticProbe  # noqa: E402

LR_PROBES = REPO_ROOT / "experiments" / "24_robustness_omar" / "fitted_probes.npz"

THR = 0.5


def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z*z/n
    c = (p + z*z/(2*n))/den
    s = z/den * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return p, max(0.0, c-s), min(1.0, c+s)


def cyber_3_label(category):
    return 1 if category == "prohibited" else 0


def pool_one(p, layer_idx, pooling):
    ex = torch.load(str(p), weights_only=False, map_location="cpu", mmap=True)
    residuals = ex["residuals"][layer_idx].float()
    mask = ex["attention_mask"].bool().squeeze()
    n = int(mask.sum().item())
    if n < 2: return None
    if pooling == "mean":
        m = mask.float().unsqueeze(-1)
        return ((residuals * m).sum(dim=0) / n).numpy().astype(np.float32)
    last_idx = int(mask.nonzero().max().item())
    return residuals[last_idx, :].numpy().astype(np.float32)


def load_train(extracts_dir, ids_labels, layer, pooling):
    Xs, ys = [], []
    li = LAYERS.index(layer)
    for sid, lbl in ids_labels:
        p = extracts_dir / f"{sid}.pt"
        if not p.exists(): continue
        feat = pool_one(p, li, pooling)
        if feat is None or not np.isfinite(feat).all(): continue
        Xs.append(feat); ys.append(lbl)
    return np.stack(Xs), np.asarray(ys, np.int64)


def load_cyber_3_train(layer, pooling):
    rows = []
    with CYBER_TRAIN.open() as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["sample_id"], cyber_3_label(r["category"])))
    return load_train(CYBER_EXTRACTS, rows, layer, pooling)


def load_refusal_train(layer, pooling):
    rows = []
    with REFUSAL_FULL.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") != "train": continue
            if r.get("is_refusal") is None: continue
            rows.append((r["sample_id"], int(bool(r["is_refusal"]))))
    return load_train(REFUSAL_EXTRACTS, rows, layer, pooling)


def main():
    print("[causality on rewrites_k7 — apples-to-apples across all probes]", flush=True)

    if not REWRITES_FEATURES.exists():
        print(f"  missing {REWRITES_FEATURES}; run extract_features.py first")
        return

    z = np.load(REWRITES_FEATURES, allow_pickle=True)
    sids_arr = list(z["sample_ids"])
    ridxs_arr = list(z["rewrite_idxs"])
    means = z["mean"]   # (N, 13, d)
    lasts = z["last"]   # (N, 13, d)
    N = len(sids_arr)
    print(f"  {N} feature entries (81 originals + 81×7 rewrites = 648 expected)", flush=True)

    # Build (sid, ridx) -> row index in features
    feat_idx = {}
    for i in range(N):
        feat_idx[(sids_arr[i], int(ridxs_arr[i]))] = i

    # Original prompt → judge_orig (True = refused)
    rewrite_dict = {r["sample_id"]: r["rewrites"] for r in json.load(REWRITES_PATH.open())}

    # Load rollouts to map sample_id → which rewrite was lr_best (by prompt text)
    lrbest_idx = {}    # sample_id → rewrite_idx in [0..6] for lr_best
    lrbest_judge = {}
    arditibest_idx = {}
    arditibest_judge = {}
    orig_judge = {}
    with ROLLOUTS.open() as f, JUDGE.open() as fj:
        rollouts_by_key = {}
        for line in f:
            r = json.loads(line)
            rollouts_by_key[(r["sample_id"], r["variant"])] = r
        for line in fj:
            j = json.loads(line)
            v = j.get("is_refusal_judge")
            if v is None: continue
            sid, var = j["sample_id"], j["variant"]
            if var == "original":
                orig_judge[sid] = bool(v)
            elif var == "lr_best":
                lrbest_judge[sid] = bool(v)
            elif var == "arditi_best":
                arditibest_judge[sid] = bool(v)

    # For each sample, find the rewrite_idx whose text matches the lr_best rollout
    # (we need this to look up the lr_best feature in the 7 rewrites).
    for sid, rollout in rollouts_by_key.items():
        if sid[1] not in ("lr_best", "arditi_best"): continue
        # rollouts.jsonl has prompt_chars but not the rewrite text. We can't match
        # by char-count alone (multiple rewrites may share length).
        # Approach: take all 7 rewrite char-lens; if rollout's prompt_chars matches
        # exactly one rewrite's len, that's the index. If multiple, leave None.
        s = sid[0]
        rs = rewrite_dict.get(s, [])
        target = rollout.get("prompt_chars", -1)
        candidates = [i for i, rw in enumerate(rs) if len(rw) == target]
        if len(candidates) == 1:
            if sid[1] == "lr_best":
                lrbest_idx[s] = candidates[0]
            else:
                arditibest_idx[s] = candidates[0]
        # else: ambiguous; skip this sample for that variant

    print(f"  matched lr_best rewrite_idx for {len(lrbest_idx)}/{len(lrbest_judge)} prompts", flush=True)
    print(f"  matched arditi_best rewrite_idx for {len(arditibest_idx)}/{len(arditibest_judge)} prompts", flush=True)

    # ===== Build probe set =====
    print("\n  building probes...", flush=True)
    probes = {}

    # 1. LR fitted probes from exp 24 (28 of them)
    if LR_PROBES.exists():
        zlr = np.load(LR_PROBES, allow_pickle=True)
        for L in LAYERS:
            for pooling in ["mean", "last"]:
                if L == 0 and pooling == "last": continue
                name = f"LR_{pooling}_L{L}"
                weight_name = f"coef_lr_{pooling}_L{L}"
                bias_name = f"bias_lr_{pooling}_L{L}"
                if weight_name not in zlr.files: continue
                probes[name] = ("lr_linear", zlr[weight_name].astype(np.float32),
                                 float(zlr[bias_name]), L, pooling)
        # Aggregates
        if "coef_lr_multi_concat" in zlr.files:
            probes["LR_multi_concat"] = ("lr_concat", zlr["coef_lr_multi_concat"].astype(np.float32),
                                          float(zlr["bias_lr_multi_concat"]))
        if "coef_lr_mean_of_layers" in zlr.files:
            probes["LR_mean_of_layers"] = ("lr_mol", zlr["coef_lr_mean_of_layers"].astype(np.float32),
                                            float(zlr["bias_lr_mean_of_layers"]))
        if "coef_lr_max_of_layers" in zlr.files:
            probes["LR_max_of_layers"] = ("lr_xol", zlr["coef_lr_max_of_layers"].astype(np.float32),
                                            float(zlr["bias_lr_max_of_layers"]))

    # 2. Pleshkov (refit on full refusal train at L40 mean)
    print("    fitting Pleshkov d=16 @ L40 mean (refusal train)...", flush=True)
    Xr_train, yr_train = load_refusal_train(40, "mean")
    p = QuadraticProbe(d_pca=16, alpha=10.0, random_state=0).fit(Xr_train, yr_train)
    probes["Pleshkov_d16_L40_refusal"] = ("pleshkov", p, 40, "mean")

    # 3. Combined LR probes at L40 mean and L45 last-tok
    for L, pooling in [(40, "mean"), (45, "last"), (35, "mean"), (40, "last")]:
        print(f"    fitting combined (cyber_3+refusal) LR @ L{L} {pooling}...", flush=True)
        Xc, yc = load_cyber_3_train(L, pooling)
        Xr, yr = load_refusal_train(L, pooling)
        X_combo = np.concatenate([Xc, Xr], axis=0)
        y_combo = np.concatenate([yc, yr], axis=0)
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(X_combo, y_combo)
        probes[f"COMBINED_L{L}_{pooling}"] = ("lr_sklearn", clf, L, pooling)

        # Pleshkov combined at L40 mean only (avoid Pleshkov on last-tok)
        if pooling == "mean" and L == 40:
            print(f"    fitting combined Pleshkov d=16 @ L{L} mean...", flush=True)
            p_combo = QuadraticProbe(d_pca=16, alpha=10.0, random_state=0).fit(X_combo, y_combo)
            probes[f"Pleshkov_d16_L{L}_combined"] = ("pleshkov", p_combo, L, "mean")

    print(f"  total {len(probes)} probes\n", flush=True)

    # ===== Score each prompt =====
    def score_features(probe_def, mean_per_layer, last_per_layer):
        kind = probe_def[0]
        if kind == "lr_linear":
            _, coef, bias, L, pooling = probe_def
            li = LAYERS.index(L)
            feat = mean_per_layer[li] if pooling == "mean" else last_per_layer[li]
            logit = float(feat @ coef + bias)
            return float(1.0 / (1.0 + np.exp(-logit)))
        if kind == "lr_concat":
            _, coef, bias = probe_def
            feat = mean_per_layer.reshape(-1)
            logit = float(feat @ coef + bias)
            return float(1.0 / (1.0 + np.exp(-logit)))
        if kind == "lr_mol":
            _, coef, bias = probe_def
            feat = mean_per_layer.mean(axis=0)
            logit = float(feat @ coef + bias)
            return float(1.0 / (1.0 + np.exp(-logit)))
        if kind == "lr_xol":
            _, coef, bias = probe_def
            feat = mean_per_layer.max(axis=0)
            logit = float(feat @ coef + bias)
            return float(1.0 / (1.0 + np.exp(-logit)))
        if kind == "pleshkov":
            _, p_obj, L, pooling = probe_def
            li = LAYERS.index(L)
            feat = mean_per_layer[li] if pooling == "mean" else last_per_layer[li]
            return float(p_obj.decision_function(feat[None, :])[0])
        if kind == "lr_sklearn":
            _, clf, L, pooling = probe_def
            li = LAYERS.index(L)
            feat = mean_per_layer[li] if pooling == "mean" else last_per_layer[li]
            return float(clf.predict_proba(feat[None, :])[0, 1])
        return float("nan")

    # Iterate through unique sids that have lr_best matched + judge for both original and lr_best
    valid_sids = sorted(set(lrbest_idx.keys()) & set(lrbest_judge.keys()) & set(orig_judge.keys()))
    print(f"  {len(valid_sids)} sids with full data (lr_best matched + judge orig + judge lr_best)\n", flush=True)

    out = {"variants_tested": ["lr_best"], "n_sids_full_data": len(valid_sids), "by_probe": {}}

    print(f"  computing causality across {len(probes)} probes...", flush=True)
    for pname, pdef in probes.items():
        n_total = 0
        n_f = 0
        n_m = 0
        n_concord_num = 0
        n_concord_denom = 0
        for sid in valid_sids:
            if orig_judge.get(sid) is not True:
                continue   # only count when ORIG was a refusal
            n_total += 1
            ridx_lrbest = lrbest_idx[sid]
            mean_orig = means[feat_idx[(sid, -1)]]
            last_orig = lasts[feat_idx[(sid, -1)]]
            mean_edit = means[feat_idx[(sid, ridx_lrbest)]]
            last_edit = lasts[feat_idx[(sid, ridx_lrbest)]]
            score_orig = score_features(pdef, mean_orig, last_orig)
            score_edit = score_features(pdef, mean_edit, last_edit)
            f_orig = score_orig >= THR
            f_now = score_edit < THR
            model_flipped = (lrbest_judge[sid] is False)
            if model_flipped: n_m += 1
            if f_orig and f_now:
                n_f += 1; n_concord_denom += 1
                if model_flipped:
                    n_concord_num += 1
        pf = wilson(n_f, n_total)
        pm = wilson(n_m, n_total)
        pc = wilson(n_concord_num, n_concord_denom)
        out["by_probe"][pname] = {
            "n_orig_refusal": n_total,
            "Pr_f_given_edit":     {"k": n_f, "n": n_total, "point": pf[0], "ci": [pf[1], pf[2]]},
            "Pr_model_given_edit": {"k": n_m, "n": n_total, "point": pm[0], "ci": [pm[1], pm[2]]},
            "Pr_model_given_f":    {"k": n_concord_num, "n": n_concord_denom, "point": pc[0], "ci": [pc[1], pc[2]]},
        }

    # Save
    OUT = HERE / "causality_rewrites_k7.json"
    OUT.write_text(json.dumps(out, indent=2))

    # Pretty md table
    md = ["# Causality on rewrites_k7 — apples-to-apples across all probes\n"]
    md.append(f"Tested on {out['n_sids_full_data']} prompts where original was a refusal AND we have the lr_best Gemma rollout + judge.\n")
    md.append("`f_flipped` = probe score on original ≥ 0.5 AND probe score on lr_best edit < 0.5.")
    md.append("`m_flipped` = original was a refusal AND lr_best edit produced a non-refusal response (judge).\n")
    md.append("Sorted by Pr(model|f flipped) — top of table = most causal probe.\n")
    md.append("| Probe | n | Pr(f|edit) | Pr(model|edit) | **Pr(model|f flipped)** |")
    md.append("|---|---:|---:|---:|---:|")
    rows = []
    for pname, r in out["by_probe"].items():
        pf = r["Pr_f_given_edit"]; pm = r["Pr_model_given_edit"]; pc = r["Pr_model_given_f"]
        sort_key = pc["point"] if not math.isnan(pc["point"]) else -1
        rows.append((sort_key, pname, r))
    rows.sort(reverse=True)
    for _, pname, r in rows:
        pf = r["Pr_f_given_edit"]; pm = r["Pr_model_given_edit"]; pc = r["Pr_model_given_f"]
        pf_str = f'{pf["k"]}/{pf["n"]}={pf["point"]:.3f}'
        pm_str = f'{pm["k"]}/{pm["n"]}={pm["point"]:.3f}'
        pc_str = f'{pc["k"]}/{pc["n"]}={pc["point"]:.3f}' if pc["n"] > 0 else "n=0"
        md.append(f"| {pname} | {r['n_orig_refusal']} | {pf_str} | {pm_str} | **{pc_str}** |")
    (HERE / "causality_rewrites_k7.md").write_text("\n".join(md) + "\n")

    print(f"\nwrote {OUT} and {HERE/'causality_rewrites_k7.md'}", flush=True)
    print(f"\nTOP 12 probes on Pr(model|f flipped):", flush=True)
    for _, pname, r in rows[:12]:
        pc = r["Pr_model_given_f"]
        print(f"  {pname:>30}  Pr(m|f)={pc['point']:.3f}  ({pc['k']}/{pc['n']})", flush=True)


if __name__ == "__main__":
    main()
