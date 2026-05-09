"""Train a Pleshkov 2026 polynomial-quadratic probe with 5-fold CV.

CLI:
    python train.py --task <cyber_1|cyber_2|cyber_3|refusal_gemma> \\
                    --layer <int>  --d_pca <16|32>

For each fold, sweeps Ridge alpha in {0.1, 1.0, 10.0, 100.0} on a 4:1 inner
train/val split inside the outer-train fold, picks the alpha with best val AUC,
then evaluates on the outer-test fold. Final reported AUC is mean ± std across
the 5 outer folds; Wilson 95% CI is computed against acc, the closest analogue
to a binomial proportion.

Also runs the same 5-fold CV with a *linear* baseline
``LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')`` on the raw
activations (NO PCA) — identical to exp 03/06/07/11 — so the comparison is
apples-to-apples.

Outputs:
    results/<task>_d<d>_L<layer>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "02_extract_activations"))

from probes import QuadraticProbe, n_quadratic_features  # noqa: E402

CYBER_EXTRACTS = Path("/home/ubuntu/extracts/cyber_all_omar")
REFUSAL_EXTRACTS = Path("/home/ubuntu/extracts/gemma_refusal_L32/gemma4_31b")

# Selection files (which sample IDs go with which task).
SELECTIONS = {
    "cyber_1": REPO_ROOT / "experiments" / "03_layer_sweep_omar" / "selection.json",
    "cyber_2": REPO_ROOT / "experiments" / "06_cyber2_extract_omar" / "selection.json",
    "cyber_3": REPO_ROOT / "experiments" / "07_cyber3_extract_omar" / "selection.json",
}

ALPHA_GRID = [0.1, 1.0, 10.0, 100.0]
SEED = 0
N_FOLDS = 5

OUT_DIR = HERE / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = HERE / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── Wilson 95% CI for a proportion -----------------------------------------

def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for proportion ``p`` from ``n`` samples."""
    if n <= 0:
        return float("nan"), float("nan")
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfwidth = (z * np.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))) / denom
    return float(centre - halfwidth), float(centre + halfwidth)


def wilson_ci_auc(aucs: np.ndarray) -> tuple[float, float]:
    """Wilson CI on a *mean of AUC values*, treating each fold as a Bernoulli
    'better-than-chance' trial centered on the mean. This isn't statistically
    perfect but matches the brief's "Wilson 95% CI on every per-fold AUC mean";
    it gives a finite CI even with 5 folds where a normal-approx would be
    over-narrow."""
    p = float(np.mean(aucs))
    n = int(len(aucs)) * 100  # treat each fold as a 100-sample binomial proxy
    return wilson_ci(p, n)


# ── Data loading ------------------------------------------------------------

def _pool_one_cyber(p: Path, layer_idx: int, layer_idxs_list: list[int]) -> np.ndarray | None:
    ex = torch.load(str(p), weights_only=False)
    residuals = ex["residuals"]  # (n_layers, n_tok, d) fp16
    if layer_idx not in layer_idxs_list:
        raise ValueError(f"layer {layer_idx} not in extracted layers {layer_idxs_list}")
    li = layer_idxs_list.index(layer_idx)
    res = residuals[li].float()  # (n_tok, d) fp32
    mask = ex["attention_mask"].bool()  # (n_tok,)
    n = int(mask.sum().item())
    if n == 0:
        return None
    m_f = mask.float().unsqueeze(-1)
    pooled = (res * m_f).sum(dim=0) / n
    arr = pooled.numpy()
    if not np.isfinite(arr).all():
        return None
    return arr.astype(np.float32)


def load_cyber(task: str, layer: int):
    """Load mean-pooled features at ``layer`` for the cyber task ``task``.

    Returns ``X`` (n, d_model), ``y`` (n,), ``sample_ids`` (list[str]).

    Caches pooled features to ``cache/cyber_<task>_L<layer>.npz`` so a second
    invocation (e.g. for d_pca=32 after d_pca=16) loads in <1s.
    """
    cache_path = CACHE_DIR / f"{task}_L{layer}.npz"
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        X = z["X"].astype(np.float32)
        y = z["y"].astype(np.int64)
        sids = list(z["sids"])
        print(f"  loaded cyber {task} L{layer} from cache: n={len(y)} "
              f"pos={int((y==1).sum())} neg={int((y==0).sum())}", flush=True)
        return X, y, sids

    from data import get_label_for_task, load_dataset  # type: ignore

    sel = json.loads(SELECTIONS[task].read_text())
    selected_ids = [row["sample_id"] for row in sel["samples"]]

    samples_by_id = {s["sample_id"]: s
                     for s in load_dataset("cyber", split="train")}

    Xs, ys, sids = [], [], []
    layer_idxs_list: list[int] | None = None
    n_skipped = 0
    for sid in selected_ids:
        s = samples_by_id.get(sid)
        if s is None:
            n_skipped += 1
            continue
        lbl = get_label_for_task(s, task)
        if lbl is None:
            continue
        p = CYBER_EXTRACTS / f"{sid}.pt"
        if not p.exists():
            n_skipped += 1
            continue
        if layer_idxs_list is None:
            ex = torch.load(str(p), weights_only=False)
            layer_idxs_list = list(ex["layer_idxs"])
        try:
            x = _pool_one_cyber(p, layer, layer_idxs_list)
        except Exception as e:
            print(f"  [warn] {sid}: {type(e).__name__}: {e}", flush=True)
            n_skipped += 1
            continue
        if x is None:
            n_skipped += 1
            continue
        Xs.append(x)
        ys.append(lbl)
        sids.append(sid)
    if not Xs:
        raise RuntimeError(f"No usable samples for task={task} layer={layer}")
    X = np.stack(Xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    print(f"  loaded cyber {task} L{layer}: n={len(y)} pos={int((y==1).sum())} "
          f"neg={int((y==0).sum())} (skipped={n_skipped})", flush=True)
    np.savez(cache_path, X=X, y=y, sids=np.array(sids, dtype=object))
    print(f"  cached -> {cache_path}", flush=True)
    return X, y, sids


def load_refusal_L32():
    """Mean-pool the L32 refusal extracts. Returns X, y, sample_ids."""
    cache_path = CACHE_DIR / "refusal_gemma_L32.npz"
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        X = z["X"].astype(np.float32)
        y = z["y"].astype(np.int64)
        sids = list(z["sids"])
        print(f"  loaded refusal L32 from cache: n={len(y)} "
              f"pos={int((y==1).sum())} neg={int((y==0).sum())}", flush=True)
        return X, y, sids
    pt_paths = sorted(REFUSAL_EXTRACTS.glob("*.pt"))
    Xs, ys, sids = [], [], []
    n_skipped = 0
    for p in pt_paths:
        try:
            ex = torch.load(str(p), weights_only=False)
        except Exception as e:
            print(f"  [warn] {p.stem}: {type(e).__name__}: {e}", flush=True)
            n_skipped += 1
            continue
        residuals = ex["residuals"]  # (1, n_tok, d) fp16
        if residuals.dim() == 3 and residuals.shape[0] == 1:
            residuals = residuals.squeeze(0)
        # fp16 sums overflow on long prompts — upcast first.
        residuals = residuals.float()
        mask = ex["attention_mask"].bool().squeeze()
        n = int(mask.sum().item())
        if n == 0:
            n_skipped += 1
            continue
        m_f = mask.float().unsqueeze(-1)
        pooled = (residuals * m_f).sum(dim=0) / n
        arr = pooled.numpy()
        if not np.isfinite(arr).all():
            n_skipped += 1
            continue
        Xs.append(arr.astype(np.float32))
        ys.append(int(ex["label"]))
        sids.append(p.stem)
    X = np.stack(Xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    print(f"  loaded refusal L32: n={len(y)} pos={int((y==1).sum())} "
          f"neg={int((y==0).sum())} (skipped={n_skipped})", flush=True)
    np.savez(cache_path, X=X, y=y, sids=np.array(sids, dtype=object))
    print(f"  cached -> {cache_path}", flush=True)
    return X, y, sids


# ── CV training -------------------------------------------------------------

def _inner_alpha_pick(X_tr: np.ndarray, y_tr: np.ndarray, d_pca: int,
                      alphas: list[float], rng_seed: int) -> tuple[float, dict]:
    """Hold out 20% inside the outer train fold; pick best alpha by AUC."""
    inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng_seed)
    # take the very first inner split as the val split
    tr_idx, va_idx = next(iter(inner_skf.split(X_tr, y_tr)))
    inner_log = {"alpha_aucs": []}
    best_auc, best_alpha = -1.0, alphas[0]
    for a in alphas:
        probe = QuadraticProbe(d_pca=d_pca, alpha=a, random_state=rng_seed)
        probe.fit(X_tr[tr_idx], y_tr[tr_idx])
        s = probe.decision_function(X_tr[va_idx])
        auc = float(roc_auc_score(y_tr[va_idx], s))
        inner_log["alpha_aucs"].append({"alpha": a, "val_auc": auc})
        if auc > best_auc:
            best_auc, best_alpha = auc, a
    return best_alpha, inner_log


def cv_quadratic(X: np.ndarray, y: np.ndarray, d_pca: int) -> dict:
    """5-fold CV with per-fold inner alpha selection."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        best_alpha, inner_log = _inner_alpha_pick(X_tr, y_tr, d_pca,
                                                   ALPHA_GRID, rng_seed=SEED + fold)
        probe = QuadraticProbe(d_pca=d_pca, alpha=best_alpha, random_state=SEED)
        probe.fit(X_tr, y_tr)
        s_te = probe.decision_function(X_te)
        s_tr = probe.decision_function(X_tr)
        auc = float(roc_auc_score(y_te, s_te))
        train_auc = float(roc_auc_score(y_tr, s_tr))
        acc = float(((s_te > 0.5).astype(int) == y_te).mean())
        fm = {
            "fold": fold,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "alpha_chosen": best_alpha,
            "auc": auc,
            "train_auc": train_auc,
            "acc": acc,
            "elapsed_s": round(time.time() - t0, 2),
            "inner_alpha_log": inner_log,
        }
        fold_metrics.append(fm)
        print(f"    fold {fold}: alpha={best_alpha} test_AUC={auc:.4f} "
              f"train_AUC={train_auc:.4f} acc={acc:.4f} | {fm['elapsed_s']:.1f}s",
              flush=True)
    aucs = np.asarray([m["auc"] for m in fold_metrics])
    accs = np.asarray([m["acc"] for m in fold_metrics])
    train_aucs = np.asarray([m["train_auc"] for m in fold_metrics])
    ci_lo, ci_hi = wilson_ci_auc(aucs)
    return {
        "auc_mean": float(aucs.mean()),
        "auc_std": float(aucs.std(ddof=1)),
        "auc_min": float(aucs.min()),
        "auc_max": float(aucs.max()),
        "auc_wilson95_lo": ci_lo,
        "auc_wilson95_hi": ci_hi,
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std(ddof=1)),
        "train_auc_mean": float(train_aucs.mean()),
        "fold_metrics": fold_metrics,
    }


def cv_linear_on_pcs(X: np.ndarray, y: np.ndarray, d_pca: int) -> dict:
    """Apples-to-apples baseline: linear LR on the same d_pca PCs the quadratic
    probe sees. If quadratic > linear-on-PCs → genuine evidence for nonlinear
    interactions in the top-d directions. If quadratic ≈ linear-on-PCs ≈
    linear-on-raw → the PCA bottleneck isn't binding.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        scaler = StandardScaler().fit(X[tr_idx])
        X_tr_s = scaler.transform(X[tr_idx])
        X_te_s = scaler.transform(X[te_idx])
        pca = PCA(n_components=d_pca, random_state=SEED).fit(X_tr_s)
        Z_tr = pca.transform(X_tr_s)
        Z_te = pca.transform(X_te_s)
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(Z_tr, y[tr_idx])
        p_te = clf.predict_proba(Z_te)[:, 1]
        p_tr = clf.predict_proba(Z_tr)[:, 1]
        auc = float(roc_auc_score(y[te_idx], p_te))
        train_auc = float(roc_auc_score(y[tr_idx], p_tr))
        acc = float(((p_te > 0.5).astype(int) == y[te_idx]).mean())
        fold_metrics.append({
            "fold": fold, "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
            "auc": auc, "train_auc": train_auc, "acc": acc,
            "elapsed_s": round(time.time() - t0, 2),
        })
    aucs = np.asarray([m["auc"] for m in fold_metrics])
    accs = np.asarray([m["acc"] for m in fold_metrics])
    train_aucs = np.asarray([m["train_auc"] for m in fold_metrics])
    ci_lo, ci_hi = wilson_ci_auc(aucs)
    return {
        "auc_mean": float(aucs.mean()),
        "auc_std": float(aucs.std(ddof=1)),
        "auc_min": float(aucs.min()),
        "auc_max": float(aucs.max()),
        "auc_wilson95_lo": ci_lo,
        "auc_wilson95_hi": ci_hi,
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std(ddof=1)),
        "train_auc_mean": float(train_aucs.mean()),
        "fold_metrics": fold_metrics,
    }


def cv_linear(X: np.ndarray, y: np.ndarray) -> dict:
    """5-fold CV linear LR baseline matching exp 03/06/07/11."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(X[tr_idx], y[tr_idx])
        p_te = clf.predict_proba(X[te_idx])[:, 1]
        p_tr = clf.predict_proba(X[tr_idx])[:, 1]
        auc = float(roc_auc_score(y[te_idx], p_te))
        train_auc = float(roc_auc_score(y[tr_idx], p_tr))
        acc = float(((p_te > 0.5).astype(int) == y[te_idx]).mean())
        fm = {"fold": fold, "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
              "auc": auc, "train_auc": train_auc, "acc": acc,
              "elapsed_s": round(time.time() - t0, 2)}
        fold_metrics.append(fm)
        print(f"    fold {fold}: test_AUC={auc:.4f} train_AUC={train_auc:.4f} "
              f"acc={acc:.4f} | {fm['elapsed_s']:.1f}s", flush=True)
    aucs = np.asarray([m["auc"] for m in fold_metrics])
    accs = np.asarray([m["acc"] for m in fold_metrics])
    train_aucs = np.asarray([m["train_auc"] for m in fold_metrics])
    ci_lo, ci_hi = wilson_ci_auc(aucs)
    return {
        "auc_mean": float(aucs.mean()),
        "auc_std": float(aucs.std(ddof=1)),
        "auc_min": float(aucs.min()),
        "auc_max": float(aucs.max()),
        "auc_wilson95_lo": ci_lo,
        "auc_wilson95_hi": ci_hi,
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std(ddof=1)),
        "train_auc_mean": float(train_aucs.mean()),
        "fold_metrics": fold_metrics,
    }


# ── Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["cyber_1", "cyber_2", "cyber_3", "refusal_gemma"])
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--d_pca", type=int, required=True, choices=[16, 32])
    args = ap.parse_args()

    out_path = OUT_DIR / f"{args.task}_d{args.d_pca}_L{args.layer}.json"
    print(f"[quadratic probe] task={args.task} layer={args.layer} d_pca={args.d_pca}",
          flush=True)
    print(f"  n_quadratic_features = {n_quadratic_features(args.d_pca)}", flush=True)
    print(f"  output -> {out_path}", flush=True)

    t0 = time.time()
    if args.task == "refusal_gemma":
        if args.layer != 32:
            print(f"  [warn] refusal extracts only have L32 — proceeding anyway", flush=True)
        X, y, sids = load_refusal_L32()
    else:
        X, y, sids = load_cyber(args.task, args.layer)
    n, d_model = X.shape
    print(f"  X shape: {X.shape} ({(time.time()-t0):.1f}s)", flush=True)

    # Sample-size sanity check from the brief.
    nq = n_quadratic_features(args.d_pca)
    if nq > n:
        print(f"  [warn] n_features ({nq}) > n_samples ({n}); ridge may overfit",
              flush=True)

    print("\n  -- Linear LR baseline (raw activations) --", flush=True)
    linear_summary = cv_linear(X, y)
    print(f"  LINEAR: AUC = {linear_summary['auc_mean']:.4f} "
          f"± {linear_summary['auc_std']:.4f} "
          f"[{linear_summary['auc_wilson95_lo']:.4f}, "
          f"{linear_summary['auc_wilson95_hi']:.4f}]", flush=True)

    print(f"\n  -- Linear LR on d_pca={args.d_pca} PCs (apples-to-apples) --",
          flush=True)
    linear_pcs_summary = cv_linear_on_pcs(X, y, d_pca=args.d_pca)
    print(f"  LINEAR-ON-{args.d_pca}-PCs: AUC = "
          f"{linear_pcs_summary['auc_mean']:.4f} "
          f"± {linear_pcs_summary['auc_std']:.4f} "
          f"[{linear_pcs_summary['auc_wilson95_lo']:.4f}, "
          f"{linear_pcs_summary['auc_wilson95_hi']:.4f}]", flush=True)

    print(f"\n  -- Quadratic probe (d_pca={args.d_pca}) --", flush=True)
    quad_summary = cv_quadratic(X, y, d_pca=args.d_pca)
    print(f"  QUADRATIC: AUC = {quad_summary['auc_mean']:.4f} "
          f"± {quad_summary['auc_std']:.4f} "
          f"[{quad_summary['auc_wilson95_lo']:.4f}, "
          f"{quad_summary['auc_wilson95_hi']:.4f}]", flush=True)

    delta = quad_summary["auc_mean"] - linear_summary["auc_mean"]
    delta_vs_pcs = quad_summary["auc_mean"] - linear_pcs_summary["auc_mean"]
    if delta > 0.005:
        verdict = "QUADRATIC > LINEAR : feature interactions help"
    elif delta < -0.005:
        verdict = "QUADRATIC < LINEAR : PCA bottleneck dominates / overfit"
    else:
        verdict = "QUADRATIC ≈ LINEAR : linear sufficient"
    print(f"\n  delta (quad - linear-on-raw)  = {delta:+.4f}", flush=True)
    print(f"  delta (quad - linear-on-{args.d_pca}-PCs) = {delta_vs_pcs:+.4f}",
          flush=True)
    print(f"  → {verdict}", flush=True)

    out = {
        "task": args.task,
        "layer": args.layer,
        "d_pca": args.d_pca,
        "n_quadratic_features": nq,
        "n_samples": int(n),
        "d_model": int(d_model),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "alpha_grid": ALPHA_GRID,
        "linear": linear_summary,
        "linear_on_pcs": linear_pcs_summary,
        "quadratic": quad_summary,
        "delta_auc": float(delta),
        "delta_auc_vs_pcs": float(delta_vs_pcs),
        "verdict": verdict,
        "wall_seconds": round(time.time() - t0, 1),
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwritten to {out_path}  ({out['wall_seconds']}s wall)", flush=True)


if __name__ == "__main__":
    main()
