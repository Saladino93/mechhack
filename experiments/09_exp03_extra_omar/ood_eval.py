"""D3 — OOD held-out evaluation for cyber_1.

Refit the probe on exp 03's 999 train extracts, evaluate on a 500-sample subset
of the cyber-test split (250 dual_use + 250 benign, stratified, seed=42).
Zero overlap between exp 03's selection and the test split — verified before
running.

For each (pooling, layer) in {mean, last_token} × {25, 30, 35, 40, 45}:
  - Fit LogisticRegression(C=1.0, max_iter=2000, lbfgs) on the 999 train rows.
  - Predict probabilities on the 500 OOD test rows.
  - Test AUC + 1000-bootstrap 95% CI on test predictions.
  - Pull CV AUC from exp 03's results.json for the same (pooling, layer).
  - Headline: (CV AUC, test AUC, gap = CV - test).

Writes results to `results.json` under key `D3_ood`. Also writes
`ood_selection.json` and a `split_manifest.json` (sample_id -> "train"/"test"
for every cyber_1-eligible sample), the latter also copied to
`/home/ubuntu/extracts/cyber_all_omar/split_manifest.json` so future
experiments can tell train from test inside the shared extracts folder.

CPU only.
"""
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments" / "02_extract_activations"))
from data import get_label_for_task, load_dataset  # noqa: E402

EXP03_SEL = REPO_ROOT / "experiments" / "03_layer_sweep_omar" / "selection.json"
EXP03_RESULTS = REPO_ROOT / "experiments" / "03_layer_sweep_omar" / "results.json"
EXP08_SEL = REPO_ROOT / "experiments" / "08_layer_sweep_3000_omar" / "selection.json"
EXTRACTS_DIR = Path("/home/ubuntu/extracts/cyber_all_omar")

OUT_DIR = Path(__file__).parent
RESULTS_PATH = OUT_DIR / "results.json"
METRICS_LOG = OUT_DIR / "metrics.jsonl"
OOD_SEL_PATH = OUT_DIR / "ood_selection.json"
MANIFEST_PATH = OUT_DIR / "split_manifest.json"
SHARED_MANIFEST_PATH = EXTRACTS_DIR / "split_manifest.json"

TASK = "cyber_1"
N_PER_CLASS = 250
SEED = 42
LAYERS_TO_REPORT = [25, 30, 35, 40, 45]
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 0.95


def append_jsonl(path: Path, obj):
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


def atomic_write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def load_results():
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {}


def pick_ood_test_rows():
    """Stratified 250+250 dual_use+benign sample from exp 08 test split."""
    sel08 = json.loads(EXP08_SEL.read_text())
    eligible = [s for s in sel08["samples"]
                if s["split"] == "test" and s["label"] in ("dual_use", "benign")]

    # Verify zero overlap with exp 03 train selection.
    sel03 = json.loads(EXP03_SEL.read_text())
    train_ids = {s["sample_id"] for s in sel03["samples"]}
    overlap = [s for s in eligible if s["sample_id"] in train_ids]
    if overlap:
        raise RuntimeError(f"unexpected overlap: {len(overlap)} sample_ids "
                           f"are in both exp 03 train and exp 08 test")

    by_label = {"dual_use": [], "benign": []}
    for s in eligible:
        by_label[s["label"]].append(s)
    print(f"  cyber_1-eligible test pool: dual_use={len(by_label['dual_use'])}, "
          f"benign={len(by_label['benign'])}", flush=True)

    rng = random.Random(SEED)
    chosen = []
    for label in ("dual_use", "benign"):
        if len(by_label[label]) < N_PER_CLASS:
            raise RuntimeError(f"not enough {label} test rows: "
                               f"have {len(by_label[label])}, need {N_PER_CLASS}")
        chosen.extend(rng.sample(by_label[label], N_PER_CLASS))
    rng.shuffle(chosen)
    return chosen


def write_split_manifest(test_ids):
    """Write sample_id -> "train"/"test" for every cyber_1-eligible row in
    {exp 03 train} ∪ {exp 08 test}. Copies to shared extracts dir too."""
    manifest = {}
    sel03 = json.loads(EXP03_SEL.read_text())
    for s in sel03["samples"]:
        manifest[s["sample_id"]] = "train"
    for sid in test_ids:
        manifest[sid] = "test"
    atomic_write_json(MANIFEST_PATH, manifest)
    shutil.copy2(MANIFEST_PATH, SHARED_MANIFEST_PATH)
    print(f"  wrote split manifest with {len(manifest)} entries "
          f"(also copied to {SHARED_MANIFEST_PATH})", flush=True)


def pool_one(p):
    ex = torch.load(str(p), weights_only=False)
    # Pool in fp16 to avoid the ~6GB fp32 upcast on the longest prompts;
    # cast the small (n_layers, d) result up to fp32 once at the end.
    residuals = ex["residuals"]  # fp16
    mask = ex["attention_mask"].bool()
    n_kept = int(mask.sum().item())
    if n_kept == 0:
        return None
    m_f = mask.to(residuals.dtype).unsqueeze(0).unsqueeze(-1)
    mean_pooled = (residuals * m_f).sum(dim=1) / n_kept
    last_idx = int(mask.nonzero().max().item())
    last_pooled = residuals[:, last_idx, :]
    return (mean_pooled.float().numpy(),
            last_pooled.float().numpy(),
            list(ex["layer_idxs"]))


def load_pooled(sample_ids, label_fn=None):
    """Returns X_mean (N, n_layers, d), X_last (N, n_layers, d), y (N,), layer_idxs."""
    pt_paths = [EXTRACTS_DIR / f"{sid}.pt" for sid in sample_ids]
    missing = [(sid, p) for sid, p in zip(sample_ids, pt_paths) if not p.exists()]
    if missing:
        print(f"  [warn] {len(missing)} extracts missing in {EXTRACTS_DIR} "
              f"(first: {missing[0][0]}); using remaining", flush=True)

    means, lasts, ys, ids_kept = [], [], [], []
    layer_idxs = None
    for i, (sid, p) in enumerate(zip(sample_ids, pt_paths)):
        if not p.exists():
            continue
        result = pool_one(p)
        if result is None:
            continue
        m, l, lidx = result
        means.append(m)
        lasts.append(l)
        if layer_idxs is None:
            layer_idxs = lidx
        if label_fn is not None:
            ys.append(label_fn(sid))
        ids_kept.append(sid)
        if (i + 1) % 100 == 0:
            print(f"    pooled {i+1}/{len(pt_paths)}", flush=True)

    X_mean = np.stack(means, axis=0)
    X_last = np.stack(lasts, axis=0)
    y = np.asarray(ys, dtype=np.int64) if label_fn is not None else None
    return X_mean, X_last, y, ids_kept, layer_idxs


def bootstrap_auc_ci(y, p, n_bootstrap, ci, rng):
    n = len(y)
    aucs = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            aucs[b] = 0.5
            continue
        aucs[b] = roc_auc_score(y[idx], p[idx])
    lo = np.quantile(aucs, (1 - ci) / 2)
    hi = np.quantile(aucs, 1 - (1 - ci) / 2)
    return float(lo), float(hi)


def get_cv_auc(pooling, layer):
    if not EXP03_RESULTS.exists():
        return None
    d = json.loads(EXP03_RESULTS.read_text())
    rows = d.get("tasks", {}).get(TASK, {}).get("per_pooling", {}).get(pooling, [])
    for r in rows:
        if r["layer"] == layer:
            return r["auc_mean"]
    return None


def main():
    print(f"[D3] OOD held-out evaluation for {TASK}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Pick OOD test rows
    print("\nPicking OOD test rows...", flush=True)
    test_rows = pick_ood_test_rows()
    test_ids = [s["sample_id"] for s in test_rows]
    print(f"  picked {len(test_ids)} test sample_ids "
          f"({sum(1 for s in test_rows if s['label']=='dual_use')} du, "
          f"{sum(1 for s in test_rows if s['label']=='benign')} ben)", flush=True)
    atomic_write_json(OOD_SEL_PATH, {
        "task": TASK, "seed": SEED, "n_per_class": N_PER_CLASS,
        "samples": [{"sample_id": s["sample_id"], "label": s["label"]} for s in test_rows],
    })
    print(f"  wrote {OOD_SEL_PATH}", flush=True)

    write_split_manifest(test_ids)

    # 2. Verify extracts exist
    print("\nVerifying extracts...", flush=True)
    train_sel = json.loads(EXP03_SEL.read_text())
    train_ids = [s["sample_id"] for s in train_sel["samples"]]
    train_missing = [sid for sid in train_ids if not (EXTRACTS_DIR / f"{sid}.pt").exists()]
    test_missing = [sid for sid in test_ids if not (EXTRACTS_DIR / f"{sid}.pt").exists()]
    print(f"  train: {len(train_ids) - len(train_missing)}/{len(train_ids)} present", flush=True)
    print(f"  test : {len(test_ids) - len(test_missing)}/{len(test_ids)} present", flush=True)
    if test_missing:
        print(f"  [warn] {len(test_missing)} test extracts missing — first: {test_missing[:3]}", flush=True)

    # 3. Load + pool features
    rows_by_id = {s["sample_id"]: s for s in load_dataset("cyber", split="train")}
    rows_by_id_test = {s["sample_id"]: s for s in load_dataset("cyber", split="test")}

    def label_fn_train(sid):
        s = rows_by_id.get(sid)
        return get_label_for_task(s, TASK) if s else None

    def label_fn_test(sid):
        s = rows_by_id_test.get(sid)
        return get_label_for_task(s, TASK) if s else None

    print("\nLoading + pooling train extracts...", flush=True)
    t0 = time.time()
    X_tr_mean, X_tr_last, y_tr, train_ids_kept, layer_idxs = load_pooled(train_ids, label_fn_train)
    # Filter out None labels
    keep = [i for i, lbl in enumerate(y_tr) if lbl is not None and lbl in (0, 1)]
    X_tr_mean = X_tr_mean[keep]; X_tr_last = X_tr_last[keep]; y_tr = y_tr[keep]
    print(f"  loaded {len(y_tr)} train rows in {time.time()-t0:.1f}s "
          f"(pos={int((y_tr==1).sum())}, neg={int((y_tr==0).sum())})", flush=True)

    print("\nLoading + pooling test extracts...", flush=True)
    t0 = time.time()
    X_te_mean, X_te_last, y_te, test_ids_kept, layer_idxs2 = load_pooled(test_ids, label_fn_test)
    keep = [i for i, lbl in enumerate(y_te) if lbl is not None and lbl in (0, 1)]
    X_te_mean = X_te_mean[keep]; X_te_last = X_te_last[keep]; y_te = y_te[keep]
    print(f"  loaded {len(y_te)} test rows in {time.time()-t0:.1f}s "
          f"(pos={int((y_te==1).sum())}, neg={int((y_te==0).sum())})", flush=True)

    if layer_idxs is None or layer_idxs2 != layer_idxs:
        print(f"  [warn] layer index mismatch or missing: train={layer_idxs}, test={layer_idxs2}", flush=True)

    # 4. Per (pooling, layer) train + evaluate
    print(f"\nFitting probes (LR C=1.0) on {len(LAYERS_TO_REPORT)} layers × 2 poolings", flush=True)
    pool_arrays = {
        "mean": (X_tr_mean, X_te_mean),
        "last_token": (X_tr_last, X_te_last),
    }
    rng = np.random.default_rng(SEED)
    rows = []
    for pname, (X_tr, X_te) in pool_arrays.items():
        for layer in LAYERS_TO_REPORT:
            try:
                li = layer_idxs.index(layer)
            except ValueError:
                print(f"  [skip] layer {layer} not in extracts", flush=True)
                continue
            X_tr_l = X_tr[:, li, :]
            X_te_l = X_te[:, li, :]

            t0 = time.time()
            clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
            clf.fit(X_tr_l, y_tr)
            p_te = clf.predict_proba(X_te_l)[:, 1]
            test_auc = float(roc_auc_score(y_te, p_te))
            test_acc = float(((p_te > 0.5).astype(int) == y_te).mean())
            ci_lo, ci_hi = bootstrap_auc_ci(y_te, p_te, N_BOOTSTRAP, BOOTSTRAP_CI, rng)
            cv_auc = get_cv_auc(pname, layer)
            gap = (cv_auc - test_auc) if cv_auc is not None else None
            elapsed = time.time() - t0

            row = {
                "pooling": pname,
                "layer": layer,
                "test_auc": test_auc,
                "test_acc": test_acc,
                "test_auc_ci95": [ci_lo, ci_hi],
                "cv_auc_exp03": cv_auc,
                "cv_minus_test_auc": gap,
                "n_train": int(len(y_tr)),
                "n_test": int(len(y_te)),
                "elapsed_s": round(elapsed, 2),
            }
            rows.append(row)
            append_jsonl(METRICS_LOG, {"diagnostic": "D3_ood", **row})
            gap_str = f"{gap:+.4f}" if gap is not None else "n/a"
            cv_str = f"{cv_auc:.4f}" if cv_auc is not None else "n/a"
            print(f"  {pname:>10} L{layer:>2}: test AUC={test_auc:.4f} "
                  f"CI95=[{ci_lo:.4f},{ci_hi:.4f}] | acc={test_acc:.4f} | "
                  f"cv={cv_str} | gap={gap_str} | {elapsed:.1f}s", flush=True)

    # 5. Find headline at mean-pool L40 + the layer with smallest gap
    headline = next((r for r in rows if r["pooling"] == "mean" and r["layer"] == 40), None)
    rows_with_gap = [r for r in rows if r["cv_minus_test_auc"] is not None]
    smallest_gap = min(rows_with_gap, key=lambda r: abs(r["cv_minus_test_auc"])) if rows_with_gap else None
    largest_test_auc = max(rows, key=lambda r: r["test_auc"]) if rows else None

    summary = {
        "task": TASK,
        "n_per_class": N_PER_CLASS,
        "seed": SEED,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "layers": LAYERS_TO_REPORT,
        "n_bootstrap": N_BOOTSTRAP,
        "ci": BOOTSTRAP_CI,
        "rows": rows,
        "headline_mean_L40": headline,
        "smallest_cv_test_gap": smallest_gap,
        "largest_test_auc": largest_test_auc,
    }

    print("\n  === D3 OOD SUMMARY ===")
    if headline:
        gap_str = f"{headline['cv_minus_test_auc']:+.4f}" if headline['cv_minus_test_auc'] is not None else "n/a"
        cv_str = f"{headline['cv_auc_exp03']:.4f}" if headline['cv_auc_exp03'] is not None else "n/a"
        print(f"  mean-pool L40: test AUC={headline['test_auc']:.4f} CI95={headline['test_auc_ci95']} "
              f"vs CV={cv_str} (gap={gap_str})")
    if largest_test_auc:
        print(f"  largest test AUC: {largest_test_auc['test_auc']:.4f} @ "
              f"{largest_test_auc['pooling']} L{largest_test_auc['layer']}")
    if smallest_gap:
        print(f"  smallest |gap|: {abs(smallest_gap['cv_minus_test_auc']):.4f} @ "
              f"{smallest_gap['pooling']} L{smallest_gap['layer']} "
              f"(cv={smallest_gap['cv_auc_exp03']:.4f}, test={smallest_gap['test_auc']:.4f})")

    results = load_results()
    results["D3_ood"] = summary
    atomic_write_json(RESULTS_PATH, results)
    print(f"\n  written to {RESULTS_PATH} (key=D3_ood)")


if __name__ == "__main__":
    main()
