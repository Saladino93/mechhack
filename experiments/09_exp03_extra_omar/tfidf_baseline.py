"""D1 — TF-IDF baseline on the same 999 cyber_1 prompts as exp 03.

Headline question: how close does plain-text TF-IDF + logistic regression get
to exp 03's 0.983? If TF-IDF is already at 0.95+, then most of the signal is
in the surface text and the 31B activations are barely helping.

Forces CPU only (CUDA_VISIBLE_DEVICES="").
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments" / "02_extract_activations"))
from data import get_label_for_task, load_dataset  # noqa: E402

EXP03_SEL = REPO_ROOT / "experiments" / "03_layer_sweep_omar" / "selection.json"
OUT_DIR = Path(__file__).parent
RESULTS_PATH = OUT_DIR / "results.json"
METRICS_LOG = OUT_DIR / "metrics.jsonl"

SEED = 0
N_FOLDS = 5
TASK = "cyber_1"


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


def main():
    print("[D1] TF-IDF baseline on exp 03's 999 cyber_1 samples", flush=True)

    sel = json.loads(EXP03_SEL.read_text())
    sel_ids = [row["sample_id"] for row in sel["samples"]]
    print(f"  selection: {len(sel_ids)} sample_ids from exp 03", flush=True)

    # Load all cyber-train, then index by sample_id.
    rows = {s["sample_id"]: s for s in load_dataset("cyber", split="train")}

    prompts, labels = [], []
    for sid in sel_ids:
        s = rows.get(sid)
        if s is None:
            continue
        lbl = get_label_for_task(s, TASK)
        if lbl is None:
            continue
        prompts.append(s["prompt"])
        labels.append(lbl)

    y = np.asarray(labels, dtype=np.int64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(f"  loaded {len(prompts)} prompts (pos={n_pos}, neg={n_neg})", flush=True)
    print(f"  example prompt (first 120 chars): {prompts[0][:120]!r}", flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(prompts, y)):
        t0 = time.time()
        # Re-fit the vectoriser per fold (no leakage from test prompts).
        vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=20000)
        X_tr = vec.fit_transform([prompts[i] for i in tr_idx])
        X_te = vec.transform([prompts[i] for i in te_idx])
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(X_tr, y[tr_idx])
        p_te = clf.predict_proba(X_te)[:, 1]
        p_tr = clf.predict_proba(X_tr)[:, 1]
        auc = float(roc_auc_score(y[te_idx], p_te))
        acc = float(((p_te > 0.5).astype(int) == y[te_idx]).mean())
        auc_tr = float(roc_auc_score(y[tr_idx], p_tr))
        acc_tr = float(((p_tr > 0.5).astype(int) == y[tr_idx]).mean())
        n_features = int(X_tr.shape[1])
        elapsed = time.time() - t0
        fm = {
            "fold": fold,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "n_features": n_features,
            "auc": auc, "acc": acc,
            "train_auc": auc_tr, "train_acc": acc_tr,
            "elapsed_s": round(elapsed, 2),
        }
        fold_metrics.append(fm)
        append_jsonl(METRICS_LOG, {"diagnostic": "D1_tfidf", **fm})
        print(f"  fold {fold}: test AUC={auc:.4f} acc={acc:.4f} | "
              f"train AUC={auc_tr:.4f} | n_feat={n_features} | {elapsed:.1f}s", flush=True)

    aucs = np.asarray([m["auc"] for m in fold_metrics])
    accs = np.asarray([m["acc"] for m in fold_metrics])
    train_aucs = np.asarray([m["train_auc"] for m in fold_metrics])
    summary = {
        "task": TASK,
        "selection": str(EXP03_SEL),
        "n_samples": int(len(prompts)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "vectorizer": {"min_df": 2, "ngram_range": [1, 2], "max_features": 20000},
        "classifier": {"C": 1.0, "solver": "lbfgs", "max_iter": 2000},
        "n_folds": N_FOLDS,
        "seed": SEED,
        "auc_mean": float(aucs.mean()),
        "auc_std": float(aucs.std(ddof=1)),
        "auc_min": float(aucs.min()),
        "auc_max": float(aucs.max()),
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std(ddof=1)),
        "train_auc_mean": float(train_aucs.mean()),
        "fold_metrics": fold_metrics,
    }

    # Compare to exp 03.
    exp03 = json.loads((REPO_ROOT / "experiments" / "03_layer_sweep_omar" / "results.json").read_text())
    best_exp03 = max(
        exp03["tasks"]["cyber_1"]["per_pooling"]["mean"], key=lambda r: r["auc_mean"]
    )
    summary["exp03_best_mean_auc"] = best_exp03["auc_mean"]
    summary["exp03_best_layer_mean"] = best_exp03["layer"]
    summary["delta_vs_exp03"] = best_exp03["auc_mean"] - summary["auc_mean"]

    print("\n  === D1 SUMMARY ===")
    print(f"  TF-IDF AUC: {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f} "
          f"(range {summary['auc_min']:.4f}-{summary['auc_max']:.4f})")
    print(f"  TF-IDF acc: {summary['acc_mean']:.4f} ± {summary['acc_std']:.4f}")
    print(f"  exp03 best mean-pool AUC (layer {best_exp03['layer']}): "
          f"{best_exp03['auc_mean']:.4f}")
    print(f"  delta (exp03 − TF-IDF): {summary['delta_vs_exp03']:+.4f}")

    results = load_results()
    results["D1_tfidf"] = summary
    atomic_write_json(RESULTS_PATH, results)
    print(f"  written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
