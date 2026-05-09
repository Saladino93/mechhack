"""D1 (extension) — TF-IDF baseline for cyber_2 and cyber_3.

Same vectoriser + classifier + CV as `tfidf_baseline.py` (which covered cyber_1),
but using exp 06's and exp 07's 1000-sample selections.

Question: how close does plain-text TF-IDF get to the 31B activation probe for
the harder tiers? If TF-IDF tracks closely (small delta), most of the signal is
surface text. If activations open a real gap, the model is encoding something
beyond bag-of-ngrams.

Outputs go into the same `results.json` under keys `D1_tfidf_cyber2` and
`D1_tfidf_cyber3` so this exp's full D1 picture is in one place.

CPU only.
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

OUT_DIR = Path(__file__).parent
RESULTS_PATH = OUT_DIR / "results.json"
METRICS_LOG = OUT_DIR / "metrics.jsonl"

SEED = 0
N_FOLDS = 5

TASKS = [
    {
        "task": "cyber_2",
        "selection": REPO_ROOT / "experiments" / "06_cyber2_extract_omar" / "selection.json",
        "act_results": REPO_ROOT / "experiments" / "06_cyber2_extract_omar" / "results.json",
        "results_key": "D1_tfidf_cyber2",
    },
    {
        "task": "cyber_3",
        "selection": REPO_ROOT / "experiments" / "07_cyber3_extract_omar" / "selection.json",
        "act_results": REPO_ROOT / "experiments" / "07_cyber3_extract_omar" / "results.json",
        "results_key": "D1_tfidf_cyber3",
    },
]


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


def best_activation_auc(act_results_path, task):
    """Return (auc, layer, pooling) for the best activation result, or None."""
    if not act_results_path.exists():
        return None
    try:
        d = json.loads(act_results_path.read_text())
        per_pool = d.get("tasks", {}).get(task, {}).get("per_pooling", {})
        best = None
        for pname, rows in per_pool.items():
            for r in rows:
                if best is None or r["auc_mean"] > best[0]:
                    best = (r["auc_mean"], r["layer"], pname)
        return best
    except Exception:
        return None


def run_one(spec):
    task = spec["task"]
    sel_path = spec["selection"]
    print(f"\n[D1] TF-IDF baseline on {task} ({sel_path.parent.name}'s selection)", flush=True)

    sel = json.loads(sel_path.read_text())
    sel_ids = [row["sample_id"] for row in sel["samples"]]
    print(f"  selection: {len(sel_ids)} sample_ids", flush=True)

    rows = {s["sample_id"]: s for s in load_dataset("cyber", split="train")}

    prompts, labels = [], []
    for sid in sel_ids:
        s = rows.get(sid)
        if s is None:
            continue
        lbl = get_label_for_task(s, task)
        if lbl is None:
            continue
        prompts.append(s["prompt"])
        labels.append(lbl)

    y = np.asarray(labels, dtype=np.int64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(f"  loaded {len(prompts)} prompts (pos={n_pos}, neg={n_neg})", flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(prompts, y)):
        t0 = time.time()
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
        append_jsonl(METRICS_LOG, {"diagnostic": spec["results_key"], **fm})
        print(f"  fold {fold}: test AUC={auc:.4f} acc={acc:.4f} | "
              f"train AUC={auc_tr:.4f} | n_feat={n_features} | {elapsed:.1f}s", flush=True)

    aucs = np.asarray([m["auc"] for m in fold_metrics])
    accs = np.asarray([m["acc"] for m in fold_metrics])
    train_aucs = np.asarray([m["train_auc"] for m in fold_metrics])
    summary = {
        "task": task,
        "selection": str(sel_path),
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

    best = best_activation_auc(spec["act_results"], task)
    if best is not None:
        auc_a, layer_a, pool_a = best
        summary["activation_best_auc"] = auc_a
        summary["activation_best_layer"] = layer_a
        summary["activation_best_pooling"] = pool_a
        summary["delta_vs_activation"] = auc_a - summary["auc_mean"]
        cmp_line = (f"  activation best (so far): AUC={auc_a:.4f} "
                    f"@ layer {layer_a}, pooling={pool_a}  →  delta = {auc_a - summary['auc_mean']:+.4f}")
    else:
        cmp_line = "  (activation results not yet available — comparison skipped)"

    print(f"\n  === {task} D1 SUMMARY ===")
    print(f"  TF-IDF AUC: {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f} "
          f"(range {summary['auc_min']:.4f}-{summary['auc_max']:.4f})")
    print(f"  TF-IDF acc: {summary['acc_mean']:.4f} ± {summary['acc_std']:.4f}")
    print(cmp_line)

    results = load_results()
    results[spec["results_key"]] = summary
    atomic_write_json(RESULTS_PATH, results)
    print(f"  written to {RESULTS_PATH} (key={spec['results_key']})")


def main():
    for spec in TASKS:
        run_one(spec)


if __name__ == "__main__":
    main()
