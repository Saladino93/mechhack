"""TF-IDF + naive (random / majority-class) baselines on refusal.

Establishes the floor for refusal AUC. Same 5-fold split (seed=0) as all probes.

Outputs: refusal_baselines.json with:
  - random_baseline (AUC = 0.5 by construction, sanity)
  - majority_class accuracy (constant predictor)
  - tfidf_lr (TfidfVectorizer + LogisticRegression with C-sweep)
  - tfidf_lr_word_char (combined word + char-3-5 n-gram features)
"""
from __future__ import annotations
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
from sklearn.pipeline import FeatureUnion

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
ATTRS_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"
OUT = HERE / "results" / "refusal_baselines.json"

N_FOLDS = 5
SEED = 0
C_GRID = [0.01, 0.1, 1.0, 10.0]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with ATTRS_FULL.open() as f:
        for line in f:
            r = json.loads(line)
            if "attack_prompt" not in r or "is_refusal" not in r:
                continue
            if r["is_refusal"] is None:
                continue
            rows.append((r["sample_id"], r["attack_prompt"], int(bool(r["is_refusal"]))))
    sids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    y = np.asarray([r[2] for r in rows], dtype=np.int64)
    n = len(y)

    print(f"[refusal naive baselines] n={n} pos={int((y==1).sum())} neg={int((y==0).sum())}",
          flush=True)

    out = {"task": "refusal_gemma", "n_samples": int(n),
           "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
           "n_folds": N_FOLDS, "seed": SEED, "C_grid": C_GRID}

    # 1. Majority class accuracy (constant predictor); AUC for any constant predictor = 0.5
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    maj_accs = []
    for fold, (tr, te) in enumerate(skf.split(np.zeros(n), y)):
        majority = int(np.bincount(y[tr]).argmax())
        acc = float(((y[te] == majority).astype(int)).mean())
        maj_accs.append(acc)
    out["majority_baseline"] = {"acc_mean": float(np.mean(maj_accs)),
                                 "acc_std": float(np.std(maj_accs, ddof=1)),
                                 "auc": 0.5,
                                 "fold_accs": maj_accs}
    print(f"  majority: acc={out['majority_baseline']['acc_mean']:.4f} ± "
          f"{out['majority_baseline']['acc_std']:.4f} (AUC=0.5 by construction)",
          flush=True)

    # 2. Random predictor (theoretical AUC=0.5)
    rng = np.random.default_rng(SEED)
    rand_aucs = []
    for fold, (tr, te) in enumerate(skf.split(np.zeros(n), y)):
        s = rng.standard_normal(len(te))
        rand_aucs.append(float(roc_auc_score(y[te], s)))
    out["random_baseline"] = {"auc_mean": float(np.mean(rand_aucs)),
                               "auc_std": float(np.std(rand_aucs, ddof=1)),
                               "fold_aucs": rand_aucs}
    print(f"  random Gaussian: AUC={out['random_baseline']['auc_mean']:.4f}", flush=True)

    # 3. TF-IDF word
    def cv_tfidf(vectorizer_factory):
        skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_aucs, fold_train_aucs, fold_cs = [], [], []
        for fold, (tr, te) in enumerate(skf2.split(np.zeros(n), y)):
            v = vectorizer_factory()
            X_tr = v.fit_transform([texts[i] for i in tr])
            X_te = v.transform([texts[i] for i in te])
            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + fold)
            itr, iva = next(iter(inner.split(np.zeros(len(tr)), y[tr])))
            best_c, best_auc = C_GRID[0], -1.0
            for c in C_GRID:
                clf = LogisticRegression(C=c, max_iter=2000, solver="liblinear").fit(X_tr[itr], y[tr][itr])
                a = roc_auc_score(y[tr][iva], clf.predict_proba(X_tr[iva])[:, 1])
                if a > best_auc: best_auc, best_c = a, c
            clf = LogisticRegression(C=best_c, max_iter=2000, solver="liblinear").fit(X_tr, y[tr])
            fold_aucs.append(roc_auc_score(y[te], clf.predict_proba(X_te)[:, 1]))
            fold_train_aucs.append(roc_auc_score(y[tr], clf.predict_proba(X_tr)[:, 1]))
            fold_cs.append(best_c)
        return {"auc_mean": float(np.mean(fold_aucs)),
                "auc_std": float(np.std(fold_aucs, ddof=1)),
                "fold_aucs": fold_aucs,
                "train_auc_mean": float(np.mean(fold_train_aucs)),
                "fold_cs": fold_cs}

    print("\n  TF-IDF (word 1-2 gram, max_features=5000)...", flush=True)
    t0 = time.time()
    out["tfidf_word_lr"] = cv_tfidf(lambda: TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), max_features=5000, min_df=2))
    out["tfidf_word_lr"]["wall_seconds"] = round(time.time() - t0, 1)
    r = out["tfidf_word_lr"]
    print(f"    AUC={r['auc_mean']:.4f} ± {r['auc_std']:.4f} "
          f"train={r['train_auc_mean']:.3f} ({r['wall_seconds']:.0f}s)", flush=True)
    OUT.write_text(json.dumps(out, indent=2))

    print("\n  TF-IDF (char 3-5 gram, max_features=20000)...", flush=True)
    t0 = time.time()
    out["tfidf_char_lr"] = cv_tfidf(lambda: TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), max_features=20000, min_df=2))
    out["tfidf_char_lr"]["wall_seconds"] = round(time.time() - t0, 1)
    r = out["tfidf_char_lr"]
    print(f"    AUC={r['auc_mean']:.4f} ± {r['auc_std']:.4f} "
          f"train={r['train_auc_mean']:.3f} ({r['wall_seconds']:.0f}s)", flush=True)
    OUT.write_text(json.dumps(out, indent=2))

    print(f"\n  Probe (Gemma L40 mean) reference: 0.9445  → activation lift over TF-IDF char = "
          f"{0.9445 - out['tfidf_char_lr']['auc_mean']:+.3f}", flush=True)

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
