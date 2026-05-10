"""TF-IDF baseline for Refusal-Qwen.

Trains LR on TF-IDF of `attack_prompt` text alone (no model internals).
Establishes the floor any activation-based probe must beat. Same dataset
splits and bootstrap method as the activation probes.

Two variants reported:
  - word 1-2gram (max_features=20 000)
  - char 3-5gram (max_features=50 000, sublinear_tf=True)

Inner C-sweep on a 20% inner-val of train, refit on full train at the
selected C, evaluate on test. 1000-bootstrap 95% CI on test AUC.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLES_FILE = REPO_ROOT / "datasets/refusal_probes/qwen36/attacks_full.jsonl"
OUT_DIR = Path(__file__).parent

samples = [json.loads(l) for l in open(SAMPLES_FILE)]
samples = [s for s in samples if s.get("is_refusal") is not None]

train = [s for s in samples if s["split"] == "train"]
test  = [s for s in samples if s["split"] == "test"]
y_train = np.array([int(s["is_refusal"]) for s in train])
y_test  = np.array([int(s["is_refusal"]) for s in test])
X_train_text = [s["attack_prompt"] for s in train]
X_test_text  = [s["attack_prompt"] for s in test]
print(f"train {len(train)} ({y_train.mean():.3f} pos)  test {len(test)} ({y_test.mean():.3f} pos)")


def auc_with_ci(y_true, y_score, n_boot=1000, seed=0):
    auc = float(roc_auc_score(y_true, y_score))
    rng = np.random.default_rng(seed)
    n = len(y_true); aucs = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, size=n)
        if len(set(y_true[ix].tolist())) < 2: continue
        aucs.append(roc_auc_score(y_true[ix], y_score[ix]))
    return auc, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def run(variant_name, vectorizer):
    Xtr = vectorizer.fit_transform(X_train_text)
    Xte = vectorizer.transform(X_test_text)
    # inner val for C sweep
    Xi_tr, Xi_va, yi_tr, yi_va = train_test_split(
        Xtr, y_train, test_size=0.2, stratify=y_train, random_state=0)
    Cs = [0.01, 0.1, 1.0, 10.0]
    inner = {}
    for C in Cs:
        lr = LogisticRegression(solver="lbfgs", max_iter=2000, C=C)
        lr.fit(Xi_tr, yi_tr)
        inner[C] = float(roc_auc_score(yi_va, lr.predict_proba(Xi_va)[:, 1]))
    bestC = max(Cs, key=lambda c: inner[c])
    lr = LogisticRegression(solver="lbfgs", max_iter=2000, C=bestC)
    lr.fit(Xtr, y_train)
    proba = lr.predict_proba(Xte)[:, 1]
    auc, lo, hi = auc_with_ci(y_test, proba)
    print(f"  {variant_name}: C={bestC} AUC {auc:.4f} [{lo:.4f},{hi:.4f}]  "
          f"inner-val: {inner}")
    return {"variant": variant_name, "best_C": bestC,
            "test_auc": auc, "ci95": [lo, hi],
            "inner_val_aucs": inner,
            "n_features": int(Xtr.shape[1])}


results = {}
results["word_1_2gram"] = run("word_1_2gram",
    TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                    max_features=20_000, sublinear_tf=True))
results["char_3_5gram"] = run("char_3_5gram",
    TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                    max_features=50_000, sublinear_tf=True))

# Merge into existing results.json
res_path = OUT_DIR / "results.json"
all_res = json.loads(res_path.read_text()) if res_path.exists() else {}
all_res.setdefault("baseline_tfidf", {})
all_res["baseline_tfidf"].update(results)
res_path.write_text(json.dumps(all_res, indent=2))
print(f"Saved to {res_path}")
