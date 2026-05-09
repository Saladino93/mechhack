"""Compute similarity of each of the 7 rewrites to its original prompt.

For every (sample_id, rewrite_idx):
  - char_length_ratio = len(rewrite) / len(original)
  - tfidf_cosine = sklearn cosine similarity on TF-IDF n-gram features (1-2 grams,
    fit on the union of the original + all 7 rewrites for THAT sample)
  - jaccard_word = |words(rewrite) ∩ words(original)| / |union|

Output:
  - similarity.jsonl : one row per (sample_id, rewrite_idx)
  - summary.json     : per-similarity-metric distribution (mean, std, percentiles)

Tells us how aggressively DeepSeek paraphrased: high cosine = mostly the same;
low cosine = significant rewrite. We'll cross-reference with PRE probe scores
to see whether small or large edits flip the probe.

CPU only.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REWRITES = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "rewrites_k7.json"
ORIGINALS = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"
OUT = Path(__file__).parent
OUT_JSONL = OUT / "similarity.jsonl"
OUT_SUMMARY = OUT / "similarity_summary.json"


def jaccard_words(a, b):
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def main():
    rewrites = json.loads(REWRITES.read_text())
    originals = {}
    with ORIGINALS.open() as f:
        for line in f:
            r = json.loads(line)
            originals[r["sample_id"]] = r["attack_prompt"]

    rows = []
    for entry in rewrites:
        sid = entry["sample_id"]
        orig = originals.get(sid)
        if orig is None:
            continue
        rws = entry["rewrites"]

        vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2), max_features=20000)
        all_texts = [orig] + rws
        X = vec.fit_transform(all_texts)
        cos = cosine_similarity(X[0:1], X[1:]).flatten()

        for j, rw in enumerate(rws):
            rows.append({
                "sample_id": sid,
                "rewrite_idx": j,
                "char_len_orig": len(orig),
                "char_len_rewrite": len(rw),
                "char_len_ratio": len(rw) / max(1, len(orig)),
                "tfidf_cosine": float(cos[j]),
                "jaccard_word": jaccard_words(orig, rw),
            })

    with OUT_JSONL.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    cos = np.array([r["tfidf_cosine"] for r in rows])
    jac = np.array([r["jaccard_word"] for r in rows])
    rat = np.array([r["char_len_ratio"] for r in rows])
    summary = {
        "n_rewrites": len(rows),
        "tfidf_cosine":   {"mean": float(cos.mean()), "std": float(cos.std()), "min": float(cos.min()), "max": float(cos.max()), "p10": float(np.percentile(cos, 10)), "p90": float(np.percentile(cos, 90))},
        "jaccard_word":   {"mean": float(jac.mean()), "std": float(jac.std()), "min": float(jac.min()), "max": float(jac.max())},
        "char_len_ratio": {"mean": float(rat.mean()), "std": float(rat.std()), "min": float(rat.min()), "max": float(rat.max())},
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT_JSONL}, {OUT_SUMMARY}")
    print(f"\nTF-IDF cosine: mean={cos.mean():.3f} (range {cos.min():.3f}-{cos.max():.3f})")
    print(f"Jaccard word:  mean={jac.mean():.3f} (range {jac.min():.3f}-{jac.max():.3f})")
    print(f"len ratio:     mean={rat.mean():.3f} (range {rat.min():.3f}-{rat.max():.3f})")


if __name__ == "__main__":
    main()
