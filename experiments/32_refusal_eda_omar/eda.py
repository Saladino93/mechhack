"""Exploratory data analysis on the Gemma refusal dataset.

Goals: surface which features correlate with refusal vs compliance.

Reads attacks_full.jsonl (878 attack prompts; ~half refused). Produces:

  histograms/
    - prompt_chars_by_refusal.png
    - prompt_tokens_by_refusal.png
    - response_chars_by_refusal.png
    - thinking_chars_by_refusal.png
    - judge_score_distribution.png

  ngrams/
    - response_first_word.csv      (top words refused/complied responses START with)
    - response_first_2gram.csv     (first bigram)
    - response_first_3gram.csv     (first trigram)
    - prompt_first_word.csv         (first word of attack prompt)
    - prompt_top_unigram_by_class.csv (most-discriminative prompt unigrams)
    - prompt_top_bigram_by_class.csv  (most-discriminative prompt bigrams)

  correlations/
    - feature_correlations.csv     (per-feature point-biserial correlation w/ is_refusal)
    - source_distribution.csv      (per-source refusal rate)

  eda_summary.md                    headline takeaways
"""
from __future__ import annotations
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
HERE.mkdir(parents=True, exist_ok=True)
(HERE / "histograms").mkdir(exist_ok=True)
(HERE / "ngrams").mkdir(exist_ok=True)
(HERE / "correlations").mkdir(exist_ok=True)

ATTRS = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"


def first_n_words(text, n):
    if not text: return ""
    words = text.strip().split()
    return " ".join(words[:n]).lower()


def main():
    rows = [json.loads(l) for l in ATTRS.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("is_refusal") is not None]
    print(f"loaded {len(rows)} prompts (with is_refusal label)", flush=True)

    df = pd.DataFrame([{
        "sample_id": r["sample_id"],
        "is_refusal": int(bool(r["is_refusal"])),
        "split": r.get("split"),
        "source": r.get("source", "unknown"),
        "prompt_chars": len(r.get("attack_prompt") or ""),
        "prompt_words": len((r.get("attack_prompt") or "").split()),
        "response_chars": len(r.get("response") or ""),
        "response_words": len((r.get("response") or "").split()),
        "thinking_chars": len(r.get("thinking") or ""),
        "n_tokens": int(r.get("n_tokens") or 0),
        "judge_score": float(r.get("judge_score") or 0),
        "prompt": r.get("attack_prompt") or "",
        "response": r.get("response") or "",
        "thinking": r.get("thinking") or "",
    } for r in rows])

    print(f"  refused: {df['is_refusal'].sum()},  complied: {(1 - df['is_refusal']).sum()}", flush=True)

    # -- 1. Histograms by class -----
    def hist_by_class(col, log=False, bins=40, fname=None):
        fig, ax = plt.subplots(figsize=(7, 4))
        for c, label in [(1, "refused"), (0, "complied")]:
            v = df[df["is_refusal"] == c][col].astype(float).values
            v = v[v >= 0]
            if log:
                v = np.log10(v + 1)
            ax.hist(v, bins=bins, alpha=0.55, label=f"{label} (n={len(v)})", density=True)
        xlab = f"log10({col} + 1)" if log else col
        ax.set_xlabel(xlab); ax.set_ylabel("density")
        ax.set_title(f"{col} distribution by class")
        ax.legend()
        plt.tight_layout()
        out = HERE / "histograms" / (fname or f"{col}.png")
        fig.savefig(out, dpi=120); plt.close(fig)
        print(f"  wrote {out}", flush=True)

    hist_by_class("prompt_chars", log=True)
    hist_by_class("prompt_words", log=True)
    hist_by_class("response_chars", log=True)
    hist_by_class("response_words", log=True)
    hist_by_class("thinking_chars", log=True)
    hist_by_class("n_tokens", log=True)
    hist_by_class("judge_score", log=False, bins=20)

    # -- 2. First N-grams of response -----
    for n in [1, 2, 3]:
        ref_counts = Counter(first_n_words(r["response"], n) for _, r in df.iterrows() if r["is_refusal"] == 1)
        com_counts = Counter(first_n_words(r["response"], n) for _, r in df.iterrows() if r["is_refusal"] == 0)
        ngram_df = pd.DataFrame([{
            "ngram": ng, "refused_count": ref_counts.get(ng, 0),
            "complied_count": com_counts.get(ng, 0),
            "refused_ratio": ref_counts.get(ng, 0) / max(1, sum(ref_counts.values())),
            "complied_ratio": com_counts.get(ng, 0) / max(1, sum(com_counts.values())),
        } for ng in set(ref_counts) | set(com_counts)])
        ngram_df["log_odds"] = np.log(
            (ngram_df["refused_count"] + 1) / (ngram_df["complied_count"] + 1)
        )
        ngram_df = ngram_df.sort_values("log_odds", ascending=False)
        ngram_df.to_csv(HERE / "ngrams" / f"response_first_{n}gram.csv", index=False)
        print(f"\n  TOP-15 response-start {n}grams MORE in refusals:", flush=True)
        for _, row in ngram_df.head(15).iterrows():
            print(f"    {row['ngram']!r:>30}  refused={row['refused_count']:3d} complied={row['complied_count']:3d}  log_odds={row['log_odds']:+.2f}",
                  flush=True)

    # -- 3. First word of prompt -----
    pf_ref = Counter(first_n_words(r["prompt"], 1) for _, r in df.iterrows() if r["is_refusal"] == 1)
    pf_com = Counter(first_n_words(r["prompt"], 1) for _, r in df.iterrows() if r["is_refusal"] == 0)
    pf_df = pd.DataFrame([{
        "word": w,
        "refused_count": pf_ref.get(w, 0),
        "complied_count": pf_com.get(w, 0),
        "log_odds": np.log((pf_ref.get(w, 0) + 1) / (pf_com.get(w, 0) + 1)),
    } for w in set(pf_ref) | set(pf_com)])
    pf_df = pf_df[pf_df["refused_count"] + pf_df["complied_count"] >= 5].sort_values("log_odds", ascending=False)
    pf_df.to_csv(HERE / "ngrams" / "prompt_first_word.csv", index=False)
    print(f"\n  TOP-12 prompt-start words MORE in refusals (≥5 occurrences):", flush=True)
    for _, row in pf_df.head(12).iterrows():
        print(f"    {row['word']!r:>20}  ref={row['refused_count']:3d} com={row['complied_count']:3d}  log_odds={row['log_odds']:+.2f}", flush=True)
    print(f"\n  TOP-12 prompt-start words MORE in compliance:", flush=True)
    for _, row in pf_df.tail(12).iloc[::-1].iterrows():
        print(f"    {row['word']!r:>20}  ref={row['refused_count']:3d} com={row['complied_count']:3d}  log_odds={row['log_odds']:+.2f}", flush=True)

    # -- 4. Discriminative prompt unigrams + bigrams (TF-IDF style) -----
    from sklearn.feature_extraction.text import CountVectorizer
    for ngram_range, fname in [((1, 1), "prompt_top_unigram_by_class.csv"),
                                ((2, 2), "prompt_top_bigram_by_class.csv")]:
        cv = CountVectorizer(analyzer="word", ngram_range=ngram_range, min_df=10,
                              max_features=5000, lowercase=True, token_pattern=r"\b[a-zA-Z]{2,}\b")
        X = cv.fit_transform(df["prompt"])
        terms = cv.get_feature_names_out()
        # Per-term mean over class
        ref_mask = df["is_refusal"].values == 1
        com_mask = df["is_refusal"].values == 0
        ref_count = np.asarray(X[ref_mask].sum(axis=0)).flatten()
        com_count = np.asarray(X[com_mask].sum(axis=0)).flatten()
        log_odds = np.log((ref_count + 1) / (com_count + 1))
        ng_df = pd.DataFrame({"term": terms, "refused_count": ref_count,
                                "complied_count": com_count, "log_odds": log_odds})
        ng_df = ng_df.sort_values("log_odds", ascending=False)
        ng_df.to_csv(HERE / "ngrams" / fname, index=False)

    # -- 5. Length-stratified refusal rate -----
    for col in ["prompt_chars", "prompt_words", "response_chars", "n_tokens"]:
        bins = np.quantile(df[col], np.linspace(0, 1, 6))
        bins = np.unique(bins)
        if len(bins) < 4: continue
        df[f"{col}_bin"] = pd.cut(df[col], bins=bins, include_lowest=True, duplicates="drop")
        rates = df.groupby(f"{col}_bin", observed=True)["is_refusal"].agg(["mean", "count"]).reset_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(rates)), rates["mean"], color="steelblue")
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(rates[f"{col}_bin"].astype(str), rates["count"])],
                            rotation=30, ha="right", fontsize=8)
        ax.axhline(df["is_refusal"].mean(), ls="--", c="grey", label="overall mean")
        ax.set_ylabel("Pr(refusal)")
        ax.set_title(f"Refusal rate by {col} (quintile bins)")
        ax.legend()
        ax.set_ylim(0, 1)
        plt.tight_layout()
        out = HERE / "histograms" / f"refusal_rate_by_{col}.png"
        fig.savefig(out, dpi=120); plt.close(fig)
        print(f"  wrote {out}", flush=True)

    # -- 6. Per-feature correlation with is_refusal -----
    feats = ["prompt_chars", "prompt_words", "response_chars", "response_words",
             "thinking_chars", "n_tokens", "judge_score"]
    corrs = []
    for f in feats:
        c = df[[f, "is_refusal"]].corr().iloc[0, 1]
        corrs.append({"feature": f, "pearson_r": c})
    corr_df = pd.DataFrame(corrs).sort_values("pearson_r", ascending=False)
    corr_df.to_csv(HERE / "correlations" / "feature_correlations.csv", index=False)
    print(f"\n  Per-feature correlation with is_refusal:", flush=True)
    for _, r in corr_df.iterrows():
        print(f"    {r['feature']:>16}  r={r['pearson_r']:+.4f}", flush=True)

    # -- 7. Source distribution -----
    src = df.groupby("source")["is_refusal"].agg(["mean", "count"]).reset_index()
    src = src.sort_values("count", ascending=False)
    src.to_csv(HERE / "correlations" / "source_distribution.csv", index=False)
    print(f"\n  Refusal rate by source (top 8):", flush=True)
    for _, r in src.head(8).iterrows():
        print(f"    {r['source']:>30}  n={r['count']:4d}  refusal_rate={r['mean']:.3f}", flush=True)

    # -- 8. Summary md -----
    overall = df["is_refusal"].mean()
    summary = [
        "# Refusal-Gemma EDA — headline findings",
        "",
        f"- **n = {len(df)}** prompts (refused: {int(df['is_refusal'].sum())}, complied: {int((1-df['is_refusal']).sum()).__index__()}, overall refusal rate {overall:.3f})",
        f"- **Strongest length correlate**: `{corr_df.iloc[0]['feature']}` r={corr_df.iloc[0]['pearson_r']:+.4f}",
        f"- **Length effect direction**: longer-prompt → {'higher' if corr_df[corr_df['feature']=='prompt_chars'].iloc[0]['pearson_r'] > 0 else 'lower'} refusal rate",
        "",
        "## Top response-start tokens that distinguish refusal",
        "",
        "(see `ngrams/response_first_*gram.csv` for full tables)",
        "",
    ]
    (HERE / "eda_summary.md").write_text("\n".join(summary) + "\n")
    print(f"\nwrote {HERE/'eda_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
