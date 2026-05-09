#!/usr/bin/env python3
"""Analyze first-sentence ablation rollout/refusal results.

Expected input: a CSV or JSONL with at least:
  sample_id, variant, is_refusal
Optionally keeps metadata from make_first_sentence_ablation.py:
  original_quadrant, original_score_logit, original_is_refusal

The row with variant='original' is treated as the rerun baseline; flips are
computed relative to that rerun, not only relative to the cached dataset label.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".jsonl", ".json"}:
        return pd.read_json(p, lines=True)
    return pd.read_csv(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="CSV/JSONL with rollout verification results.")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    df = read_any(args.results)
    if "is_refusal" not in df.columns:
        raise SystemExit("Need an is_refusal column from the verifier output.")
    df["is_refusal"] = df["is_refusal"].astype(int)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    piv = df.pivot_table(index="sample_id", columns="variant", values="is_refusal", aggfunc="first")
    meta_cols = [c for c in ["original_quadrant", "original_score_logit", "original_is_refusal"] if c in df.columns]
    meta = df.drop_duplicates("sample_id").set_index("sample_id")[meta_cols] if meta_cols else pd.DataFrame(index=piv.index)
    joined = meta.join(piv)

    if "original" not in joined.columns:
        raise SystemExit("Need variant='original' rows to compute rerun-relative flips.")

    summary = {"n_samples": int(len(joined)), "variants": {}}
    for v in [c for c in joined.columns if c not in meta_cols and c != "original"]:
        valid = joined[["original", v]].dropna()
        before = valid["original"].astype(int)
        after = valid[v].astype(int)
        refusal_to_compliance = int(((before == 1) & (after == 0)).sum())
        compliance_to_refusal = int(((before == 0) & (after == 1)).sum())
        unchanged = int((before == after).sum())
        summary["variants"][v] = {
            "n": int(len(valid)),
            "original_refusal_rate": float(before.mean()) if len(valid) else None,
            "ablated_refusal_rate": float(after.mean()) if len(valid) else None,
            "delta_refusal_rate": float(after.mean() - before.mean()) if len(valid) else None,
            "refusal_to_compliance": refusal_to_compliance,
            "compliance_to_refusal": compliance_to_refusal,
            "unchanged": unchanged,
        }
        joined[f"flip_{v}"] = after.reindex(joined.index) - before.reindex(joined.index)

        if "original_quadrant" in joined.columns:
            rows = []
            for q, sub in joined.dropna(subset=["original", v]).groupby("original_quadrant"):
                b = sub["original"].astype(int)
                a = sub[v].astype(int)
                rows.append({
                    "quadrant": q,
                    "n": len(sub),
                    "original_refusal_rate": b.mean(),
                    "ablated_refusal_rate": a.mean(),
                    "delta_refusal_rate": a.mean() - b.mean(),
                    "refusal_to_compliance": int(((b == 1) & (a == 0)).sum()),
                    "compliance_to_refusal": int(((b == 0) & (a == 1)).sum()),
                    "unchanged": int((b == a).sum()),
                })
            pd.DataFrame(rows).to_csv(out / f"summary_by_quadrant_{v}.csv", index=False)

    joined.reset_index().to_csv(out / "paired_ablation_results.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
