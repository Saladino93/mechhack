"""Compute Pleshkov-as-f causality metrics.

Joins:
  - edits_pleshkov_scored.jsonl  (per-edit Pleshkov scores)
  - judgements_phase3.jsonl       (judge labels)
  - attacks_full.jsonl            (orig=refusal labels)

For Pleshkov's score (not a probability — Ridge regression on {0,1}):
  threshold τ chosen so that on the TRAINING set with original prompts,
  τ matches the threshold producing acc≈0.83 (the cv mean acc from exp 17
  refusal_gemma_d16_L32). We compute it directly from per-fold mean.

Output:
  pleshkov_metrics.json — Pr(f|edit), Pr(m|edit), Pr(m|f flipped, orig=refusal)
                          with Wilson 95% CIs.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent

EDITS = HERE / "edits_pleshkov_scored.jsonl"
JUDGE = REPO_ROOT / "experiments" / "13_pre_rewrites_omar" / "judgements_phase3.jsonl"
ATTRS_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"
OUT = HERE / "pleshkov_metrics.json"


def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n))/denom
    half = (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)))/denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def main():
    if not EDITS.exists():
        print(f"missing {EDITS}; run score_pleshkov.py first")
        return

    rows = [json.loads(l) for l in EDITS.read_text().splitlines() if l.strip()]
    print(f"loaded {len(rows)} pleshkov-scored edits")

    # orig labels
    orig_label = {}
    with ATTRS_FULL.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("is_refusal") is not None:
                orig_label[r["sample_id"]] = int(bool(r["is_refusal"]))

    # judge
    judge_flip = {}
    with JUDGE.open() as f:
        for line in f:
            r = json.loads(line)
            v = r.get("is_refusal_judge")
            if v is None: continue
            judge_flip[(r.get("sample_id"), r.get("variant", "?"))] = (v is False)

    # By (sid, kind) -> score
    score_map = {(r["sample_id"], r["edit_kind"]): r["pleshkov_score"] for r in rows}

    # Pleshkov is a Ridge on {0,1}. The natural threshold for "predicts class 1
    # (=refusal)" is 0.5. But scores can range outside [0,1]. Pick τ=0.5.
    THR = 0.5

    edit_kinds = sorted({k[1] for k in score_map.keys() if k[1] != "original"})
    summary = {"threshold": THR, "n_edits": len(rows), "by_edit_kind": {},
               "overall": None}

    overall_n_total = 0
    overall_n_f = 0
    overall_n_m = 0
    overall_n_concord_denom = 0
    overall_n_concord_num = 0

    for kind in edit_kinds:
        n_total, n_probe_flip, n_behavior_flip, n_concord = 0, 0, 0, 0
        n_probe_concord_denom = 0
        for sid in {k[0] for k in score_map.keys()}:
            if orig_label.get(sid) != 1: continue
            orig = score_map.get((sid, "original"))
            edit = score_map.get((sid, kind))
            if orig is None or edit is None: continue
            n_total += 1
            f_orig = orig >= THR
            f_now = edit < THR
            jud = judge_flip.get((sid, kind))
            if jud is True:
                n_behavior_flip += 1
            if f_orig and f_now:
                n_probe_flip += 1
                n_probe_concord_denom += 1
                if jud is True:
                    n_concord += 1

        pe = wilson(n_probe_flip, n_total)
        pme = wilson(n_behavior_flip, n_total)
        pc = wilson(n_concord, n_probe_concord_denom)
        summary["by_edit_kind"][kind] = {
            "n_orig_refusal": n_total,
            "Pr_f_given_edit":     {"k": n_probe_flip, "n": n_total, "point": pe[0], "ci": [pe[1], pe[2]]},
            "Pr_model_given_edit": {"k": n_behavior_flip, "n": n_total, "point": pme[0], "ci": [pme[1], pme[2]]},
            "Pr_model_given_f":    {"k": n_concord, "n": n_probe_concord_denom, "point": pc[0], "ci": [pc[1], pc[2]]},
        }
        overall_n_total += n_total
        overall_n_f += n_probe_flip
        overall_n_m += n_behavior_flip
        overall_n_concord_denom += n_probe_concord_denom
        overall_n_concord_num += n_concord
        print(f"  {kind:>30}  n={n_total:3d}  Pr(f|e)={pe[0]:.3f}  "
              f"Pr(m|e)={pme[0]:.3f}  Pr(m|f)={pc[0]:.3f}  ({n_concord}/{n_probe_concord_denom})")

    pe = wilson(overall_n_f, overall_n_total)
    pme = wilson(overall_n_m, overall_n_total)
    pc = wilson(overall_n_concord_num, overall_n_concord_denom)
    summary["overall"] = {
        "n_orig_refusal": overall_n_total,
        "Pr_f_given_edit":     {"k": overall_n_f, "n": overall_n_total, "point": pe[0], "ci": [pe[1], pe[2]]},
        "Pr_model_given_edit": {"k": overall_n_m, "n": overall_n_total, "point": pme[0], "ci": [pme[1], pme[2]]},
        "Pr_model_given_f":    {"k": overall_n_concord_num, "n": overall_n_concord_denom, "point": pc[0], "ci": [pc[1], pc[2]]},
    }
    print(f"\n  OVERALL Pleshkov (4 edit kinds combined): n={overall_n_total}")
    print(f"    Pr(f|edit) = {pe[0]:.3f}  CI=[{pe[1]:.3f},{pe[2]:.3f}]")
    print(f"    Pr(model|edit) = {pme[0]:.3f}  CI=[{pme[1]:.3f},{pme[2]:.3f}]")
    print(f"    Pr(model|f flipped, orig=refusal) = {pc[0]:.3f}  "
          f"CI=[{pc[1]:.3f},{pc[2]:.3f}]  ({overall_n_concord_num}/{overall_n_concord_denom})")

    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
