"""Regenerate the results table in notes.md from results/*.json.

Reads every per-task JSON, builds the markdown table, and rewrites the
"Results" + "Per-fold ablation" + "Takeaway" sections of notes.md in place.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
NOTES = HERE / "notes.md"


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:.4f}, {hi:.4f}]"


def main():
    files = sorted(RESULTS.glob("*.json"))
    if not files:
        print("no result files yet")
        return

    by_task: dict[str, dict] = {}
    for f in files:
        r = json.loads(f.read_text())
        by_task.setdefault(r["task"], {})[r["d_pca"]] = r

    # Build the main results table.
    lines = []
    lines.append("| Task | Layer | n | linear (raw) | quad d=16 | quad d=32 | Δ d=16 | Δ d=32 |")
    lines.append("|------|------:|--:|:-------------|:----------|:----------|-------:|-------:|")
    rows = []
    for task in ["cyber_1", "cyber_2", "cyber_3", "refusal_gemma"]:
        if task not in by_task:
            continue
        r16 = by_task[task].get(16)
        r32 = by_task[task].get(32)
        ref = r16 or r32
        lin = ref["linear"]
        layer = ref["layer"]
        n = ref["n_samples"]
        lin_str = (f"{lin['auc_mean']:.4f} ± {lin['auc_std']:.4f} "
                   f"{fmt_ci(lin['auc_wilson95_lo'], lin['auc_wilson95_hi'])}")
        d16_str = "—"
        delta16 = float("nan")
        if r16 is not None:
            q = r16["quadratic"]
            d16_str = (f"{q['auc_mean']:.4f} ± {q['auc_std']:.4f} "
                       f"{fmt_ci(q['auc_wilson95_lo'], q['auc_wilson95_hi'])}")
            delta16 = r16["delta_auc"]
        d32_str = "—"
        delta32 = float("nan")
        if r32 is not None:
            q = r32["quadratic"]
            d32_str = (f"{q['auc_mean']:.4f} ± {q['auc_std']:.4f} "
                       f"{fmt_ci(q['auc_wilson95_lo'], q['auc_wilson95_hi'])}")
            delta32 = r32["delta_auc"]
        d16_delta = "—" if r16 is None else f"{delta16:+.4f}"
        d32_delta = "—" if r32 is None else f"{delta32:+.4f}"
        lines.append(f"| {task} | {layer} | {n} | {lin_str} | {d16_str} | {d32_str} | {d16_delta} | {d32_delta} |")
        rows.append((task, layer, n, lin, r16, r32, delta16, delta32))

    table_md = "\n".join(lines)

    # Apples-to-apples table: linear-on-raw vs linear-on-d-PCs vs quadratic-on-d-PCs
    apples_lines = []
    apples_lines.append("| Task | d | linear (raw) | linear-on-d-PCs | quadratic | Δ vs raw | Δ vs d-PCs |")
    apples_lines.append("|------|--:|:-------------|:----------------|:----------|---------:|-----------:|")
    for task, layer, n, lin, r16, r32, d16, d32 in rows:
        for d, r in [(16, r16), (32, r32)]:
            if r is None:
                continue
            lin_pcs = r.get("linear_on_pcs")
            q = r["quadratic"]
            lin_str = f"{lin['auc_mean']:.4f}"
            if lin_pcs is not None:
                lpcs_str = f"{lin_pcs['auc_mean']:.4f}"
                d_pcs = q["auc_mean"] - lin_pcs["auc_mean"]
                d_pcs_str = f"{d_pcs:+.4f}"
            else:
                lpcs_str = "—"
                d_pcs_str = "—"
            q_str = f"{q['auc_mean']:.4f}"
            d_raw = q["auc_mean"] - lin["auc_mean"]
            apples_lines.append(f"| {task} | {d} | {lin_str} | {lpcs_str} | {q_str} | "
                                f"{d_raw:+.4f} | {d_pcs_str} |")
    apples_md = "\n".join(apples_lines)

    # Per-fold breakdown
    fold_lines = []
    for task, layer, n, lin, r16, r32, d16, d32 in rows:
        fold_lines.append(f"\n### {task} (L{layer}, n={n})")
        fold_lines.append("\n| fold | linear AUC | quad d=16 AUC (α) | quad d=32 AUC (α) |")
        fold_lines.append("|-----:|-----------:|------------------:|------------------:|")
        lin_folds = lin["fold_metrics"]
        q16_folds = r16["quadratic"]["fold_metrics"] if r16 else None
        q32_folds = r32["quadratic"]["fold_metrics"] if r32 else None
        for fi in range(5):
            ln = lin_folds[fi]["auc"]
            ln_s = f"{ln:.4f}"
            if q16_folds:
                q = q16_folds[fi]
                q16_s = f"{q['auc']:.4f} (α={q['alpha_chosen']})"
            else:
                q16_s = "—"
            if q32_folds:
                q = q32_folds[fi]
                q32_s = f"{q['auc']:.4f} (α={q['alpha_chosen']})"
            else:
                q32_s = "—"
            fold_lines.append(f"| {fi} | {ln_s} | {q16_s} | {q32_s} |")

    # Takeaway
    take = []
    take.append("**Decision rule** (from the brief): Δ > +0.005 → quadratic helps; "
                "|Δ| ≤ 0.005 → linear sufficient; Δ < -0.005 → quadratic overfits or PCA bottleneck dominates.\n")
    for task, layer, n, lin, r16, r32, d16, d32 in rows:
        helps = []
        for d, delta in [(16, d16), (32, d32)]:
            if delta != delta:  # NaN
                continue
            if delta > 0.005:
                helps.append(f"d={d}: **quadratic helps** ({delta:+.4f})")
            elif delta < -0.005:
                helps.append(f"d={d}: quadratic worse ({delta:+.4f})")
            else:
                helps.append(f"d={d}: ≈ linear ({delta:+.4f})")
        take.append(f"- **{task}** (L{layer}, n={n}, linear={lin['auc_mean']:.4f}): "
                    + "; ".join(helps))
    takeaway_md = "\n".join(take)

    # Bottom-line
    notable = []
    for task, layer, n, lin, r16, r32, d16, d32 in rows:
        for d, delta in [(16, d16), (32, d32)]:
            if delta != delta:
                continue
            if delta > 0.005:
                notable.append(f"  - {task} d={d}: linear {lin['auc_mean']:.4f} → "
                               f"quadratic {(r16 if d==16 else r32)['quadratic']['auc_mean']:.4f} "
                               f"({delta:+.4f})")
    bottom = "\n**Notable wins for quadratic:**\n" + (
        "\n".join(notable) if notable else "  - (none — quadratic does not exceed linear by >0.005 on any task)"
    )

    # Splice these into notes.md
    src = NOTES.read_text()
    head, _, _ = src.partition("## Results")
    new = (head
           + "## Results\n\n"
           + table_md + "\n\n"
           + "### Apples-to-apples: linear-on-raw vs linear-on-d-PCs vs quadratic\n\n"
           + "If `quadratic > linear-on-d-PCs` then the quadratic terms are doing genuine work "
           + "(beyond what a linear probe could squeeze out of the same d PCs).\n"
           + "If `quadratic ≈ linear-on-d-PCs < linear-on-raw` then the gap is the PCA bottleneck, "
           + "not nonlinearity vs linearity.\n\n"
           + apples_md + "\n\n"
           + "## Per-fold ablation\n"
           + "\n".join(fold_lines) + "\n\n"
           + "## Takeaway\n\n"
           + takeaway_md + "\n"
           + bottom + "\n\n"
           + "## Files\n\n"
           + "- `probes.py` — `QuadraticProbe` (sklearn-style)\n"
           + "- `train.py` — CLI `--task --layer --d_pca`, 5-fold CV, writes JSON\n"
           + "- `run_all.sh` — sweeps {cyber_1, cyber_2, cyber_3, refusal_gemma} × {16, 32}\n"
           + "- `regen_notes.py` — rebuilds the tables here from `results/*.json`\n"
           + "- `results/<task>_d<d>_L<layer>.json` — full per-fold metrics + inner alpha log\n"
           + "- `logs/<task>_d<d>_L<layer>.log` — stdout from each `train.py` invocation\n"
    )
    NOTES.write_text(new)
    print(f"updated {NOTES} ({len(rows)} task rows)")


if __name__ == "__main__":
    main()
