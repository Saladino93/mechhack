#!/usr/bin/env bash
# Sweep all (task, d_pca, layer) combos for the Pleshkov polynomial-quadratic probe.
#
# Layer choices come from the existing best-layer selections:
#   cyber_1 → L40 (mean-pool baseline 0.9825)
#   cyber_2 → L40 (mean-pool baseline 0.9462)
#   cyber_3 → L35 (mean-pool baseline 0.9549)
#   refusal → L32 (only layer extracted; mean-pool baseline 0.9265)
#
# d_pca rules (from notes.md):
#   - d=16  always safe (152 quadratic features)
#   - d=32  borderline for refusal (560 features vs ~832 samples) — RUN ANYWAY,
#           the brief asks; alpha sweep and ridge will catch overfit.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs results

run() {
    local task=$1 layer=$2 d=$3
    local tag="${task}_d${d}_L${layer}"
    echo "[run] $tag"
    python3 train.py --task "$task" --layer "$layer" --d_pca "$d" \
        2>&1 | tee "logs/${tag}.log"
}

# --- cyber tasks: both d=16 and d=32 -------------------------------------
run cyber_1 40 16
run cyber_1 40 32
run cyber_2 40 16
run cyber_2 40 32
run cyber_3 35 16
run cyber_3 35 32

# --- refusal: try both d=16 and d=32 (832 samples; d=32 = 560 features) --
run refusal_gemma 32 16
run refusal_gemma 32 32

echo
echo "=== summary ==="
python3 - <<'PY'
import json, glob, os
rows = []
for f in sorted(glob.glob("results/*.json")):
    r = json.load(open(f))
    rows.append((r["task"], r["layer"], r["d_pca"], r["n_samples"],
                 r["linear"]["auc_mean"], r["quadratic"]["auc_mean"],
                 r["delta_auc"], r["verdict"]))
print(f"{'task':<16}{'L':>4}{'d':>4}{'n':>6}{'linAUC':>10}{'quadAUC':>10}{'delta':>9}  verdict")
for t, l, d, n, la, qa, de, v in rows:
    print(f"{t:<16}{l:>4}{d:>4}{n:>6}{la:>10.4f}{qa:>10.4f}{de:>+9.4f}  {v}")
PY

echo
echo "=== regenerating notes.md ==="
python3 regen_notes.py
