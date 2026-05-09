#!/usr/bin/env bash
# After cache_data.py has prefilled cache/, this is the fast-path: each
# train.py invocation loads the .npz cache (<1s) and runs CV (<30s for refusal,
# <2min for each cyber task). Run all 8 jobs in parallel since they don't share
# data anymore.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs results

run_bg() {
    local task=$1 layer=$2 d=$3
    local tag="${task}_d${d}_L${layer}"
    python3 train.py --task "$task" --layer "$layer" --d_pca "$d" \
        > "logs/${tag}.log" 2>&1 &
    echo "[bg] $tag (pid $!)"
}

run_bg cyber_1 40 16
run_bg cyber_1 40 32
run_bg cyber_2 40 16
run_bg cyber_2 40 32
run_bg cyber_3 35 16
run_bg cyber_3 35 32
run_bg refusal_gemma 32 16
run_bg refusal_gemma 32 32
wait

echo
echo "=== summary ==="
python3 - <<'PY'
import json, glob
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
