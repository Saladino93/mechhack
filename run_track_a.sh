#!/bin/bash
# Master Track-A launcher.
#
# Phase 1 (parallel, all CPU):
#   - Pleshkov 13-layer sweep d=16 for refusal + cyber_1/2/3 (builds caches)
#   - TF-IDF + naive baselines (text-only, no extracts)
#
# Phase 2 (parallel, after refusal cache exists):
#   - pair_triple_concat (refusal)
#   - aggregate_layers (refusal)
#   - Pleshkov refusal d=32 sweep
#
# Phase 3 (after cyber caches exist):
#   - 4×4 cross-task (cyber_1/2/3 + refusal)
#   - combined-data probe (cyber_3 ∪ refusal)
#
# Each step commits + pushes when done.

set -u
cd /home/ubuntu/georgia/mechhack

mkdir -p logs

ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

commit_push() {
    local msg="$1"
    git add -A 2>&1 | tail -1
    git diff --cached --quiet && { echo "$(ts) nothing to commit for: $msg"; return; }
    git commit -m "$(printf 'Track A: %s\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>' "$msg")" 2>&1 | tail -3
    git push 2>&1 | tail -3 || {
        echo "$(ts) push failed, attempting rebase + retry"
        git pull --rebase 2>&1 | tail -3
        git push 2>&1 | tail -3
    }
}

run_one() {
    local name="$1"; shift
    echo "$(ts) START $name"
    "$@" > "logs/${name}.log" 2>&1
    echo "$(ts) DONE  $name (rc=$?)"
}

echo "$(ts) ===== TRACK A start ====="

# --- Phase 1: build caches in parallel (5 jobs) ---
echo "$(ts) Phase 1: 5 parallel CPU jobs"

run_one pleshkov_refusal_d16 python3 experiments/17_quadratic_probe_omar/sweep_layers.py --task refusal_gemma --d_pca 16 &
PID_RD16=$!

run_one pleshkov_cyber_1_d16 python3 experiments/17_quadratic_probe_omar/sweep_layers.py --task cyber_1 --d_pca 16 &
PID_C1=$!

run_one pleshkov_cyber_2_d16 python3 experiments/17_quadratic_probe_omar/sweep_layers.py --task cyber_2 --d_pca 16 &
PID_C2=$!

run_one pleshkov_cyber_3_d16 python3 experiments/17_quadratic_probe_omar/sweep_layers.py --task cyber_3 --d_pca 16 &
PID_C3=$!

run_one tfidf_baselines python3 experiments/21_probe_zoo_omar/tfidf_naive_baselines.py &
PID_TF=$!

echo "$(ts) Phase 1 PIDs: refusal=$PID_RD16 c1=$PID_C1 c2=$PID_C2 c3=$PID_C3 tfidf=$PID_TF"

# --- wait for refusal cache to exist before launching dependents ---
while [ ! -f experiments/17_quadratic_probe_omar/cache/refusal_13layer_mean.npz ]; do
    sleep 30
    echo "$(ts) waiting for refusal cache..."
done
echo "$(ts) refusal cache ready"

# --- Phase 2: refusal-cache dependents ---
echo "$(ts) Phase 2: refusal-cache dependents"

run_one pair_triple_concat python3 experiments/21_probe_zoo_omar/pair_triple_concat.py &
PID_PT=$!

run_one aggregate_layers python3 experiments/21_probe_zoo_omar/aggregate_layers.py &
PID_AG=$!

# Wait for the d=16 refusal sweep to finish (cache is shared with d=32)
wait $PID_RD16
echo "$(ts) pleshkov refusal d=16 done"
commit_push "pleshkov refusal d=16 13-layer sweep + tfidf baselines kickoff"

# Run d=32 sweep using same cache (no extra disk IO)
run_one pleshkov_refusal_d32 python3 experiments/17_quadratic_probe_omar/sweep_layers.py --task refusal_gemma --d_pca 32 &
PID_RD32=$!

# Wait for TF-IDF
wait $PID_TF
echo "$(ts) tfidf baselines done"

# Wait for pair-triple and aggregate
wait $PID_PT
wait $PID_AG
commit_push "refusal probe zoo: pair/triple concat + aggregate-layers + tfidf baselines"

# --- Phase 3: after cyber caches exist, run cross-task + combined ---
wait $PID_C1
echo "$(ts) cyber_1 d=16 done"
wait $PID_C2
echo "$(ts) cyber_2 d=16 done"
wait $PID_C3
echo "$(ts) cyber_3 d=16 done"
wait $PID_RD32
echo "$(ts) pleshkov refusal d=32 done"
commit_push "Pleshkov 13-layer sweep complete: cyber_1/2/3 d=16 + refusal d=16/32"

# 4x4 cross-task
run_one cross_task_4x4 python3 experiments/22_cross_task_4x4_omar/run_4x4.py
commit_push "4×4 cross-task transfer matrix incl. refusal"

# Combined probe
run_one combined_probe python3 experiments/23_combined_probe_omar/run_combined.py
commit_push "Combined-data probe (cyber_3 ∪ refusal) at L35"

# Generate plots for all sweeps
echo "$(ts) generating Pleshkov sweep plots"
for task in refusal_gemma cyber_1 cyber_2 cyber_3; do
    for d in 16 32; do
        if [ -f "experiments/17_quadratic_probe_omar/results/${task}_d${d}_sweep.json" ]; then
            python3 experiments/17_quadratic_probe_omar/plot_sweep.py --task $task --d_pca $d > "logs/plot_${task}_d${d}.log" 2>&1 || true
        fi
    done
done
commit_push "Pleshkov sweep plots"

echo "$(ts) ===== TRACK A complete ====="
