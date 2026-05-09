#!/bin/bash
# Track B: GPU jobs (Kramar architectures A/C/D + CC++ relaunch).
#
# Runs sequentially because they share the H100. Each takes ~30-90 min.
#
# Pre-condition: Phase 3a Gemma rollouts must be done (PID 245830 free).

set -u
cd /home/ubuntu/georgia/mechhack

mkdir -p logs

ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

commit_push() {
    local msg="$1"
    git add -A 2>&1 | tail -1
    git diff --cached --quiet && { echo "$(ts) nothing to commit for: $msg"; return; }
    git commit -m "$(printf 'Track B: %s\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>' "$msg")" 2>&1 | tail -3
    git push 2>&1 | tail -3 || {
        echo "$(ts) push failed, rebase + retry"
        git pull --rebase 2>&1 | tail -3
        git push 2>&1 | tail -3
    }
}

# Wait for: (1) master pipeline to finish or be killed, AND
#           (2) cyber test extraction (Track G) to finish if running.
# Otherwise we'd race for the GPU.
echo "$(ts) waiting for master pipeline + cyber extract to clear..."
while pgrep -f "run_pipeline_while_away.sh\|run_track_g.sh\|experiments/25_cyber_test_extract_omar/extract.py" > /dev/null 2>&1; do
    sleep 60
    n_rollouts=$(wc -l < experiments/13_pre_rewrites_omar/rollouts_phase3.jsonl 2>/dev/null || echo 0)
    n_cyber_test=$(ls /home/ubuntu/extracts/cyber_all_omar/ 2>/dev/null | wc -l)
    echo "$(ts) waiting... rollouts=$n_rollouts; cyber_extracts=$n_cyber_test"
done
echo "$(ts) GPU is free; starting Track B"

run_one() {
    local name="$1"; shift
    echo "$(ts) START $name"
    "$@" > "logs/${name}.log" 2>&1
    local rc=$?
    echo "$(ts) DONE  $name (rc=$rc)"
    return $rc
}

# --- B1/B2/B3: Kramar arch A, C, D on refusal at L30 (only layer Rolling has) ---
echo "$(ts) ===== Kramar architectures A/C/D on refusal (L30) ====="
for VARIANT in attention_kramar multimax rolling_multimax; do
    NAME="kramar_${VARIANT}_refusal_L30"
    run_one "$NAME" python3 experiments/16_multimax_probe_omar/train.py \
        --task refusal_gemma4_31b --layer 30 --variant "$VARIANT"
    commit_push "Kramar arch ${VARIANT} on refusal L30"
done

# --- B4: CC++ relaunch on GPU for cyber_1 ---
echo "$(ts) ===== CC++ on cyber_1 (GPU) ====="
run_one cc_plus_plus_cyber_1 python3 experiments/05_cc_plus_plus_omar/train_heads.py
commit_push "CC++ cyber_1 GPU rerun"

# --- B5: CC++ on refusal L40 mean and L45 last-tok would need a new script ---
# For now: skip — exp 05 is cyber_1-coded. Document gap.

echo "$(ts) ===== TRACK B complete ====="
