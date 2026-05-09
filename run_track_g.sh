#!/bin/bash
# Track G: Cyber test extraction (kills Phase 4 of master pipeline first).
#
# Sequence:
#   1. Wait for Phase 3c to finish (signaled by phase3_summary.json existing).
#   2. Kill master pipeline (PID 240152) and any in-progress Phase 4 Rolling.
#   3. Run cyber test extraction (~2h GPU on 1257 missing prompts).
#   4. Commit + push.
# After this, Track B (waiting for run_pipeline_while_away.sh to die) starts.

set -u
cd /home/ubuntu/georgia/mechhack

mkdir -p logs

ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

commit_push() {
    local msg="$1"
    git add -A 2>&1 | tail -1
    git diff --cached --quiet && { echo "$(ts) nothing to commit for: $msg"; return; }
    git commit -m "$(printf 'Track G: %s\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>' "$msg")" 2>&1 | tail -3
    git push 2>&1 | tail -3 || { git pull --rebase 2>&1 | tail -3; git push 2>&1 | tail -3; }
}

# Wait for Phase 3c
echo "$(ts) waiting for phase3_summary.json (Phase 3c done)..."
while [ ! -f experiments/13_pre_rewrites_omar/phase3_summary.json ]; do
    sleep 60
    n_rollouts=$(wc -l < experiments/13_pre_rewrites_omar/rollouts_phase3.jsonl 2>/dev/null || echo 0)
    n_judge=$(wc -l < experiments/13_pre_rewrites_omar/judgements_phase3.jsonl 2>/dev/null || echo 0)
    echo "$(ts) waiting... rollouts=$n_rollouts; judgements=$n_judge"
done
echo "$(ts) Phase 3c done. Killing master pipeline + Phase 4 Rolling probe."

# Kill master pipeline (parent) so it doesn't proceed to Phase 4
pkill -f "run_pipeline_while_away.sh" 2>/dev/null
sleep 5
# Kill any in-progress Phase 4 Rolling probe (heredoc python or train.py)
pkill -f "experiments/16_multimax_probe_omar/train.py" 2>/dev/null
sleep 3

# Show what's still alive on GPU
echo "$(ts) GPU state after kill:"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>&1 | head -5
sleep 5

# Wait until GPU has < 1 GiB allocated (model unloaded)
while true; do
    used=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo 0)
    if [ -z "$used" ] || [ "$used" -lt 1000 ]; then break; fi
    echo "$(ts) GPU still has $used MiB allocated, waiting..."
    sleep 10
done

echo "$(ts) GPU is free. Starting cyber test extraction."
python3 experiments/25_cyber_test_extract_omar/extract.py > logs/cyber_test_extract.log 2>&1
commit_push "G1: cyber test split extracted to 100% (~1257 new .pt files)"

echo "$(ts) ===== TRACK G complete ====="
