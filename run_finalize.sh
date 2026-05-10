#!/bin/bash
# Final auto-coordinator. Run P1 (gradient rollouts) once attention_kramar
# finishes, then commit everything + push. Survives disconnect.

set -u
cd /home/ubuntu/georgia/mechhack
mkdir -p logs

ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

commit_push() {
    local msg="$1"
    git add -A 2>&1 | tail -1
    git diff --cached --quiet && return
    git commit -m "$(printf 'Finalize: %s\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>' "$msg")" 2>&1 | tail -3
    git push 2>&1 | tail -3 || { git pull --rebase 2>&1 | tail -3; git push 2>&1 | tail -3; }
}

# Wait for attention_kramar (GPU) to finish
echo "$(ts) waiting for attention_kramar to finish..."
while pgrep -f "experiments/16_multimax_probe_omar/train.py" > /dev/null 2>&1; do
    sleep 60
    echo "$(ts) attention_kramar still running"
done
echo "$(ts) attention_kramar done"
commit_push "attention_kramar (Kramar arch A) refusal L30 done"

# Wait for combined probe to finish
echo "$(ts) waiting for combined probe..."
while pgrep -f "experiments/23_combined_probe_omar/run_combined.py" > /dev/null 2>&1; do
    sleep 30
done
commit_push "combined-data probe (cyber_3 ∪ refusal) results"

# Now run P1 — Gemma rollouts on the 472 gradient edits.
echo "$(ts) launching P1 (gradient edit rollouts)"
python3 experiments/26_probe_gradient_edits_omar/rollout_grad_edits.py \
    > logs/p1_gradient_rollouts.log 2>&1
commit_push "P1: gradient-edit Gemma rollouts (472 edits)"

# Re-run consensus on enriched scored set if any new edits scored
python3 experiments/27_consensus_omar/consensus_filter.py \
    > logs/p3_p4_v2.log 2>&1 || true
python3 experiments/24_robustness_omar/compute_robustness.py \
    > logs/compute_robustness_v2.log 2>&1 || true
commit_push "Final: consensus + robustness re-computed"

echo "$(ts) ===== FINALIZE complete ====="
