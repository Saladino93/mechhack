#!/bin/bash
# Track P: Phase-2 maximization (push toward all-three-high Pr metrics).
#
# P2 (CPU): probe-gradient edit picker — generate edits using fitted probe coef
#           Runs once fitted_probes.npz exists.
# P3 (CPU): multi-probe consensus filter — Pr(model | f1∧f2 / f1∧f2∧f3)
#           Runs once edits_scored_multi.jsonl exists (Track E E1b done).
# P4 (CPU): best-of-N edit picker — runs alongside P3.
# P1 (GPU): roll Gemma on gradient edits → judge → final causal-flip metrics.
#           Runs after Track B/E done (no GPU contention).

set -u
cd /home/ubuntu/georgia/mechhack

mkdir -p logs

ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

commit_push() {
    local msg="$1"
    git add -A 2>&1 | tail -1
    git diff --cached --quiet && return
    git commit -m "$(printf 'Track P: %s\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>' "$msg")" 2>&1 | tail -3
    git push 2>&1 | tail -3 || { git pull --rebase 2>&1 | tail -3; git push 2>&1 | tail -3; }
}

# --- P2: probe-gradient edit picker ---
echo "$(ts) waiting for fitted_probes.npz..."
while [ ! -f experiments/24_robustness_omar/fitted_probes.npz ]; do
    sleep 30
done
echo "$(ts) fitted_probes.npz ready; running P2 for 3 probe variants"
for PROBE in lr_mean_L40 lr_last_L45 lr_multi_concat; do
    python3 experiments/26_probe_gradient_edits_omar/picker.py --probe $PROBE \
        > logs/p2_picker_$PROBE.log 2>&1
done
commit_push "P2: probe-gradient edits generated for 3 probes"

# --- P3 + P4: consensus + best-of-N ---
echo "$(ts) waiting for edits_scored_multi.jsonl (Track E E1b)..."
while [ ! -f experiments/24_robustness_omar/edits_scored_multi.jsonl ]; do
    sleep 60
    n=$(wc -l < experiments/24_robustness_omar/edits_scored_multi.jsonl 2>/dev/null || echo 0)
    echo "$(ts) waiting... edits_scored_multi=$n"
done
sleep 30  # let E1b finalize
echo "$(ts) running P3+P4 consensus + best-of-N"
python3 experiments/27_consensus_omar/consensus_filter.py > logs/p3_p4_consensus.log 2>&1
commit_push "P3+P4: multi-probe consensus + best-of-N picker"

# --- P1: roll Gemma on gradient edits (GPU) ---
echo "$(ts) waiting for GPU to clear (Track B + Track E + Track G)..."
while pgrep -f "run_pipeline_while_away.sh\|run_track_b.sh\|run_track_e.sh\|run_track_g.sh\|score_edits_multi_probe\|extract.py\|train.py --task refusal_gemma" > /dev/null 2>&1; do
    sleep 120
    echo "$(ts) GPU busy; waiting"
done
echo "$(ts) GPU free; rolling Gemma on gradient edits"
python3 experiments/26_probe_gradient_edits_omar/rollout_grad_edits.py \
    > logs/p1_gradient_rollouts.log 2>&1
commit_push "P1: gradient-edit rollouts (Gemma)"

echo "$(ts) ===== TRACK P complete ====="
