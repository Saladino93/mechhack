#!/bin/bash
# Loop-runs judge_rollouts.py until rollouts.py has finished AND every rollout
# row has a corresponding judgement. Each iteration the judge resumes from
# judgements.jsonl, so it only judges new (sample_id, variant) pairs.
set -u

ROLLOUTS_PID=${ROLLOUTS_PID:-117762}
EXP_DIR="/home/ubuntu/georgia/mechhack/experiments/13_pre_rewrites_omar"

source /home/ubuntu/.openrouter_env

iter=0
while true; do
  iter=$((iter+1))
  echo "=== judge_loop iter $iter at $(date +%H:%M:%S) ==="
  python3 "$EXP_DIR/judge_rollouts.py" 2>&1 | tail -50
  rc=$?
  echo "  judge exit=$rc"

  # are there pending rows?
  pending=$(python3 -c "
import json
roll = set()
for line in open('$EXP_DIR/rollouts.jsonl'):
    line = line.strip()
    if not line: continue
    r = json.loads(line)
    roll.add((r['sample_id'], r['variant']))
done = set()
import os
if os.path.exists('$EXP_DIR/judgements.jsonl'):
    for line in open('$EXP_DIR/judgements.jsonl'):
        line = line.strip()
        if not line: continue
        try:
            r = json.loads(line)
            done.add((r['sample_id'], r['variant']))
        except: pass
print(len(roll - done))
")

  rollouts_alive="no"
  if kill -0 "$ROLLOUTS_PID" 2>/dev/null; then rollouts_alive="yes"; fi
  echo "  pending=$pending  rollouts_alive=$rollouts_alive"

  if [ "$pending" = "0" ] && [ "$rollouts_alive" = "no" ]; then
    echo "=== all caught up and rollouts.py finished. exiting. ==="
    break
  fi
  sleep 30
done
