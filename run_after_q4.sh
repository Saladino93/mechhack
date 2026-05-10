#!/bin/bash
# Wait for Q4 to finish, then run transfer eval
cd /home/ubuntu/georgia/mechhack
mkdir -p logs
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) waiting for Q4 honest eval to finish..."
while pgrep -f "experiments/31_honest_eval_omar/eval_test_split.py" > /dev/null 2>&1; do
    sleep 60
done
echo "$(ts) Q4 done; running Q5 transfer eval"

python3 experiments/31_honest_eval_omar/transfer_combined.py > logs/q5_transfer_combined.log 2>&1
echo "$(ts) Q5 done"

git add experiments/31_honest_eval_omar/ logs/q4_honest_eval_v2.log logs/q5_transfer_combined.log 2>&1 | tail -1
git commit -m "$(printf 'Q4 + Q5: honest train/test split eval + transfer/combined\n\nQ4: per-probe-per-layer test AUC across cyber_1/2/3 + refusal_gemma.\nQ5: cyber_3 ↔ refusal transfer + combined-data probe.\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>')" 2>&1 | tail -3
git push 2>&1 | tail -3 || { git pull --rebase 2>&1 | tail -3; git push 2>&1 | tail -3; }
echo "$(ts) commit + push done"
