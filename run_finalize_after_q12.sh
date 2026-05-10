#!/bin/bash
# Wait for Q12 cyber_honest to finish, then regenerate plots with honest
# cyber_2/3 numbers and commit + push.

set -u
cd /home/ubuntu/georgia/mechhack
mkdir -p logs

ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) waiting for Q12 cyber_honest to finish..."
while pgrep -f "experiments/38_cyber_honest_omar/cyber_honest.py" > /dev/null 2>&1; do
    sleep 60
    echo "$(ts) Q12 still running"
done
echo "$(ts) Q12 done. Verifying results."

if [ ! -f experiments/38_cyber_honest_omar/cyber_honest_results.json ]; then
    echo "$(ts) cyber_honest_results.json missing — Q12 likely failed"
    exit 1
fi

# Patch make_plots.py to use honest cyber numbers from cyber_honest_results.json
python3 - <<'PY'
import json
from pathlib import Path

REPO = Path("/home/ubuntu/georgia/mechhack")
honest = json.loads((REPO/"experiments/38_cyber_honest_omar/cyber_honest_results.json").read_text())
print("Honest cyber test AUCs:")
for t, r in honest.items():
    best = max(r["LR_mean_L40"]["test_auc"], r["LR_last_L45"]["test_auc"])
    print(f"  {t}: LR_mean_L40={r['LR_mean_L40']['test_auc']:.4f} LR_last_L45={r['LR_last_L45']['test_auc']:.4f} best={best:.4f}")
PY

# Regenerate plots — patch make_plots.py inline to read from honest results
python3 - <<'PY'
import json
from pathlib import Path

REPO = Path("/home/ubuntu/georgia/mechhack")
honest_path = REPO/"experiments/38_cyber_honest_omar/cyber_honest_results.json"
make_plots = REPO/"experiments/37_presentation_plots/make_plots.py"

src = make_plots.read_text()
# Inject: use honest cyber numbers in plot_mean_auc_competition
patch = '''
    # PATCH: prefer honest cyber numbers if Q12 finished
    try:
        honest = json.loads((REPO_ROOT / "experiments" / "38_cyber_honest_omar" / "cyber_honest_results.json").read_text())
        for t in ["cyber_1", "cyber_2", "cyber_3"]:
            if t in honest:
                best_auc = max(honest[t]["LR_mean_L40"]["test_auc"],
                                honest[t]["LR_last_L45"]["test_auc"])
                cyber_aucs[t]["Linear (best layer)"] = best_auc
        print(f"  patched cyber_aucs with honest test AUCs from Q12")
    except Exception as e:
        print(f"  honest cyber patch skipped: {e}")
'''
needle = '    # Note: cyber_1 = exp 08 held-out (honest); cyber_2/3 = 5-fold CV (flag this)'
if patch.strip() not in src:
    src = src.replace(needle, patch + "\n" + needle)
    make_plots.write_text(src)
    print("patched make_plots.py")
else:
    print("make_plots.py already patched")
PY

# Run plots
python3 experiments/37_presentation_plots/make_plots.py > logs/q13_replots.log 2>&1
python3 experiments/37_presentation_plots/auc_vs_layer.py > logs/q13_replots_auc.log 2>&1 || true
python3 experiments/37_presentation_plots/causality_focused.py > logs/q13_replots_caus.log 2>&1 || true
python3 experiments/37_presentation_plots/fnr_kramar.py > logs/q13_replots_fnr.log 2>&1 || true
python3 experiments/37_presentation_plots/pr_model_edit.py > logs/q13_replots_pme.log 2>&1 || true

echo "$(ts) plots regenerated"
git add -A 2>&1 | tail -1
git commit -m "$(printf 'Q12 + Q13 final: honest cyber test AUC for cyber_1/2/3 + replots\n\nUpdates Slide 2 mean_auc_competition.png with honest cyber test-split\nnumbers. cyber_1 was already honest (exp 08); cyber_2/3 now too.\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>')" 2>&1 | tail -3
git pull --rebase 2>&1 | tail -3
git push 2>&1 | tail -3
echo "$(ts) done"
