#!/bin/bash
# Train Kramar 2026 Selected Probe (RollingAttentionProbe) on all 4 tasks.
# CPU-only; ~10–30 min per task.
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root

mkdir -p experiments/16_multimax_probe_omar/results

python3 experiments/16_multimax_probe_omar/train.py --task all \
    --out_dir experiments/16_multimax_probe_omar/results \
    2>&1 | tee experiments/16_multimax_probe_omar/results/run_all.log
