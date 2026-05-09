# Experiment 15: MultiMax & Rolling Attention Probes (Omar)

## Goal

Implement MultiMax and Rolling Attention probes from Kramár et al. 2026 (GDM).
These architectures solve the long-context signal dilution problem that linear and
standard attention probes suffer from. The "Selected Probe" (Rolling Attention with
max-over-windows) is GDM's production pick for Gemini misuse detection.

## Method

See `IMPLEMENTATION_SPEC.md` for the full spec with equations, pseudocode, and
hyperparameters.

## How to run

```bash
# See IMPLEMENTATION_SPEC.md for full CLI examples
python experiments/15_multimax_probe_omar/train.py \
    --extracts_dir ./extracts/gemma4_31b_cyber \
    --dataset cyber --task cyber_1 \
    --arch multimax \
    --out_dir experiments/15_multimax_probe_omar/results
```

## Results

(To be filled after running)

## Takeaways

(To be filled)
