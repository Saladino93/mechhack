# Experiment 10: PRE Rewrite (Omar)

## Goal

Apply PRE (Probe-Rewrite-Evaluate) to the 81 Gemma attribution-eval refusals.
Generate k=7 rewrites per prompt, score with probe, pick best, verify behavior.

## How to run

```bash
# Smoke test (3 prompts, CPU, needs AIAAS_KEY)
AIAAS_KEY=<key> python experiments/10_pre_rewrite_omar/run.py \
    --limit 3 \
    --output experiments/10_pre_rewrite_omar/smoke.jsonl

# Generate all rewrites (CPU)
AIAAS_KEY=<key> python experiments/10_pre_rewrite_omar/run.py \
    --output experiments/10_pre_rewrite_omar/rewrites.jsonl

# Full pipeline (GPU, cluster — after probes trained)
AIAAS_KEY=<key> python experiments/10_pre_rewrite_omar/run.py \
    --probe_path ./probes/constitutional/refusal_gemma4_31b_probe.pt \
    --model_key gemma4_31b \
    --verify_behavior \
    --output experiments/10_pre_rewrite_omar/results.jsonl
```

## Results

(To be filled after running)

## Takeaways

(To be filled)
