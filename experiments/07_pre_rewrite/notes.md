# Experiment 07: PRE-style Level-2 Rewriting

## Goal

Apply the Probe-Rewrite-Evaluate (PRE) approach (Xiong 2025) to the 81 Gemma
attribution-eval prompts. Generate k=7 candidate rewrites per prompt, score each
with the trained refusal probe, pick the best (min refusal probability), and
verify whether the model actually flips from refusal to compliance.

Reports the three Level-2 metrics: **Pr(f|edit)**, **Pr(model|edit)**, **Pr(model|f)**.

## How to run

### Quick smoke test (CPU only, 3 prompts)
```bash
AIAAS_KEY=<your_key> python experiments/07_pre_rewrite/run.py \
    --limit 3 \
    --output experiments/07_pre_rewrite/smoke.jsonl
```

### Step A — generate rewrites only (CPU, any machine)
```bash
AIAAS_KEY=<your_key> python experiments/07_pre_rewrite/run.py \
    --output experiments/07_pre_rewrite/rewrites.jsonl
```
Produces 81 records, each with `rewrites` (list of 7 strings) and `best_rewrite`
(index 0 by default when no probe is given).

### Step B — full pipeline (GPU, on cluster)
```bash
AIAAS_KEY=<your_key> python experiments/07_pre_rewrite/run.py \
    --probe_path ./probes/constitutional/refusal_gemma4_31b_probe.pt \
    --model_key gemma4_31b \
    --verify_behavior \
    --output experiments/07_pre_rewrite/results.jsonl
```

### Qwen variant
```bash
AIAAS_KEY=<your_key> python experiments/07_pre_rewrite/run.py \
    --dataset refusal_qwen36 \
    --probe_path ./probes/constitutional/refusal_qwen36_probe.pt \
    --model_key qwen36 \
    --verify_behavior \
    --output experiments/07_pre_rewrite/results_qwen.jsonl
```

### Change rewriter model
```bash
# Faster / cheaper
--editor qwen3-30b

# Best multi-sentence rewrites (slow)
--editor qwen3-235b
```

## Output format

Each line in the output JSONL is one processed prompt:

```json
{
  "sample_id": "EVO_0012",
  "original_prompt": "...",
  "is_refusal_original": true,
  "rewrites": ["rewrite_1", ..., "rewrite_7"],
  "rewriter_elapsed_s": 4.3,
  "original_probe_score": 0.94,
  "probe_scores": [0.62, 0.41, 0.78, 0.35, 0.51, 0.67, 0.45],
  "best_idx": 3,
  "best_probe_score": 0.35,
  "probe_flipped": true,
  "best_rewrite": "...",
  "behavior_is_refusal": false,
  "behavior_flipped": true,
  "behavior_judge_reason": "model wrote the code",
  "behavior_elapsed_s": 12.1
}
```

Metrics are saved to `<output>.metrics.json`:
```json
{
  "n_total": 81,
  "pr_f_given_edit":     0.68,
  "n_probe_flipped":     55,
  "pr_model_given_edit": 0.42,
  "n_behavior_flipped":  34,
  "pr_model_given_f":    0.62
}
```

## Method

**Rewriting:** One AIaaS call per prompt produces k=7 paraphrases in a JSON
array. Using a single call (rather than k=7 separate calls) is 7× more efficient
while still providing diversity.

**Probe scoring:** Each rewrite is passed through the activation extractor
(Experiment 02) at the probe's training layer. The StreamingLinearProbe (Experiment 03)
returns P(refusal). The candidate with the lowest P(refusal) is selected.

**Behavior verification:** The selected best rewrite is rolled through Gemma
4-31B-it (greedy, 256 new tokens). MiniMax-M2.7 judges refusal vs compliance.

**Probe flip threshold:** P(refusal) ≥ 0.5 on original → flipped if < 0.5 on best.

## Results

(To be filled after running on cluster)

## Takeaways

(To be filled)
