# Experiment 01: Data Exploration

## Goal
Understand the datasets before building anything. Class distributions, token lengths, class balance per probe task, example prompts.

## Method
Loaded all JSONL files, computed summary stats, produced distribution plots.

## Results

### Cyber probes (7,259 samples)
- **Class sizes:** prohibited=2574, benign=2238, high_risk_dual_use=1495, dual_use=952
- `dual_use` is the smallest class (952) — this is the positive class for Probe-1
- Token lengths: mean=1972, median=1607, range=[15, 8189]
- All categories have similar token length distributions (no length-based shortcut)
- Sources: gepa_attack (4292), swebench (1130), hard_neg (1080), wildchat (717), purplelama (40)

**Probe task balance (test split):**
| Probe | Positive | Negative | Pos ratio |
|-------|----------|----------|-----------|
| Probe-1 | 267 (dual_use) | 656 (benign) | 0.29 |
| Probe-2 | 443 (high_risk_dual_use) | 923 (du+benign) | 0.32 |
| Probe-3 | 814 (prohibited) | 1366 (rest) | 0.37 |

All probes are imbalanced toward negatives — AUC is robust to this but worth noting for threshold-based metrics.

### Refusal probes

**Gemma 4-31B-it (878 samples):**
- Refusal rate: 48% — nearly balanced
- Refusal prompts are shorter (mean=2021 tokens) vs compliance (mean=3631)
  - **This is a potential confounder!** A naive model could use length as a signal
- Attribution eval: 81 samples, all refusals, tokens ≤2048
- Judge scores bimodal: most are ≥0.9 (high confidence), very few in the middle

**Qwen 3.6-27B (878 samples, same prompts):**
- Refusal rate: 31% — lower than Gemma (Qwen complies more)
- Same token length pattern: refusal shorter (mean=1788) vs compliance (mean=3348)
- Attribution eval: 62 samples (fewer refusals to work with)
- 16 samples have None judge scores

### Key observation: same prompt, different outcomes
EVO_0001 (DLL injection prompt): Gemma **complies**, Qwen **refuses**. Shows the models have genuinely different refusal boundaries on the same content.

## Takeaways

1. **Token length correlates with refusal** — shorter prompts get refused more. Our probes should not just learn a length heuristic. Worth checking if residual activations encode something beyond length.
2. **Cyber probes are content-based** (no model behavior), so no length confounder there — but `dual_use` being small (952) may make Probe-1 harder.
3. **Gemma is better balanced** (48% refusal) and has more attribution eval samples (81 vs 62) — confirms it as the priority target.
4. **Judge scores are high-confidence** — most ≥0.9, so label noise is low.
5. **Next step:** Extract residuals (Experiment 02), then check if a simple refusal direction separates the classes (Experiment 03).
