# Experiment 16 (omar): Kramár MultiMax / Rolling Attention probes

## Goal

Implement the Kramár 2026 attention-probe family (4 architectures from §3.2):
`attention_kramar` (baseline), `multimax`, `rolling`, `rolling_multimax`.
Train on cyber_1 / cyber_2 / cyber_3 and Refusal-Gemma. Reuse for Level 2 PRE
scoring on the refusal task.

Spec lives in `IMPLEMENTATION_SPEC.md` (alongside this note).

## Why this experiment exists

Exp 15 showed that the Gemma refusal LR mean-pool probe has a clean
threshold-shift vulnerability on long prompts:

| Bucket | AUC | FNR @ 1% FPR |
|---|---:|---:|
| short | 0.920 | 49.4% |
| **long** | 0.918 | **79.5%** |

AUC barely moves but FNR at the strict-production threshold jumps from 49% to
80%. Kramár 2026's Selected Probe (Rolling Attention with `w=10`,
max-over-windows) is the paper's fix for exactly this failure mode. We
implement it here and re-run exp 15's Kramár FNR analysis with the new probe
to claim a gap closure (or document its limit).

## Method (per `IMPLEMENTATION_SPEC.md`)

Four architectures sharing the same MLP-transform front-end (`TransformMLP`,
2-layer ReLU, `d_hidden=100`, paper Eq. 5) and differing only in the
aggregation stage:

| Arch | Aggregation | Paper §/Eq. |
|---|---|---|
| `attention_kramar` | softmax over all tokens, weighted sum of values per head | §3.2, Eq. 7-8 |
| `multimax` | hard `argmax` per head, all-tokens scope | §3.2.1, Eq. 9 |
| `rolling` | softmax within sliding windows of width 10, max over windows | §3.2.2, Eq. 10 |
| `rolling_multimax` | hard `argmax` within window AND max across windows | combined ablation |

Hyperparameters from Kramár Appendix C: AdamW, lr=1e-4, weight_decay=3e-3,
1000 steps, n_heads H=10, d_hidden=100, window w=10. CPU-only training.

Training data pipeline reuses the existing extracts:
- Cyber: `/home/ubuntu/extracts/cyber_all_omar/`, 13 layers stored,
  pick layer 40 for cyber_1/cyber_2 and layer 35 for cyber_3.
- Refusal: `/home/ubuntu/extracts/gemma_refusal_L32/gemma4_31b/`, single
  layer 32.

Implementation tip from earlier (exp 11) experience: fp16 sums over long
prompts overflow → upcast to fp32 before reducing.

## Status (in flight at write time)

A background agent is implementing only the **Rolling probe (Eq. 10)** —
the spec's other 3 architectures are deferred for time. Outputs will land in
`probes.py`, `train.py`, and per-task `results/<task>_rolling.{pt,json}`.

The agent's report when complete should include:

- per-task AUC for Rolling vs the LR mean-pool baseline (cyber_1 0.988
  held-out, cyber_2 0.946-0.955, cyber_3 0.955, refusal 0.927)
- per-length-bucket AUC and (ideally) per-length-bucket FNR @ 1% FPR for
  refusal — re-running the exp 15 evaluation with the new probe so we
  can claim "Rolling narrows the gap from 49→80% to X%"
- wall-time per task

## Expected outcomes

1. **On cyber_1/2/3**: Rolling probe likely matches LR mean-pool. Cyber
   long prompts are *easier*, not harder (exp 10), so the long-context
   pathology Kramár targets isn't present. Expect a similar AUC ±0.005.
2. **On refusal**: Rolling probe should narrow the FNR @ strict-FPR gap.
   The paper reports going from 87.9% → 3.0% FNR with their Selected Probe
   on Gemini production traffic; we won't see anything that dramatic at
   our scale (832 samples, single layer, no production calibration), but
   even narrowing 80% → 60% would validate the architecture choice.
3. **If Rolling LOSES to LR baseline anywhere**: also a finding, would
   suggest the long-prompt distribution shift in our data isn't severe
   enough for Rolling's regularization to pay.

## What this gives the slides (whether the implementation lands or not)

- **If it lands**: the long-prompt FNR closure is the strongest possible
  story for the architecture choice — directly motivated by data, with the
  paper-replicated fix.
- **If only the spec lands (no trained results)**: the spec itself + the
  exp 15 motivation table is enough to argue the architecture choice in
  prose. Document that it didn't fit in the time budget.

## Outputs (planned)

- `probes.py` — `TransformMLP`, `RollingAttentionProbe` (and stubs / full
  classes for the other 3 if time).
- `train.py` — CLI: `--task <cyber_1|cyber_2|cyber_3|refusal_gemma>` `--arch
  <name>`. 5-fold CV + held-out test, length-bucket FNR for refusal.
- `results/` — per-task `.json` + `.pt` (probe weights).
- `notes.md` (this file) — updated with headline numbers when the agent's
  run completes.
