# Experiment 10 (omar): Length-stratified AUC for cyber_1/2/3

## Goal

For each cyber task, bucket prompts by char-length tertiles and compute
5-fold CV AUC per bucket at the task's best mean-pool layer. Tests
whether the cyber probes degrade on long prompts (the production-relevant
distribution-shift question Kramár 2026 raises).

## Method

- Tasks + best layers (from exp 03, 06, 07): cyber_1 mean L40, cyber_2 mean
  L40, cyber_3 mean L35.
- Load features from `/home/ubuntu/extracts/cyber_all_omar/`, mean-pool, fit
  LR with 5-fold StratifiedKFold (`random_state=0`), evaluate per-bucket
  inside each test fold.
- Bucket cuts at 33.33 / 66.67 percentiles of `len(prompt)` (chars), within
  each task.

## Status (cyber_1 done, cyber_2/3 in flight at write time)

### cyber_1 (mean L40)

- n=999 (pos=500, neg=499)
- char-length bucket boundaries: short ≤ 4,626; medium 4,626-9,675; long > 9,675; max 32,476.

| Bucket | n / fold | AUC | Std |
|---|---:|---:|---:|
| short | 67 | 0.9793 | 0.015 |
| medium | 67 | 0.9797 | 0.010 |
| **long** | 67 | **0.9914** | 0.011 |

**On cyber_1, long prompts are EASIER, not harder.** AUC actually rises by
~0.012 from short → long. Opposite of the Kramár prediction.

Plausible reasons:
- More tokens → mean-pool averages over more signal-bearing positions →
  better SNR on the dual_use vs benign discrimination.
- Long cyber prompts in this corpus tend to be elaborate scenario
  descriptions of dual-use technical content (more clearly "dual-use").
  Short prompts may be terse questions where the dual_use signal is
  underdeveloped.

### cyber_2, cyber_3

In flight at write time. Will land in `results.json` and `auc_by_length.png`
when the script completes. The cyber_2/3 tasks have a finer-grained boundary
(harm-tier discrimination) and a more class-skewed selection (harm-positive
samples are scarcer), so length sensitivity could differ from cyber_1.

## Implications

- **Kramár MultiMax/Rolling motivation lives ELSEWHERE on cyber.** The
  cyber_1 result rules out the long-prompt-degradation argument for
  implementing the Selected Probe on cyber tasks.
- **The story is REFUSAL-specific.** See exp 15: refusal AUC drops modestly
  (0.927 → 0.910 short → long) AND FNR @ 1% FPR jumps from 49% → 80%.
  That's where Kramár-style aggregation has clear value.
- The asymmetry (long EASIER on cyber, long HARDER on refusal) is itself
  interesting and worth a slide bullet.

## Outputs (pending completion)

- `results.json` — per (task, bucket) AUC mean ± std + bucket boundaries
  in chars.
- `auc_by_length.png` — three task-curves on shared axes.
