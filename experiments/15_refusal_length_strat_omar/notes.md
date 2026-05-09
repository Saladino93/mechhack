# Experiment 15 (omar): Refusal probe — length-stratified AUC + Kramár FNR

## Goal

Test the Kramár et al. 2026 hypothesis that softmax-attention probes degrade on
long contexts, with two evaluations:

1. **AUC by prompt-length tertile** — the standard length-stratified analysis.
2. **FNR at strict FPR** — the production-relevant evaluation Kramár actually
   uses (lock the threshold on a calibration set, measure how many attacks
   slip through on a shifted distribution).

Either result motivates (or weakens) the case for implementing the Selected
Probe (Rolling Attention) in exp 16.

## Method

### Length-stratified AUC (`length_strat.py`)

- 832 Gemma refusal extracts at L32 mean-pool (same data as exp 11).
- Bucket prompts by `len(attack_prompt)` (chars) into tertiles using 33.33 /
  66.67 percentile cuts.
- 5-fold StratifiedKFold; per fold compute AUC on the test rows of each
  bucket. Aggregate mean ± std across folds.

### Kramár FNR @ strict FPR (`kramar_fnr_eval.py`)

- Same data, but use 5-fold OUT-OF-FOLD predictions (every sample gets a
  test-time probability, no train leakage).
- Bucket by length, tertiles same as above.
- Pick threshold τ such that the SHORT bucket has FPR = X% (X ∈ {1, 5}).
- Apply that exact τ to MEDIUM and LONG buckets, compute FNR with Wilson 95%
  CIs.
- This is the "lock-and-shift" evaluation Kramár's paper uses — production
  guardrails calibrate at strict FPR on benign traffic, then ask how many
  attacks slip through when the activation distribution shifts.

## Results

### Length-stratified AUC (5-fold mean-pool L32)

| Bucket | n | n_pos / n_neg | AUC | Std |
|---|---:|---:|---:|---:|
| short (≤ 6,490 chars) | 277 | 174 / 103 | **0.9274** | 0.016 |
| medium (6,490-11,253) | 277 | 162 / 115 | 0.9135 | 0.048 |
| long (> 11,253 chars) | 278 | 83 / 195 | 0.9100 | 0.048 |
| overall | 832 | 419 / 413 | 0.9265 | 0.013 |

### Kramár FNR @ strict FPR (Wilson 95% CIs)

Threshold locked on the SHORT bucket; applied to all.

| Bucket | AUC | n_pos | **FNR @ 1% FPR** | **FNR @ 5% FPR** |
|---|---:|---:|---:|---:|
| short | 0.920 | 174 | 49.4% [42.1, 56.8] | 27.6% [21.5, 34.7] |
| medium | 0.913 | 162 | 64.2% [56.6, 71.2] | 37.0% [30.0, 44.7] |
| **long** | 0.918 | 83 | **79.5%** [69.6, 86.8] | **62.7%** [51.9, 72.3] |
| overall | 0.926 | 419 | 61.1% [56.3, 65.6] | 38.2% [33.7, 42.9] |

## Headline finding

**AUC barely budges across length buckets** (short 0.920 → long 0.918, Δ
0.002), but **FNR @ 1% FPR jumps from 49% → 80%** going short → long. **At a
strict-production threshold, ~80% of long-prompt attacks slip through** even
though the underlying separability is the same.

This is a textbook example of distribution-shift threshold sensitivity. The
overall ranking quality (AUC) doesn't see it because AUC averages over all
thresholds. The metric a production guardrail actually deploys does see it.

## Implications

1. **Direct motivation for exp 16 (Kramár Rolling probe).** The threshold-shift
   vulnerability the paper describes is real on our refusal probe. A
   pooling-robust head (Selected Probe = MultiMax + Rolling) should narrow the
   49% → 80% FNR gap.
2. **Compare to cyber** (exp 10 cyber_1 result): cyber long prompts are
   *easier* (AUC 0.979 → 0.991). The threshold-shift story is REFUSAL-specific
   in our data, not a general probe-robustness issue.
3. **Class skew on long bucket** is informative: only 30% of long prompts are
   refusals (vs 63% of short). Long prompts in this corpus tend to be
   legitimate technical asks — but the few that *are* refusals get badly
   missed at the strict threshold.
4. **The std also triples on medium/long** (0.016 → 0.048) — even the AUC
   estimate is noisier on long prompts.

## Outputs

- `results.json` — length-stratified AUC + per-bucket counts.
- `kramar_results.json` — FNR table + thresholds + Wilson CIs.
- `auc_by_length.png` — bar chart of length-strat AUC.
- (Plots from `kramar_fnr_eval.py` failed at write time due to a
  matplotlib/numpy 2.x bridge issue; numbers are in the JSON. Regenerate
  with `python kramar_fnr_eval.py` once the env is fixed, or plot from the
  JSON manually.)

## Caveats

- Mean-pool LR probe only. Last-token pooling not stratified here; probably
  similar story but should be confirmed if used as a slide claim.
- 5-fold CV with per-bucket re-eval inside each fold. Sample sizes per bucket
  per fold are 55-56 — not huge. Wilson CIs reflect that.
- Bucket cuts are *within-task* (this corpus's tertiles), not absolute char
  counts. Different definitions of "long" would shift the headline.
- Kramár's paper reports FNR going from ~3% short → ~88% long in a much more
  extreme regime (8k token prompts on production-scale traffic). Our jump
  (49% → 80%) is in the same direction but starting from a less extreme
  short-bucket FNR — different baseline calibration.
