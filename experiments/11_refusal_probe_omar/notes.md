# Experiment 11 (omar): Refusal probe + Arditi direction (Gemma 4-31B-it)

## Goal

Train a logistic-regression refusal probe on Gemma 4-31B-it residual activations
at layer 32 (mid-stack), and compute the Arditi 2024 unit-norm refusal
direction. Both products feed Level 2: the LR probe scores candidate edits, the
Arditi direction provides per-token attribution.

## Method

- **Data**: 832 of the 878 attack rollouts in
  `datasets/refusal_probes/gemma4_31b/attacks_full.jsonl` (extraction dropped 46
  on OOM during exp 9 GPU pass). Each `.pt` carries `(1, n_tokens, 5376)` fp16
  residuals at layer 32 plus the binary `label` (1 = refusal, 0 = compliance).
- **Pooling**: mean over attended tokens (fp32 upcast before sum to avoid fp16
  overflow on long prompts — this matters here, refusal prompts are typically
  longer than cyber prompts), and last-token pooling at the last attended
  position.
- **Probe**: `LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")`,
  5-fold StratifiedKFold (`random_state=0`).
- **Arditi direction**: `mu_pos − mu_neg` on mean-pooled features, unit-normed
  → `arditi_direction.npy`. Reported with the raw L2 norm of the diff so the
  scale is reproducible.

## Headline

| Pooling | 5-fold CV AUC | Acc |
|---|---:|---:|
| **mean** | **0.9265 ± 0.0134** | 0.829 |
| last-token | 0.9170 ± 0.0157 | 0.831 |

**Arditi-direction projection** (no LR fit, just `feature @ refusal_dir`):
AUC **0.7582**. The full LR probe adds ≈ +0.17 AUC over the bare 1-D direction
— the probe genuinely uses dimensions beyond the diff-of-means.

## Outputs

- `arditi_direction.npy` — unit vector ∈ R^5376 (mean-pool features, layer 32).
  Used by exp 12 for per-token attribution and exp 13 for an alternative
  scorer.
- `lr_probe.npz` — trained LR coefficients + bias on all 832 samples. (Final
  full-data fit; used by exp 13 for scoring rewrites.)
- `results.json` — full 5-fold metrics + Arditi-direction stats.
- `metrics.jsonl` — per-fold log.
- `auc_by_pooling.png`, `fold_auc.png` — plots.

## Implications

1. **The refusal probe at L32 mean-pool is around 0.93 AUC.** This is the
   number that fills row 4 of the 5-task table (Refusal-Gemma).
2. **The Arditi direction is real signal but a weak probe.** AUC 0.76 by
   itself is well above chance but well below the LR probe — i.e. refusal is
   NOT a single linear direction in the way Arditi 2024 demonstrated for
   chat-style refusal on smaller models. There is a primary direction plus
   additional structure the probe captures.
3. **Two `f`s for Level 2.** Per `experiments/extra_omar/notes.md`'s
   "two-different-f's" recommendation: use the Arditi direction for *attribution*
   (which tokens push refusal — exp 12) and the LR probe for *scoring*
   candidate edits (exp 13). They are not interchangeable.
4. **Long-prompt vulnerability.** Probe AUC is fine on the average prompt but
   collapses at strict production thresholds on long prompts — see exp 15's
   Kramár FNR analysis. Motivates exp 16 (Rolling probe).

## Caveats

- Single-layer extracts only (L32). We don't have a layer sweep for refusal
  the way we do for cyber. The L32 choice follows the original TODO and the
  Kramár paper's typical mid-stack pick; not optimised on this data.
- 832/878 samples used (46 dropped at extract time). Should be unbiased
  w.r.t. label but verify if a result hinges on sample composition.
- The Arditi direction is computed on mean-pooled features. A per-token
  variant (compute direction at each token position separately) would be a
  natural follow-up but isn't done here.
