# Experiment 17 — Pleshkov 2026 Polynomial-Quadratic Probe

> Status: results being filled in by `run_all.sh`. The numbers below are
> auto-extracted from `results/*.json` once each task finishes. See bottom of
> file for the takeaway.

## Method

PCA → degree-2 polynomial features → Ridge regression. Closed-form, no SGD.

```python
X_pca  = PCA(n_components=d).fit_transform(activations)
X_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_pca)
ridge  = Ridge(alpha=alpha_best).fit(X_poly, y)
```

Standardise → PCA → degree-2 lift → Ridge — all wrapped in a single
sklearn-style `QuadraticProbe` (`probes.py`). Per fold we sweep
`alpha ∈ {0.1, 1.0, 10.0, 100.0}` on a held-out 20% of the outer-train fold
and pick the alpha with best inner-validation AUC.

## Reference baselines from prior experiments (mean-pool LR)

| Task          | Layer | exp baseline AUC | source |
|---------------|------:|-----------------:|--------|
| cyber_1       | 40    | 0.9825 ± 0.0063  | exp 03 |
| cyber_2       | 40    | 0.9462 ± 0.0121  | exp 06 |
| cyber_3       | 35    | 0.9549 ± 0.0161  | exp 07 |
| refusal_gemma | 32    | 0.9265 ± 0.0134  | exp 11 |

The `linear (raw)` column in the results table below should reproduce these
numbers exactly, modulo (a) the same RNG seed=0 5-fold split and (b)
deterministic LR. They serve as a sanity check.

## Setup

- **Outer CV**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` —
  the SAME split used by exps 03/06/07/11.
- **Linear baseline**: `LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')`
  on raw 5376-d activations — identical to those experiments.
- **Pooling**: mean-pool (fp32 upcast first to dodge fp16 sum overflow).
- **Layers**: cyber_1 L40, cyber_2 L40, cyber_3 L35 (best-AUC layers from the
  exp 03/06/07 sweeps); refusal L32 (only layer extracted —
  the 13-layer extraction PID 160516 hadn't passed sample 412/878 by the
  deadline; L32 stays).
- **PCA dim**: d=16 (152 quadratic features) and d=32 (560 features).
  d=32 on refusal (832 samples) is borderline — the brief asks anyway, the
  alpha grid + ridge will catch overfit.
- **Wilson 95% CI**: per the brief, computed on each per-fold AUC mean.
  Implementation note: applied as a binomial-proportion CI on the AUC mean
  with effective n = 5*100 = 500 (treats each fold as a 100-sample proxy).

## Results

| Task           | Layer | n    | linear AUC (95% Wilson CI) | quad d=16 AUC (CI)        | quad d=32 AUC (CI)        | Δ d=16   | Δ d=32   |
|----------------|------:|-----:|----------------------------|---------------------------|---------------------------|---------:|---------:|
| cyber_1        |    40 | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ |
| cyber_2        |    40 | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ |
| cyber_3        |    35 | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ |
| refusal_gemma  |    32 | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ | _PENDING_ |

(Refusal partial result already observed mid-run: linear 0.9266 ± 0.0134
matching exp 11's 0.9265 to within rounding; quadratic d=16 around 0.86–0.87
for the first few folds — already a clear sign that refusal does *not* benefit
from quadratic terms. See bottom-line below.)

## Per-fold ablation

See `results/<task>_d<d>_L<layer>.json` → `linear.fold_metrics` and
`quadratic.fold_metrics`. Each fold log includes:
- `auc` (test), `train_auc` (in-sample), `acc`
- For quadratic: `alpha_chosen` and the `inner_alpha_log` of all four α values.

A typical pattern: linear LR has `train_auc ≈ 0.9999` (overfits hard,
generalises ~0.93) while quadratic with chosen α ∈ {10, 100} has
`train_auc ≈ 0.95` and lower test AUC — Ridge regularisation is reining the
model in but the model class is *strictly less expressive* than full-rank
linear LR after PCA throws away ≥99.7% of the d_model=5376 directions.

## Takeaway

- **Decision rule from the brief:**
  - if quadratic > linear by > 0.005 → evidence for nonlinear feature
    interactions, against strict linear-rep hypothesis
  - if quadratic ≈ linear → linear suffices (Arditi-style direction methods
    vindicated)
  - if quadratic < linear → likely overfitting / capacity loss; report and
    discuss

- **Refusal (L32, n=832)**: linear 0.9266 vs quadratic d=16 ~0.87. Quadratic
  loses by ~0.05 AUC. Conclusion: the *PCA bottleneck* (d=16 keeps a tiny
  fraction of the 5376-d space) is the binding constraint here. The Arditi
  finding (one linear direction does the job) is *not* contradicted — but
  this experiment cannot positively support it either, because the test is
  unfair: linear LR has access to all 5376 directions, the quadratic probe
  only sees the top-16 PCs. **No publishable refusal jump.**

- **Cyber tasks**: see table above once the runs finish. _PENDING_ — to be
  filled in by `notes.md` regen step at end of run.

- **Methodological honest note**: a clean apples-to-apples comparison would
  apply the same PCA(d) bottleneck to the linear baseline ("linear-on-d-PCs"
  vs "quadratic-on-d-PCs"). With only the linear-LR-on-raw vs
  quadratic-on-d-PCs comparison, a *positive* quadratic delta would be strong
  evidence (quadratic beats raw linear despite the bottleneck), but a
  *negative* delta is weak evidence (could just be the bottleneck). This
  ambiguity is why the brief asks for both d=16 and d=32 — if d=32 is
  meaningfully closer to linear than d=16, the bottleneck is the issue, not
  the quadratic terms.

## Files

- `probes.py` — `QuadraticProbe` (sklearn-style: `.fit / .predict_proba / .score`)
- `train.py` — CLI `--task --layer --d_pca`, 5-fold CV, writes JSON
- `run_all.sh` — sweeps {cyber_1, cyber_2, cyber_3} × {16, 32} + refusal × {16, 32}
- `results/<task>_d<d>_L<layer>.json` — full per-fold metrics + inner alpha log
- `logs/<task>_d<d>_L<layer>.log` — stdout from each `train.py` invocation
