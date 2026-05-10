# Presentation figures — README, methodology, and reading guide

Five publication-quality figures in `figures/`, plus the source script
`make_plots.py` (and `auc_vs_layer.py` for the per-layer plot).

---

## Figure 1 — `refusal_probe_family_comparison.png`

**Refusal-Gemma probe family comparison on honest test split.**

- **X-axis**: probe family / variant.
- **Y-axis**: AUC.
- **Bars**: test AUC (color = performance band). Bar height is the headline.
- **Red ✕ markers**: train AUC. The vertical gap between train ✕ and test bar
  shows overfitting (large gap = probe memorized training set).
- **Dashed grey line**: chance AUC = 0.50.

What it shows:
- LR last-token at L40-L45 wins. Test AUC ≈ 0.94, train ≈ 1.0 → small overfit gap.
- LR multi-layer concat (69,888-dim) drops to 0.91, train ≈ 1.0 → bigger gap.
- Pleshkov quadratic d=16 ≈ 0.85, well below LR → PCA bottleneck.
- Minimal MultiMax (per-token linear → max, no MLP transform) at 0.83.
  Same architecture with mean-pool gets 0.92 → max-pool needs the MLP transform.
- TF-IDF baseline 0.88. Activation lift = +0.07.

> **Why Constitutional probe is excluded from this plot**: 8.9M parameters
> (13×5376=69,888 input → Dense(128) → Dense(1)) on 555 training samples.
> Train AUC = 0.996, test AUC = 0.500 (random). Catastrophic overfit; the
> probe memorized noise. Same lesson the multi-layer-concat-LR shows but
> 100× more parameters, so 100× more catastrophic. Documented in
> `experiments/30_rewrites_causality_omar/extra_probes_summary.json`.

> Read top-down for "winners" and bottom-up for "what doesn't work."

---

## Figure 2 — `refusal_layer_sweep.png`

**Pleshkov 13-layer sweep on refusal: linear vs linear-on-PCs vs quadratic.**

- **X-axis**: residual-stream layer (0 → 60).
- **Y-axis**: 5-fold CV AUC, error bars = ±1σ across folds.
- **Three lines**:
  - Blue: Linear LR on raw 5,376-d activations (full capacity).
  - Orange: Linear LR on the 16-D PCA projection (apples-to-apples
    bottleneck baseline for Pleshkov).
  - Green: Pleshkov quadratic = 16-D PCA → degree-2 polynomial features → ridge.

What it shows:
- Linear-on-raw rises monotonically from L0=0.85 → L40 ≈ 0.94.
- Linear-on-16-PCs and Pleshkov-quadratic both top out around 0.86 — they
  *track each other*. The quadratic terms add ~zero on top of the bottleneck.
- The 0.07-AUC gap between blue and (orange, green) is the **PCA bottleneck**.
- Conclusion: Pleshkov 2026's quadratic probe doesn't beat linear on Gemma
  refusal — the limitation is dimensionality, not nonlinearity.

---

## Figure 3 — `cyber_layer_sweeps.png`

Same 3-line plot for each cyber task in 3 panels (cyber_1 / cyber_2 / cyber_3).

What it shows:
- Same pattern: linear ≫ Pleshkov ≈ linear-on-PCs.
- cyber_1 reaches ~0.99 (easiest task), cyber_2/3 ~0.95 (harder).
- Pleshkov never beats linear, on any task or layer.

---

## Figure 4 — `mean_auc_competition.png` ⭐ COMPETITION HEADLINE

**Mean AUC across 4 tasks (cyber_1/2/3 + refusal_gemma; Qwen excluded).**

The challenge metric is `mean_auc = (auc_cyber1 + auc_cyber2 + auc_cyber3 +
auc_refusal_gemma + auc_refusal_qwen) / 5`. We report the 4-task mean.

- **X-axis**: probe family.
- **Bars (4 colors)**: per-task test AUC.
- **Black line + dot + error bar**: **mean across the 4 tasks ± standard
  deviation**. *This is the rules-compliant metric.*

Result:
| Probe family | mean AUC | std |
|---|---:|---:|
| Linear (best layer) | **0.960** | 0.020 |
| Linear-on-16-PCs | 0.916 | 0.036 |
| Pleshkov d=16 | 0.902 | 0.041 |
| TF-IDF | 0.900 | 0.031 |
| Random | 0.500 | 0.000 |

> Linear at the best layer wins by 0.044 over Linear-on-16-PCs and 0.060 over
> Pleshkov. Pleshkov ≈ Linear-on-16-PCs ≈ TF-IDF — the PCA bottleneck reduces
> deep representation to TF-IDF level.

⚠️ **Caveat**: cyber_1 number = honest 923-sample held-out test (exp 08).
cyber_2/3 numbers in this plot are 5-fold CV (mixed train+test) because
running honest cyber test eval was killed for disk-IO budget. Refusal is
honest test split (Q4r). Cyber_2/3 honest test eval is a TODO before final
submission.

---

## Figure 5 — `causality_scatter.png` ⭐ LEVEL 2 HEADLINE

**Per-probe Pr(f|edit) vs Pr(model|f flipped) on substantial paraphrases.**

- **X-axis**: Pr(f flipped | edit) — how often the probe says "this edit
  flipped my prediction." High = permissive probe.
- **Y-axis**: Pr(model flipped | f flipped) — given the probe said flip,
  how often did Gemma actually produce a non-refusal response. High = causal.
- **Marker size**: number of probe-flips (n).
- **Quadrants**:
  - Top-left = **causal + robust** (rare flips, high concordance) ✓
  - Top-right = **causal + permissive** (many flips, mostly track model)
  - Bottom-right = **gamed f** (many flips, model doesn't follow)
  - Bottom-left = **over-robust** (rare flips, those that exist don't track)

What it shows (n_orig_refusal = 79 prompts):
- LR last-tok L40-L60 cluster top-center: Pr(f|edit) ≈ 0.17, Pr(m|f) ≈ 0.92.
  This is the textbook causal regime.
- COMBINED (cyber_3 + refusal) probes are slightly to the right and below LR:
  more permissive, slightly less causal. Combining harms refusal causality.
- Pleshkov_d16_combined sits at (0.13, 0.22) — gamed: 7 of 9 flips don't
  flip the model.

---

## Figure 6 (additional) — `refusal_auc_acc_vs_layer.png`

**AUC + Accuracy vs layer (refusal_gemma honest test split).**

- **Two panels** (AUC, Accuracy at τ=0.5).
- **Two lines per panel**: mean-pool (blue), last-token (red).
- **Shaded bands**: 95% bootstrap CI (n_boot = 500).

What it shows:
- AUC peaks at L40 last-tok (0.94 ± ~0.02).
- Accuracy peaks similarly. Both metrics agree on best layer.
- Bootstrap CI shows where the difference between mean-pool and last-token
  is statistically meaningful (CIs don't overlap at L40-L60).

---

# Methodology

## Data and splits

- **Cyber**: `datasets/cyber_probes/{train,test}.jsonl` — 5,079 train + 2,180
  test rollouts across 4 categories (`benign / dual_use / high_risk_dual_use
  / prohibited`). Splits are deterministic.
  - cyber_1 = `dual_use` vs `benign`
  - cyber_2 = `high_risk_dual_use` vs (`dual_use ∪ benign`)
  - cyber_3 = `prohibited` vs (`high_risk_dual_use ∪ dual_use ∪ benign`)
- **Refusal-Gemma**: `datasets/refusal_probes/gemma4_31b/{train,test}_split.jsonl` —
  589 train + 289 test attack prompts; 832 of 878 had successful Gemma extracts.
  We use 555 train / 277 test rows where the extract succeeded.
- **Qwen** excluded (model not downloaded).

## Probe architectures

| Probe | Description | Params |
|---|---|---:|
| LR mean-pool | Logistic regression on (mean over tokens of one layer) | 5,376 + 1 |
| LR last-token | Same, last token's residual at one layer | 5,376 + 1 |
| LR multi-layer concat | LR on 13 layers × 5,376 = 69,888-dim concat | 69,888 + 1 |
| LR mean-of-layers | LR on `mean_L(activations)` (5,376-d) | 5,376 + 1 |
| LR max-of-layers | LR on `max_L(activations)` (5,376-d) | 5,376 + 1 |
| MLP probe | mean-pool L40 → Dense(256) → ReLU → Dropout(0.2) → Dense(64) → ReLU → Dense(1) | 1,394,945 |
| Constitutional probe | 13-layer concat → Dense(128) → ReLU → Dropout(0.3) → Dense(1) | 8,945,793 |
| Pleshkov quadratic d=16 | Z-score → PCA(d=16) → degree-2 features (152) → Ridge(α∈{0.1,1,10,100} CV) | 152 |
| Minimal MultiMax | per-token Linear(d_model→1) → max over tokens | 5,377 |
| Minimal mean (ablation) | per-token Linear → mean over tokens | 5,377 |
| Kramar Rolling probe | TransformMLP(5376→100) + 10-head softmax over rolling-window-10 → max-over-windows | ~544k |
| TF-IDF + LR | TfidfVectorizer(word/char 1-2/3-5 gram) → LR | varies |

## Training protocol (rules-compliant)

- **Train only on `split=='train'`** rows.
- **Evaluate only on `split=='test'`** rows.
- LR: `sklearn.LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')`.
  C-sweep where indicated (multi-layer concat: C=0.1).
- MLP / Constitutional: AdamW lr=1e-3, wd=1e-4 to 1e-3, 30-50 epochs,
  batch 16-32, BCE loss.
- Pleshkov: per-fold inner alpha CV in {0.1, 1, 10, 100} on 80/20 inner
  split, then refit at chosen alpha.

## Metric

`AUC` = `sklearn.metrics.roc_auc_score(y_test, probe.predict_proba(X_test)[:, 1])`

`Accuracy` = `(probe_score > 0.5) == y_test` averaged.

**Mean across 5 tasks (rules):**
`mean_auc = (auc_cyber1 + auc_cyber2 + auc_cyber3 + auc_refusal_gemma + auc_refusal_qwen) / 5`.
We report mean across 4 (Qwen excluded), with sample standard deviation
across the 4 tasks as the variance metric.

## Error bars

Two different error bars appear in the figures:

1. **5-fold CV ±1σ** (figures 2, 3 — refusal/cyber layer sweeps):
   the probe is fit 5× on different 80/20 splits of the *training* set;
   error bar = standard deviation of the 5 fold-AUCs. Reflects variance
   of the probe-fitting procedure but doesn't address train/test
   contamination on cyber_2/3 because the split itself sampled across
   the deterministic train/test boundary.

2. **Bootstrap 95% CI** (figure 6 — AUC/ACC vs layer):
   resample the test set with replacement 500 times; CI = 2.5/97.5
   percentiles of bootstrap-AUC. Reflects sampling uncertainty *of the
   test-set evaluation only*, given a fixed probe. This is the right
   error bar for "how reliable is this single test-set AUC number."

3. **Wilson 95% CI** (Pr metrics — Level 2): for the binomial
   proportions Pr(f|edit), Pr(model|edit), Pr(model|f flipped). Standard
   form: p ± z × sqrt(p(1-p)/n + z²/(4n²)) / (1 + z²/n).

## Honest test-split status per task

| Task | Honest test eval status |
|---|---|
| Cyber-1 | ✅ Honest 923-sample held-out test (exp 08), AUC 0.988 |
| Cyber-2 | ⚠️ Only 5-fold CV (0.946) — honest test pending (Q4 cyber killed for time) |
| Cyber-3 | ⚠️ Only 5-fold CV (0.955) — honest test pending |
| Refusal-Gemma | ✅ Honest test split (Q4r) — n_test=277, AUC 0.9417 |

If we have time before deadline, run a slim cyber-honest-test eval
(`cyber_test_extract` is at 100%; load → fit on train → eval on test).
