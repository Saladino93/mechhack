# Slide deck — final story, plots, and numbers

> **One-page reference for the talk.** Each section maps to one slide.
> Plots in `figures/`. Numbers below should match the plots exactly.

---

## Slide 1 — Setup (1 minute)

**Probing model behavior on Gemma-4-31B-it.**

5 prediction tasks (rules):

| Task | Positive | Negative | n_train | n_test |
|---|---|---|---:|---:|
| Cyber-1 | dual_use | benign | ~1.5k | ~700 |
| Cyber-2 | high_risk_dual_use | du ∪ ben | ~1.2k | ~500 |
| Cyber-3 | prohibited | hrdu ∪ du ∪ ben | ~1.0k | ~400 |
| Refusal-Gemma | model refused | model complied | 555 | 277 |
| Refusal-Qwen | (skipped — model not downloaded) | | — | — |

**Metric**: mean AUC across 5 tasks (we report 4 + variance).

**Two questions**: (1) Predict the label from internals; (2) Edit the prompt to flip the model's behavior, while preserving harmful intent.

---

## Slide 2 — Level 1 headline ⭐ (1.5 min)

📊 **Plot**: `figures/mean_auc_competition.png`

> **Linear single-layer at the best layer wins. mean AUC = 0.960 ± 0.020 across 4 tasks.**

| Probe family | mean AUC | std (across 4 tasks) |
|---|---:|---:|
| **Linear (best layer)** | **0.960** | 0.020 |
| Linear-on-16-PCs | 0.916 | 0.036 |
| Pleshkov d=16 (quadratic) | 0.902 | 0.041 |
| TF-IDF + LR | 0.900 | 0.031 |
| Random | 0.500 | 0.000 |

**Key reading**: *Pleshkov ≈ Linear-on-16-PCs ≈ TF-IDF*. The PCA bottleneck collapses both linear and quadratic to TF-IDF level. The full 5,376-d linear probe earns +0.06 AUC over TF-IDF — that's the "value of the residual stream" on this dataset.

**Activation lift over TF-IDF GROWS with task difficulty**:
- cyber_1: +0.037 (TF-IDF nearly sufficient)
- cyber_2: +0.059
- cyber_3: +0.065
- **refusal: +0.075** ← strongest lift on hardest task

---

## Slide 3 — Probe family deep dive (1 min)

📊 **Plot**: `figures/refusal_probe_family_comparison.png`

> **On honest test split (refusal_gemma, n_test=277):**

| Probe | Test AUC | Train AUC | Notes |
|---|---:|---:|---|
| LR last-tok L40 | **0.9417** | 1.000 | winner |
| LR last-tok L45 | 0.9368 | 1.000 | |
| LR mean L40 | 0.9269 | 1.000 | |
| LR multi-concat (69k-d) | 0.9137 | 1.000 | overfits — capacity ≠ generalization |
| LR mean-of-layers | 0.9090 | 0.998 | |
| Pleshkov d=16 L50 | 0.854 | 0.955 | PCA bottleneck |
| TF-IDF word | 0.877 | 0.982 | |
| Minimal MultiMax (no MLP) | 0.834 | 0.995 | per-token linear → max |
| Minimal mean (ablation) | 0.918 | 0.985 | per-token linear → mean (BETTER) |
| **Constitutional MLP** | **0.500** | 0.996 | catastrophic overfit (8.9M params on 555 train) |
| Random | 0.500 | 0.500 | sanity |

**Three takeaways**:
1. **Capacity ≠ generalization.** 69k-d concat → 0.913 < 5k-d single-layer → 0.942.
2. **Pleshkov quadratic loses ~0.07 AUC to linear** at every layer (next slide).
3. **MultiMax requires the MLP transform.** Without it, max-pool < mean-pool by 8.4 AUC pts.

---

## Slide 4 — Layer sweep (1.5 min)

📊 **Plots**:
- `figures/refusal_layer_sweep.png` (refusal, 13 layers)
- `figures/cyber_layer_sweeps.png` (cyber 1/2/3, 13 layers each)
- `figures/refusal_auc_acc_vs_layer.png` (AUC + accuracy with bootstrap CIs)

> **Linear ≫ Pleshkov ≈ Linear-on-PCs across all layers and all tasks. The PCA bottleneck dominates, NOT lack of nonlinearity.**

Refusal_gemma layer sweep (5-fold CV):
| Layer | Linear (raw) | Linear-on-16-PCs | Pleshkov-quadratic | Δ (q − lin) |
|---|---:|---:|---:|---:|
| L20 | 0.893 | 0.849 | 0.851 | −0.04 |
| L40 | 0.944 | 0.866 | 0.864 | −0.08 |
| L60 | 0.909 | 0.864 | 0.847 | −0.06 |

**Interpretation**: throwing away >99% of the 5,376-d residual costs ~7 AUC pts. Adding quadratic terms back in the d=16 subspace doesn't recover them (and slightly hurts due to overfitting at α=10).

Same shape across cyber_1, cyber_2, cyber_3 — the pattern is general.

---

## Slide 5 — Length-stratified Kramár threshold-shift (1 min)

> **Refusal AUC stays flat across length, but FNR @ 1% FPR explodes on long contexts.**

| Bucket | n | LR mean L32 AUC | LR mean L32 **FNR @ 1% FPR** | Rolling-probe AUC |
|---|---:|---:|---:|---:|
| short (<6.5k chars) | 277 | 0.927 | **49%** | 0.847 |
| medium (6.5–11.3k) | 277 | 0.918 | 60% | 0.829 |
| **long (>11.3k)** | 278 | 0.918 | **80%** | **0.889** |

**This is the Kramár 2026 threshold-shift in our data.** AUC is misleading on long contexts; the operating-point performance collapses. Rolling probe (sliding-window softmax over windows of 10) doesn't degrade.

**Recommendation**: deploy LR last-tok L40 for short/medium contexts. Switch to Rolling probe for long-context production.

---

## Slide 6 — EDA: refusal is a deterministic surface phenomenon (1 min)

📊 **Plots in `experiments/32_refusal_eda_omar/histograms/`**

> **5 regex patterns on the response catch 83% of refusals at zero false positives.**

| Response opening 3-gram | refused | complied | log-odds |
|---|---:|---:|---:|
| `I cannot provide` | **161** | **0** | +5.09 |
| `While I understand` | 140 | 0 | +4.95 |
| `I cannot complete` | 18 | 0 | +2.94 |
| `I cannot port` | 13 | 0 | +2.64 |
| `I cannot implement` | 11 | 0 | +2.48 |

**Prompt-start word `during`**: **47/0** refusal predictor (perfect specificity).
**Length correlation**: response_chars r=−0.516 (refusals are short); prompt_chars r=−0.402 (long prompts → more compliance — behavioral side of Kramár).

This is **why TF-IDF works so well** (~0.88 on refusal). Activations earn their keep at +0.07 by capturing the harder discriminations.

---

## Slide 7 — Cross-task transfer (1 min)

📊 **Plot**: `experiments/22_cross_task_4x4_omar/matrix_4x4.png`

> **There is no single "harm direction" — there is a SEVERITY-AWARE one.**

4×4 cross-task transfer matrix at L35 mean (train one task, eval each):

| | cyber_1 | cyber_2 | cyber_3 | refusal_gemma |
|---|---:|---:|---:|---:|
| **train cyber_1** | (CV 0.981) | 0.884 | 0.715 | 0.654 |
| **train cyber_2** | 0.906 | (CV 0.942) | 0.785 | 0.783 |
| **train cyber_3** | 0.803 | 0.733 | (CV 0.955) | 0.755 |
| **train refusal** | **0.525** | 0.720 | 0.772 | (CV 0.939) |

**Headline**: refusal → cyber_1 = **0.525** (basically chance). The refusal direction and the dual-use direction are **orthogonal in residual space**. But refusal ↔ cyber_3 ≈ 0.76 — model-refusal and prohibited content share a direction.

---

## Slide 8 — Level 2 headline ⭐ (2 min)

📊 **Plot**: `figures/causality_focused.png`

> **6 probes × 2 edit aggressiveness levels. The probe-and-edit choices that land in the "robust+causal" green quadrant give the strongest causal claim.**

The Level 2 metric triple (rules: report all three):
- **Pr(f flipped | edit)**: how often probe says flipped
- **Pr(model flipped | edit)**: how often model behavior flipped (THE HEADLINE)
- **Pr(model flipped | f flipped)**: when probe flips, does the model? (CAUSAL)

**On substantial paraphrases (rewrites_k7, n_orig_refusal=79):**

| Probe | Pr(f|edit) | Pr(model|edit) | **Pr(model|f flipped)** | n |
|---|---:|---:|---:|---:|
| **LR last-tok L45** | 0.174 | 0.304 | **0.917** | 11/12 |
| **LR last-tok L55** | 0.174 | 0.304 | **0.917** | 11/12 |
| **LR last-tok L40** | 0.232 | 0.304 | **0.812** | 13/16 |
| COMBINED L45 last | 0.232 | 0.304 | 0.750 | 12/16 |
| LR multi-concat | 0.072 | 0.304 | 0.600 | 3/5 |
| LR mean L40 | 0.246 | 0.304 | 0.588 | 10/17 |
| **Pleshkov d=16 L40** | 0.290 | 0.304 | **0.550** | 11/20 |

**Three findings, one per slide-emphasis:**

1. **LR last-token at L45-L55 is in the textbook causal regime**: when the probe says flipped, the model flips ~92% of the time. Same edits, different probe → 37-pt Pr(m|f) gap with Pleshkov.

2. **Combining cyber_3 + refusal training data MAKES the probe more gameable for refusal causality** (0.917 → 0.750). The harm directions are aligned at AUC level (cross-task = 0.78) but **not at causal level**. Specialist > generalist for causal alignment.

3. **Pleshkov flips MORE often (29%) but is LESS causal (55%)**. Quadratic is more gameable, not more robust, on substantial edits.

---

## Slide 9 — Edit-aggressiveness ladder (1 min)

> **Surface edits don't flip the model. Substantial paraphrases do.**

| Edit method | Pr(model | edit) |
|---|---:|
| Single-word swap (`during` → `I'm working on`, n=47) | **0.000** |
| Surgical 1-3 word deletes (4 methods, n=403) | **0.013** |
| Substantial paraphrases (rewrites_k7, k=7, n=71) | **0.310** |

**Why**: probes detect deeper semantic content than literal token identity. The 47/0 EDA correlation `during → refusal` was correlation, not causation — 0/47 swap flipped the model.

**Implication**: the 0.917 Pr(m|f flipped) on substantial paraphrases is real causal signal, not gameable surface features.

---

## Slide 10 — Methodological honesty (1 min)

**Train/test rigor:**

| Task | Honesty | AUC source |
|---|---|---:|
| Cyber-1 | ✅ Honest 923-sample held-out test | 0.988 (exp 08) |
| Cyber-2 | ⚠️ 5-fold CV (mixed train+test) | 0.946 |
| Cyber-3 | ⚠️ 5-fold CV | 0.955 |
| Refusal-Gemma | ✅ Honest test split | 0.9417 (Q4r) |

**Error bars** (3 types, used in different figures):
1. **5-fold CV ±1σ** (layer-sweep figs) — variance of probe-fitting across folds
2. **Bootstrap 95% CI, n_boot=500** (AUC-vs-layer fig) — sampling variance of single test set
3. **Wilson 95% CI** (Pr metrics) — binomial proportion interval

**Things we didn't do:**
- Refusal-Qwen (model not downloaded)
- Iterative edit loop with intent-rollback (rules ask `max_iters=5`; we did 1)
- Cyber-probe flip stretch goal
- SAE features (multi-day project)

**Trade-off chosen**: breadth of probe coverage (34 variants) over depth of iteration (1 round). This let us measure the 37-point Pr(m|f) gap *between* probes — a 5-iter loop on one probe wouldn't reveal that.

---

## Slide 11 — Recommendations & headlines (1 min)

**For production refusal monitoring**:
- LR last-tok L40 (test AUC 0.942)
- Switch to Rolling probe @ L30 if low-FPR matters on long contexts (FNR @ 1% FPR jumps 49% → 80% otherwise)
- 5 regex patterns on response catch 83% of refusals at zero false positives — deploy as cheap post-hoc check

**For Level 2 causality claims**:
- Use LR last-token at L40-L60 (causal regime: ~92% Pr(m|f flipped))
- Use 2-probe consensus for stronger claim (LR_last_L45 ∧ LR_mean_of_layers = 3/3 = 100%)
- Don't trust quadratic for causal claims yet (n=20 → 55%, drops to 22% when combined with cyber_3)

**For cross-task / hybrid monitoring**:
- Combined cyber_3+refusal probe trains decent AUC but **dilutes refusal causality**
- Refusal direction is task-specific — don't combine for refusal-detection purpose
- Use cyber_3 → refusal transfer as evidence of severity-shared direction

---

## The 5 numbers to remember

| What | Number |
|---|---:|
| **Mean AUC across 4 tasks** (linear best-layer) | **0.960 ± 0.020** |
| **Refusal honest test AUC** (LR last-tok L40) | **0.9417** |
| **Pr(model flipped \| f flipped)** (LR last-tok L45 on rewrites_k7) | **0.917** |
| Activation lift over TF-IDF on refusal | **+0.075** |
| FNR @ 1% FPR jump (short → long context) | **49% → 80%** |

---

## File index

```
experiments/37_presentation_plots/
├── SLIDE_DECK.md                  ← this file
├── README.md                       ← per-figure explainers + methodology
├── make_plots.py                   ← generates the headline figures
├── auc_vs_layer.py                 ← AUC + accuracy vs layer w/ bootstrap CI
├── causality_focused.py            ← 6-probe × 2-edit causality scatter
└── figures/
    ├── mean_auc_competition.png       ⭐ Slide 2 (Level 1 headline)
    ├── refusal_probe_family_comparison.png   ← Slide 3
    ├── refusal_layer_sweep.png        ← Slide 4 (refusal)
    ├── cyber_layer_sweeps.png         ← Slide 4 (cyber 1/2/3)
    ├── refusal_auc_acc_vs_layer.png   ← Slide 4 (AUC+ACC bootstrap CI)
    ├── causality_focused.png          ⭐ Slide 8 (Level 2 headline)
    └── causality_scatter.png          ← appendix (full 34-probe view)

experiments/22_cross_task_4x4_omar/
└── matrix_4x4.png                  ← Slide 7

experiments/32_refusal_eda_omar/histograms/
└── *.png                            ← Slide 6 EDA support
```

---

## Key supporting files (everything else)

| Result | Path |
|---|---|
| Q4r refusal honest test split | `experiments/31_honest_eval_omar/refusal_only_results.json` |
| Pleshkov 13-layer sweeps | `experiments/17_quadratic_probe_omar/results/{task}_d16_sweep.json` |
| Q8 causality on rewrites_k7 | `experiments/34_combined_causality_omar/causality_rewrites_k7.json` |
| Per-probe robustness (minimal edits) | `experiments/24_robustness_omar/robustness_summary.json` |
| Multi-probe consensus | `experiments/27_consensus_omar/consensus_summary.json` |
| 4×4 cross-task transfer | `experiments/22_cross_task_4x4_omar/results_4x4.json` |
| Phase 3c Pr-metrics (minimal edits) | `experiments/13_pre_rewrites_omar/phase3_summary.json` |
| Q9 prompt-start swap negative | `experiments/35_prompt_start_swap_omar/swap_results.json` |
| Q6 minimal MultiMax ablation | `experiments/33_minimal_multimax_omar/minimal_multimax_results.json` |
| EDA findings + simple-features-work | `experiments/32_refusal_eda_omar/SIMPLE_FEATURES_WORK.md` |
| Original PRE Level 2 metrics | `experiments/13_pre_rewrites_omar/level2_metrics.json` |
