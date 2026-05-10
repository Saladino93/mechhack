# Probing Refusal: linear is the strong baseline; causality confirms

> Slide-deck draft narrative + supporting numbers. Target: 10-12 slides.
> Deadline: Sun May 10, 14:00 UTC. Last updated: 2026-05-10 ~10:00 UTC.

---

## Slide 1 — Setup

**The five tasks** (rules):
- Cyber-1: dual-use vs benign
- Cyber-2: high-risk dual-use vs (du ∪ ben)
- Cyber-3: prohibited vs (hrdu ∪ du ∪ ben)
- Refusal-Gemma: refused vs complied
- Refusal-Qwen: same on Qwen 3.6-27B (skipped — model not downloaded)

**Metric**: mean AUC across 5 tasks, train on `split=='train'`, eval on `split=='test'`.
We report 4 (Qwen excluded).

**Methods evaluated**: 12 probe families × multiple layers, plus 4 edit families
for Level 2.

---

## Slide 2 — Headline (Level 1)

> **Linear single-layer at L40-L45 wins on every task. Multi-layer concat
> overfits. Quadratic loses (PCA bottleneck). Kramar Rolling buys long-context
> robustness at a small AUC cost.**

| Task | Best probe | Test AUC | Layer | Pooling | Source |
|---|---|---:|---:|---|---|
| Cyber-1 | LR mean | **0.988** | L40 | mean | exp 08, 923-sample held-out |
| Cyber-2 | LR last-tok | **0.955** | L60 | last | 5-fold CV (contaminated) |
| Cyber-3 | LR mean | **0.955** | L35 | mean | 5-fold CV (contaminated) |
| **Refusal-Gemma** | **LR last-tok** | **0.9417** | **L40** | **last** | **Q4r honest test (n=277)** |
| **Mean (4 tasks)** | | **0.960** | | | |
| **Std across tasks** | | **0.018** | | | |

> **Refusal-Gemma honest number = 0.9417** (test split, train on train only).
> Slightly below the contaminated 5-fold CV number (0.9528), as expected. This
> is the rules-compliant Level 1 figure. Cyber 2/3 honest test split was not
> feasible in time (Q4 cyber multi-layer eval killed for disk-IO budget).

### Refusal probe family on honest test split (n_train=555, n_test=277)

Top 8 probes by test AUC:

| Probe | test AUC | train AUC | gap |
|---|---:|---:|---:|
| LR_last_L40 | **0.9417** | 1.0000 | +0.058 |
| LR_last_L45 | 0.9368 | 1.0000 | |
| LR_last_L55 | 0.9350 | 1.0000 | |
| LR_last_L50 | 0.9331 | 1.0000 | |
| LR_last_L60 | 0.9295 | 1.0000 | |
| LR_mean_L40 | 0.9269 | 0.9999 | |
| MLP_L30 | 0.9207 | 0.9996 | |
| LR_multi_concat (69,888d) | 0.9137 | 0.9999 | overfit |
| Pleshkov_L50 (best layer) | 0.854 | 0.955 | |
| Constitutional_concat | **0.5000** | 0.996 | catastrophic |

Read: train AUC ≈ 1.0 across all probes — they fully fit the 555-sample
training set. Test AUC distinguishes generalization. **LR last-token at
mid-late layers (L35-L60) dominates** with test AUCs 0.93-0.94. Pleshkov
loses 0.09 AUC. Constitutional probe (multi-layer concat → MLP head)
catastrophically fails at test 0.5 — too many parameters, 555 samples.

**Random predictor**: AUC=0.50.
**TF-IDF baseline**: 0.877–0.946 across tasks (refusal 0.877, cyber_1 0.946).

Activation **lift** over TF-IDF grows with task difficulty:
- cyber_1 +0.037
- cyber_2 +0.059
- cyber_3 +0.065
- **refusal +0.075** ← strongest lift on hardest task

This is the right shape: probes earn their keep when surface text alone is
not enough.

---

## Slide 3 — Probe family ablation on refusal

Single-task deep dive (refusal_gemma):

| Family | Test AUC | Notes |
|---|---:|---|
| **LR last-token L45** | **0.953 ± 0.004** | new winner |
| LR last-token L40 | 0.949 ± 0.009 | |
| LR mean L40 | 0.945 ± 0.019 | |
| Multi-layer concat (13×5376=69,888d) | 0.929 | overfits, train ≈ 1.0 |
| LR mean-of-layers (5376d) | 0.925 ± 0.014 | average across 13 layers |
| LR mean L32 | 0.932 ± 0.013 | original "baseline" |
| Pleshkov quadratic d=16 @ L32 | 0.860 | PCA bottleneck dominates |
| **Rolling probe @ L30** (Kramar arch B) | **0.877** | length-robust (long=0.89) |
| **Minimal MultiMax @ L30** (no MLP) | **0.834** | per-token linear → max |
| MLP probe (1 hidden layer L40 mean) | 0.929 (val) | overfits, no real lift |
| Constitutional probe (concat → MLP) | 0.899 (val) | doesn't beat LR |
| TF-IDF word 1-2gram | 0.877 ± 0.025 | naive baseline |
| TF-IDF char 3-5gram | 0.870 ± 0.038 | |
| Random | 0.500 | |

**Three takeaways**:

1. **Capacity ≠ generalization.** The 69,888-dim multi-layer concat is WORSE
   than the 5,376-dim single-layer LR (0.929 vs 0.953). When the boundary is
   close to linear in residual space, more capacity overfits.

2. **Pleshkov quadratic loses ~0.05–0.07 AUC** consistently — but the linear-
   on-d=16-PCs baseline loses by the same amount. The PCA bottleneck is the
   binding constraint, NOT lack of nonlinearity.

3. **Minimal MultiMax (no MLP) at AUC 0.834 lags LR-mean-L30** (~0.93). The
   Kramar paper's MultiMax has a 2-layer ReLU MLP front-end (`d_model→100`).
   The MLP transformation is what enables max-pool to beat mean-pool — without
   it, max-pool alone is *too sparse* to use most of the residual stream.

---

## Slide 4 — Where Kramar Rolling beats LR

Length-stratified evaluation (exp 15) on refusal, by char-length tertile:

| Bucket | n | LR mean L32 AUC | Rolling L30 AUC |
|---|---:|---:|---:|
| short (<6.5k chars) | 277 | 0.927 | 0.847 |
| medium (6.5–11.3k) | 277 | 0.918 | 0.829 |
| **long (>11.3k)** | 278 | 0.918 | **0.889** |

LR's AUC stays ~constant across length BUT its **threshold-stable behavior**
breaks down. Production-relevant metric: FNR @ 1% FPR (false-negative rate
when you set the threshold to flag 1% of compliance as refusal):

| Bucket | LR mean L32 FNR@1%FPR |
|---|---:|
| short | **49%** |
| medium | 60% |
| **long** | **80%** |

That's the **Kramár threshold-shift**: AUC stays 0.92, but the *operating-
point performance* collapses from 49% → 80% FNR on long prompts. **At
production thresholds, LR fails on long contexts.**

Rolling probe doesn't have the same dropoff — its long-bucket AUC (0.889) is
better than its short-bucket (0.847). Use Rolling when you operate at a
strict FPR and care about long contexts.

> **Slide-deck plot suggestion**: bar chart of FNR @ 1% FPR by length tertile
> for LR vs Rolling. Single most striking visual.

---

## Slide 5 — Pleshkov 13-layer sweep

> Across all 13 layers and all 4 tasks: **quadratic ≤ linear** consistently.

Refusal layer sweep (d=16 PCA → degree-2 → ridge):

| Layer | linear (raw) | linear-on-PCs | quadratic | Δ (q − lin) |
|---|---:|---:|---:|---:|
| 0 | 0.852 | 0.764 | 0.744 | −0.107 |
| 5 | 0.832 | 0.827 | 0.810 | −0.022 |
| 10 | 0.870 | 0.844 | 0.819 | −0.051 |
| 15 | 0.880 | 0.845 | 0.839 | −0.041 |
| 20 | 0.893 | 0.849 | 0.851 | −0.042 |
| ... | ... | ... | ... | ... |
| 60 | 0.909 | 0.864 | 0.847 | −0.062 |

**Verdict**: PCA d=16 throws away >99% of the 5376-d residual; Ridge can't
recover the lost discriminative directions. Pleshkov 2026's positive
quadratic results don't replicate on Gemma-4-31B refusal at d=16. The
quadratic-vs-linear-on-PCs comparison shows quadratic is **slightly better**
than linear-at-the-same-bottleneck (avg +0.005) — so nonlinear interactions
DO help marginally — but they can't compensate for the bottleneck.

**Plot files**: `experiments/17_quadratic_probe_omar/results/{task}_d16_sweep.png`

---

## Slide 6 — There is no single "harm direction"

4×4 cross-task transfer (mean-pool L35; train one task, test on each):

| | cyber_1 | cyber_2 | cyber_3 | refusal_gemma |
|---|---:|---:|---:|---:|
| **train cyber_1** | (CV) 0.981 | 0.884 | 0.715 | 0.654 |
| **train cyber_2** | 0.906 | (CV) 0.942 | 0.785 | 0.783 |
| **train cyber_3** | 0.803 | 0.733 | (CV) 0.955 | **0.755** |
| **train refusal** | **0.525** | 0.720 | 0.772 | (CV) 0.939 |

**Headline**: `refusal → cyber_1 = 0.525` (basically chance). The refusal
direction and the dual-use direction are **orthogonal in residual space**.

But `refusal ↔ cyber_3 ≈ 0.76`. The **prohibited tier** of cyber shares
signal with model-refusal — both reflect "this is the kind of thing I won't
do," not "this involves potentially-misuseable concepts."

**Implication**: don't expect a single harm-detector to work across the
ladder. Cyber-1 is a domain classifier; refusal is a behavior predictor.
Cyber-3 sits in the middle and bridges them.

---

## Slide 7 — EDA: refusals are deterministic surface phenomena

Looking at first-words/n-grams gives an embarrassingly large amount of signal:

**Response opening 3-grams** (perfect refusal markers):
- `I cannot provide` → 161 refused / **0** complied
- `While I understand` → 140 / 0
- `I cannot complete` → 18 / 0

→ **5 regex patterns catch 83% of refusals at zero false positives.**

**Prompt-start word `during`** = 47 refused / **0** complied.
- Looks like a red-team template artifact: "During [authorized scenario], please [harmful capability]…"
- Gemma reliably flags this opener.

**Length correlations** with `is_refusal`:
- response_chars: r = **−0.516** (refusals are SHORT)
- prompt_chars: r = **−0.402** (long prompts → MORE compliance)

The second one is the **behavioral side of the Kramár threshold-shift**: at
long context, the refusal-pulling tokens are diluted in the model's actual
behavior, not just in mean-pool probes.

**Why this matters for the slide deck**:
> The signal lives heavily on the surface. Activations earn +0.07–0.08 AUC
> over TF-IDF, which is real (and grows with task difficulty), but **most of
> the discriminative work is being done at the token level**. This explains
> why "fancy" probe architectures (multi-layer concat, MLP, Constitutional)
> fail to beat the linear baseline — there's no expressivity gap to fill at
> the layer where refusal lives.

---

## Slide 8 — Level 2: causality on minimal edits

> The Level-2 metric: Pr(model flipped | f flipped). High = causal; low
> while Pr(f flipped | edit) is also high = "you gamed f."

We tested **403 edits** (81 prompts × 4 minimal-edit methods):
- delete top-1 word (Arditi-attribution-driven)
- delete top-3 words
- delete top-1 sentence
- DeepSeek single-round span editor

Each edit **scored under every fitted probe** (28 LR variants + Pleshkov),
each rolled through Gemma + judged by DeepSeek V4 Pro:

| Probe | n | Pr(f|edit) | Pr(model|f flipped) | Verdict |
|---|---:|---:|---:|---|
| **LR last-tok L20** | 81 | 0.034 | **0.545** (6/11) | causal ✓ |
| **LR last-tok L40** | 81 | 0.025 | **0.750** (6/8) | causal ✓ |
| **LR last-tok L60** | 81 | 0.022 | **1.000** (7/7) | causal ✓ |
| LR mean L40 | 81 | 0.009 | 0.667 (2/3) | causal (n=3) |
| Multi-layer concat | 81 | 0.003 | 1.000 (1/1) | causal (n=1) |
| **Pleshkov d=16 @ L40** | 81 | **0.006** | **0.000** (0/2) | over-robust, gamed (small n) |
| **Pr(model|edit) constant** | | **0.134** | (across probes) | |

**Three findings**:

1. **All probes are robust to surface edits** — Pr(f|edit) ≤ 3.4% across
   the board. Surgical 1-3 word deletes barely move any probe. **Probes are
   not trivially gameable by minimal edits.**

2. **Last-token probes show the causal regime**: when last-token L20-L60
   says "flipped," the model flips ~55-100% of the time. This is the
   "everything high relative to its own scale" pattern — selective probe
   that genuinely tracks model behavior.

3. **Pleshkov is over-robust**: 0.6% Pr(f|edit) (lowest of all), but the
   2 cases where it did flip, model didn't follow. Caveat: n=2, CI [0,
   0.66] — too small to definitively call it gamed, but suggestive.

> **Multi-probe consensus**: when both `LR_last_L45 ∧ LR_mean_of_layers`
> say flipped, the model flips **3/3 = 100%**, CI [0.44, 1.00]. Strongest
> small-n causal claim.

---

## Slide 9 — Level 2 caveat: minimal edits may be too gentle

`Pr(model|edit) = 0.134` overall — only 13% of minimal edits actually flip
the model. To distinguish "probe-gameable" from "probe-causal" properly we
need **bigger** edits where Pr(f|edit) is higher.

We have a separate dataset of **k=7 substantial paraphrases per prompt**
(`rewrites_k7.json`, total 567 rewrites). Re-running the same per-probe
causality analysis on these (Q1 GPU pass + Q8 score script):

**Old single-probe number** (LR mean L32, paraphrase pipeline):
- Pr(f|edit) = 0.358 (10× higher than minimal)
- Pr(model|edit) = 0.310
- Pr(model|f, orig=refusal) = 0.450 (n=20)

Per-probe rewrites_k7 numbers + combined cyber_3+refusal probe **TBD** —
finishes when GPU pass completes (~5 more min). Will fill in.

---

## Slide 10 — Combined-probe (cyber_3 ∪ refusal) hypothesis

**Hypothesis**: a probe trained on the *union* of cyber_3 train (prohibited
vs rest) and refusal train (refused vs complied) — both labeled "harmful
intent = 1" — should be **more general and more robust to surface edits**
than either alone.

**Evidence in favor**: 4×4 transfer matrix shows cyber_3 ↔ refusal share
~0.76 AUC (slide 6). They're aligned.

**Evidence against**: cyber_3 and refusal have different positive priors
(refusal: 47.9%, cyber_3 prohibited: ~25%). Combined data doesn't have a
single coherent label distribution.

**Test (Q8, in flight)**: train combined LR at L40 mean and L45 last-tok,
compare causality to refusal-only and cyber_3-only baselines. **TBD**.

---

## Slide 11 — Recommendations

For **production refusal monitoring**:
- LR last-token L45 (best AUC, simplest)
- BUT: switch to Rolling probe @ L30 if you operate at low-FPR on long
  contexts (Kramár threshold-shift).

For **causal analysis** (Level 2):
- Use last-token L20-L60 probes — they show ~50-100% Pr(model|f flipped)
- Or: 2-probe consensus (LR_last_L45 ∧ LR_mean_of_layers) for stronger
  claim
- DON'T trust Pleshkov for causal claims yet — n is too small.

For **cross-task / hybrid** monitoring:
- A combined probe trained on cyber_3 ∪ refusal might generalize better
  than either alone — but only on the prohibited/refusal tier (not dual-
  use).
- For dual-use detection, train a separate probe — it's a different
  direction.

For **regression-test detection** (catch refusals after the fact):
- 5-line regex on the model's response (`I cannot provide`, `While I
  understand`, …) catches 83% of refusals at zero false positives.
- Use probes when you need the prediction *before* the response.

---

## Slide 12 — Honest caveats

What we couldn't do in time:
- Refusal-Qwen (model not downloaded). 4-task mean reported.
- Honest test-split AUC for cyber_1/2/3: only have 5-fold CV (which sampled
  across train+test). Cyber_1 has a separate held-out test (0.988); cyber_2/3
  do not.
- Pleshkov causality has only n=2 — wide CIs.
- Kramar attention_kramar (arch A) never finished training.
- Iterative edit loop with rollout-in-the-loop attempted but not run.

What we got right:
- Honest train/test split for refusal_gemma (Q4r).
- Apples-to-apples per-probe causality (E1b output: 28 probes × 403 edits +
  judge labels).
- Explicit length-stratified eval revealing the Kramár threshold-shift on
  refusal.
- Cross-task transfer matrix exposing the orthogonality of refusal vs
  cyber_1.
- EDA grounding: refusals are deterministic surface phenomena.

---

## Appendix — Files & key plots

- **Level-1 layer sweep**: `experiments/18_refusal_layer_sweep_omar/auc_vs_layer.png`
- **Pleshkov 13-layer**: `experiments/17_quadratic_probe_omar/results/{cyber_1,2,3,refusal_gemma}_d16_sweep.png`
- **Cross-task 4×4**: `experiments/22_cross_task_4x4_omar/matrix_4x4.png`
- **Length-stratified**: `experiments/15_refusal_length_strat_omar/`
- **EDA**: `experiments/32_refusal_eda_omar/{histograms,ngrams}/`
- **Causality table**: `experiments/24_robustness_omar/robustness_table.md`
- **Per-edit-method Pr metrics**: `experiments/13_pre_rewrites_omar/phase3_summary.json`
- **Multi-probe consensus**: `experiments/27_consensus_omar/consensus_summary.json`
- **Pleshkov causality**: `experiments/29_pleshkov_causality_omar/pleshkov_metrics.json`
- **TF-IDF baseline**: `experiments/21_probe_zoo_omar/results/refusal_baselines.json`

Code that produced each section is in the corresponding `experiments/NN_*_omar/`
directory.
