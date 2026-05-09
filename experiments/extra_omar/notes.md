# Omar's Strategy Notes — Method Assessment & Brainstorm

## Overview

Assessment of all methods from the reference papers, ranked by practical usefulness
for this hackathon. Covers Level 1 (predict) and Level 2 (flip), with honest
tradeoffs for each approach.

---

## Tier 1 — Do These First (high impact, low effort)

### 1. Refusal Direction (Arditi 2024)

**The foundation for everything.**

The refusal direction is a single linear direction in the residual stream that
mediates refusal across 13+ chat LLMs. Extracted trivially:

```python
refusal_dir = mean(refused_activations) - mean(complied_activations)
refusal_dir /= refusal_dir.norm()  # unit vector
```

Only needs ~128 refused + ~128 complied samples. No training.

**Why it's #1:**
- Level 1: project activations onto this direction → threshold → AUC. Instant probe.
- Level 2: per-token contribution to the refusal direction → direct attribution.
  `token_score[i] = residual[i] @ refusal_dir` tells you which tokens drive refusal.
- Causally validated: Arditi showed that ablating this direction *actually removes
  refusal*, not just correlation. This means editing tokens that load on this
  direction should causally affect behavior → high concordance.
- The top-8 attention heads that write to this direction attend to specific input
  tokens — those are exactly the tokens to edit for Level 2.

**For Level 2 attribution:**
```
Per-token refusal contribution:
  score_i = residual_at_token_i · refusal_direction

High score → this token is pushing the model toward refusal
Edit these tokens → should causally flip behavior
```

**Limitations:**
- Assumes refusal is linear (single direction). If it's nonlinear, this misses it.
- Found via difference-in-means, which is a first-order approximation.
- May not work as cleanly for cyber probes (4-class graded hierarchy ≠ binary refusal).

**Implementation: Experiment 03**

---

### 2. Attention-Pooling Probes (Boppana 2026 / starter code)

Already implemented in `starter_code/train_probe.py`:
- `AttentionProbe`: single learned query attends over all token positions
- `MultiHeadAttentionProbe`: K=4 learned queries, concat pooled vectors

**Why it matters:**
- Boppana showed attention-pooling probes on residual stream decode the model's
  final answer from the *first tokens of reasoning* (87.98% accuracy on DeepSeek-R1
  671B MMLU), while text-only monitors remain near chance.
- The learned query automatically focuses on informative tokens, ignoring
  padding/filler. Better than mean-pooling which dilutes the signal.
- For Level 2: harder to game than a simple linear probe (the attention weights
  are non-trivially distributed), so concordance should be higher.

**Limitations:**
- Needs more careful training (lower LR, early stopping).
- Slightly slower than linear probes (attention computation), but still milliseconds.

**Implementation: Already in starter code, use in Experiment 04**

---

### 3. SWiM Smoothing + Softmax-Weighted Loss (CC++ / Cunningham 2026)

From Anthropic's production jailbreak-defense stack (Constitutional Classifiers++).

**SWiM (Sliding Window Max):**
```python
# Instead of: logit = probe(mean_pool(residuals))
# Do: logit = max over sliding windows of width 16
for window_start in range(0, n_tokens, stride):
    window = residuals[window_start : window_start + 16]
    window_logit = probe(mean_pool(window))
    logit = max(logit, window_logit)
```
This prevents signal dilution on long sequences. The harmful content might be
in 20 tokens out of 8000 — mean-pooling buries it, SWiM catches it.

**Softmax-weighted loss:**
```python
# Standard: BCE on mean-pooled representation
# CC++: weight the loss by softmax over token-level probe logits
token_logits = probe(residuals)  # (n_tokens,)
weights = softmax(token_logits)  # concentrate on informative positions
loss = weighted_BCE(weights @ residuals, label)
```
This concentrates gradient on the most informative token positions during training.

**Multi-layer concatenation:**
Instead of probing one layer, concatenate residuals from multiple layers.
CC++ found this beats single-layer probes.

**Production stats:**
- Probes cost ~377K FLOPs/token vs ~8B for a small classifier (10,000x cheaper)
- 0.05% false-positive rate
- Probe+classifier ensemble outperforms either alone

**Implementation: Experiment 05**

---

## Tier 2 — Try If Time Allows (good novelty, moderate effort)

### 4. MultiMax / Rolling Attention (Kramar 2026)

From Google's "Production-Ready Probes for Gemini" paper.

**The problem:** Mean-pooling and softmax attention both fail on long contexts.
Signal dilution causes FNR to jump from 3% (short) to 87.9% (long).

**MultiMax:**
```python
# Per attention head, take hard max instead of softmax-weighted mean
for head in range(n_heads):
    head_logits = Q_head @ K.T  # attention scores
    max_position = argmax(head_logits)
    pooled[head] = V[max_position]  # hard max, not soft attention
```
Prevents dilution because it picks the single most relevant position per head.

**Rolling Attention:**
```python
# Attention within sliding windows, then max across windows
for window in sliding_windows(residuals, width=10):
    window_repr = attention_pool(window)
    window_logit = probe(window_repr)
all_logits = [window_logit for each window]
final_logit = max(all_logits)
```

**Results:** Long-context FNR drops from 87.9% (EMA probe) to 3.0% (Selected Probe).

**Most useful for:** Cyber probes (prompts up to 8189 tokens, highly variable length).
Less critical for refusal probes (attribution eval is ≤2048 tokens).

**Implementation: Experiment 05 (alongside SWiM)**

---

### 5. Quadratic (Polynomial) Probes (Pleshkov 2026)

**The idea:**
Standard linear probe: `y = w^T · x`
Quadratic probe: `y = w^T · [x, x_i * x_j for all pairs i,j]`

The quadratic terms capture feature interactions that a linear probe misses.
For example: "code context" × "harmful technique" might encode something different
from either feature alone.

**Method:**
```python
# 1. PCA to reduce dimensionality (REQUIRED — see constraints below)
X_pca = PCA(n_components=d).fit_transform(activations)  # d ≤ 32

# 2. Polynomial lift
# [1, p_1, p_2, ..., p_d, p_1*p_1, p_1*p_2, ..., p_d*p_d]
X_poly = polynomial_features(X_pca, degree=2)

# 3. Ridge regression (closed-form, no SGD)
w = np.linalg.solve(X_poly.T @ X_poly + alpha * I, X_poly.T @ y)

# 4. Predict
y_pred = X_poly_test @ w
```

**Constraints — dataset size vs PCA dimension:**

| PCA dim | Quadratic features (M) | Min samples needed (N >> M) | Works for |
|---------|------------------------|---------------------------|-----------|
| d=16 | 153 | ~500 | Refusal (589 train) |
| d=32 | 561 | ~2000 | Cyber (5079 train) |
| d=64 | 2145 | ~10K | Too big for refusal |

At full d_model=3584, quadratic lift produces ~6.4M features → impossible.
PCA reduction is mandatory.

**When it's useful:**
- If the model encodes harm categories via feature *interactions*, not single directions
- Example: "negation" × "harmful" → "not harmful" is a quadratic interaction
  that a linear probe would miss
- If quadratic AUC >> linear AUC → you've found evidence against the Linear
  Representation Hypothesis for this domain. That's a publishable finding.

**When it's NOT useful:**
- For refusal probes: Arditi showed a single linear direction suffices.
  Quadratic likely won't beat linear here (and might overfit).
- For Level 2 attribution: the gradient of a quadratic probe has cross-terms
  (∂y/∂x_i = w_i + Σ_j w_ij * x_j), making per-token attribution harder to
  interpret. The refusal direction is cleaner.
- Small datasets risk overfitting even with Ridge regularization.

**Best use:** Ablation study on cyber probes.
"Linear AUC = X, quadratic AUC = Y, delta = Z."
If delta is significant → interesting finding about representation geometry.
If delta ≈ 0 → confirms linear representations, also interesting.

**Implementation: Experiment 05 (ablation alongside other advanced probes)**

---

### 6. PRE Loop for Level 2 (Xiong 2025)

The Probe-Rewrite-Evaluate paper gives us the exact Level 2 inner loop:

```python
def edit_with_pre(probe, prompt, rewriter_llm, k=7):
    """Generate k candidate rewrites, score with probe, pick best."""
    candidates = [rewriter_llm(prompt, guidance) for _ in range(k)]
    scores = [probe(candidate) for candidate in candidates]
    best = candidates[argmin(scores)]  # lowest refusal score
    return best
```

**Key findings from the paper:**
- k=7 candidate rewrites scored by probe reduces deception by 25-35%
- Probe score changes are monotonically predictive of behavioral transitions
- The probe transfers cross-architecture (trained on Llama, works on Claude/GPT/Gemini)
- The rewriter system prompt is the key engineering artifact

**For our hackathon:**
- Use the refusal direction for attribution (what tokens to target)
- Use the PRE k-best-of-n selection for editing (how to pick the best edit)
- Use the attention-pooling probe for scoring (harder to game)

**Implementation: Experiment 07**

---

## Tier 3 — Skip for This Hackathon

| Method | Paper | Why Skip |
|--------|-------|----------|
| SAE features | Cunningham 2023 | Need pre-trained SAEs for Gemma/Qwen — don't exist. Training takes days. |
| ACDC circuits | Conmy 2023 | Demonstrated on GPT-2 Small (85M params). Our models are 300x bigger. Won't finish. |
| Sparse feature circuits | Marks 2024 | Depends on SAEs (see above). |
| Tuned lens | Belrose 2023 | Need to train affine maps per layer. Interesting but lower priority than probes. |
| VPD weight decomposition | Bushnaq 2026 | Only demonstrated at 67M scale. Not applicable to 31B models yet. |

---

## Recommended Pipeline

### Level 1: Predict

```
Experiment 03: Refusal direction (Arditi)
    ↓ gives baseline AUC + attribution source for Level 2
Experiment 04: Linear + attention probes for all 5 tasks
    ↓ gives proper AUC with error bars
Experiment 05: Advanced probes
    ├── SWiM smoothing (CC++ recipe)
    ├── MultiMax / Rolling (Kramar, for long cyber prompts)
    └── Quadratic ablation (Pleshkov, on cyber probes)
```

### Level 2: Flip

```
Attribution: Per-token refusal direction projection (Arditi)
    ↓ identifies which tokens to edit
Edit generation: k-best-of-n with LLM rewriter (PRE / Xiong)
    ↓ generates 5-7 candidate rewrites targeting attributed tokens
Edit scoring: Attention-pooling probe (Boppana)
    ↓ ranks candidates, picks best
Verification: Re-roll Gemma on edited prompt
    ↓ computes honest Pr(model flipped | edit)
Report: Pr(f|edit), Pr(model|edit), Pr(model|f)
```

---

## Key Insight: Two Different `f`s for Level 2

Use different methods for different jobs:

| Job | Best `f` | Why |
|-----|---------|-----|
| **Attribution** (what to edit) | Refusal direction | Causal, not just correlational |
| **Scoring** (which edit is best) | Attention-pooling probe | Fast, harder to game than linear |

A single linear probe for everything → high Pr(f|edit) but low Pr(model|f).
Split approach → higher concordance → "genuinely causal features" finding.

---

## What Would Win

1. **High AUC on all 5 tasks** with error bars — probes that work, not just one task
2. **High concordance** Pr(model|f) — proves your attribution found causal features
3. **Interesting findings** — "quadratic beats linear on cyber but not refusal"
   says something about how models encode harm categories
4. **Worked examples** — show original prompt, attributed tokens, edit, result
5. **Honesty** — if concordance is low, say so. "We gamed the probe" is a finding.
