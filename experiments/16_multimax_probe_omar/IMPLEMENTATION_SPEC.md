# Experiment 16: MultiMax & Rolling Attention Probes — Implementation Spec

## Overview

Implement the **MultiMax** and **Max of Rolling Means Attention Probe** ("Selected Probe")
architectures from Kramár et al. 2026 ("Building Production-Ready Probes for Gemini",
`papers/kramar2026_production_ready_probes_gemini.pdf`). These are the two best-performing
probe architectures in the paper — the Selected Probe is GDM's production pick for Gemini.

The key problem these solve: **softmax attention dilutes on long contexts**. When harmful
content is 20 tokens inside a 8000-token prompt, softmax attention spreads weight across
all tokens and the signal drowns. MultiMax and Rolling Attention fix this.

---

## Scope (4 architectures × 2 levels)

This spec defines four architectures that share the same MLP-transform front-end and
differ only in the aggregation stage. All four are implemented as a single `probes.py`
exposing four classes (or one class with a `--variant` flag):

1. **`attention_kramar`** — Standard attention probe with the Kramár 2-layer ReLU MLP
   transform on the front (paper Eq. 7-8). This is the baseline; everything else is a
   drop-in swap of the aggregation stage.
2. **`multimax`** — Hard `argmax` per head instead of softmax-weighted average over
   tokens (paper Eq. 9, §3.2.1). One token per head dominates each example.
3. **`rolling`** — Standard softmax attention *within* sliding windows of width `w=10`,
   then `max` over windows (paper Eq. 10, §3.2.2). The "Selected Probe" architecture.
4. **`rolling_multimax`** — Hard `argmax` *within* each window AND `max` across windows.
   The combined architecture used in some Kramár ablations.

Both **Level 1** and **Level 2** of the hackathon will use these probes:

- **Level 1**: train each architecture on cyber_1, cyber_2, cyber_3, and Gemma refusal
  → fill in the 5-task probe AUC table with the long-prompt-robust variants. Especially
  important on long prompts where exp 15 showed a measurable drop (refusal AUC short
  0.927 → long 0.910 with mean-pool LR; MultiMax should narrow that gap).
- **Level 2**: use the trained `rolling` or `rolling_multimax` probe as the **scoring
  `f`** in the PRE rewrite pipeline (Arditi for attribution, Kramár probe for scoring —
  the "two different `f`s" from `experiments/extra_omar/notes.md`). The current PRE
  pipeline (exp 13) uses the LR mean-pool probe; swapping in a Kramár probe is the
  cleanest upgrade once the architecture passes Level 1 sanity-checks.

Training defaults from Kramár Appendix C: AdamW, lr=1e-4, wd=3e-3, 1000 steps,
`d_hidden=100`, `H=10` heads, `w=10` window for rolling variants. CPU-only training is
fine for Level 1; ~10–30 min per (architecture, task) pair on the existing extracts.

---

## Architecture Reference (from the paper)

All probes in the Kramár framework follow a 6-stage pipeline (Figure 2 in the paper):

```
1) Input residuals    (n, d)          — per-token hidden states from a fixed layer
2) Transformation     (n, d) → (n, d')  — identity, ReLU MLP, or Gated MLP
3) Scores             (n, H)          — per-head score for each token
4) Aggregation Weighting  (n, H)      — how to weight tokens (softmax, hard max, rolling)
5) Aggregation Per-Head   (H,)        — one value per head
6) Probe Output           (1,)        — final scalar logit
```

### Stage 2: MLP Transformation (shared by all architectures)

Before computing attention, apply a 2-layer ReLU MLP to each token independently:

```python
# Paper Eq. 5: MLP_M(X) = A_1 · ReLU(A_2 · X)
# Paper uses M=2 layers, hidden width 100
class TransformMLP(nn.Module):
    def __init__(self, d_in, d_hidden=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (B, T, d_in) or (T, d_in)
        return self.net(x)  # → (B, T, d_hidden) or (T, d_hidden)
```

The output dimension d' = d_hidden = 100 (the paper's default). All downstream
attention/scoring operates on d' = 100, NOT on the full d_model = 3584.

### Architecture A: Attention Probe (Baseline — already exists in starter code)

For reference, the attention probe computes (paper Eq. 7-8):

```
y_{i,j} = MLP_M(x_{i,j})                          # transform
s_{i,j,h} = q_h^T y_{i,j}                          # attention scores
alpha_{j,h} = softmax(s_{:,h})[j]                   # softmax weighting
output_h = sum_j alpha_{j,h} * (v_h^T y_{i,j})      # weighted sum of values
output = sum_h output_h                              # sum over heads
```

Where q_h, v_h ∈ R^{d'} are learned query and value vectors for head h.

**Important**: The value projection `v_h^T y` is separate from the query projection `q_h^T y`.
The attention weights come from q, the pooled representation comes from v. This is
different from the starter code's `AttentionProbe` which uses the same projection for both.

### Architecture B: MultiMax Probe (paper §3.2.1, Eq. 9)

**The key change:** Replace softmax-weighted average with **hard argmax per head**.

```
f_MultiMax(S_i) = sum_{h=1}^{H} max_{j ∈ [n_i]} [v_h^T y_{i,j}]     # Eq. 9
```

In words: for each head h, compute `v_h^T y_j` at every token position j, then take the
**max** over all positions. Sum the per-head maxima to get the final logit.

Pseudocode:
```python
class MultiMaxProbe(nn.Module):
    def __init__(self, d_model, d_hidden=100, n_heads=10):
        super().__init__()
        self.mlp = TransformMLP(d_model, d_hidden)
        self.v = nn.Parameter(torch.randn(n_heads, d_hidden))  # value vectors
        # No query vectors needed — MultiMax doesn't use attention scores
        # No final linear — output is just the sum of per-head maxima

    def forward(self, x_full, mask):
        """
        x_full: (B, T, d_model) — per-token residuals
        mask:   (B, T) — bool, True for real tokens

        Returns: (B,) — one logit per sample
        """
        y = self.mlp(x_full)                      # (B, T, d_hidden)
        # Per-head value scores at each token
        scores = torch.einsum("btd,hd->bth", y, self.v)  # (B, T, H)
        # Mask padding tokens with -inf so they can't be the max
        scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        # Hard max over token dimension, per head
        head_maxes = scores.max(dim=1).values     # (B, H)
        # Sum over heads → one logit per sample
        return head_maxes.sum(dim=-1)              # (B,)
```

**Why this works for long contexts:** Softmax attention dilutes — if the signal is in 20 out
of 8000 tokens, the softmax weight on those 20 tokens is ~0.0025 each. With 10 heads,
you get 10 * 0.0025 = 0.025 total weight on the signal. MultiMax instead picks the single
best token per head — no dilution regardless of context length.

**Training note:** `max` is not differentiable everywhere, but `torch.max` returns the
argmax index and gradients flow through that single position (straight-through-like).
This works fine in practice — GDM trained with AdamW for 1000 steps.

### Architecture C: Max of Rolling Means Attention Probe (paper §3.2.2, Eq. 10)

This is the **Selected Probe** — GDM's production pick. It combines attention pooling
within sliding windows with max-over-windows aggregation.

```
For each window ending at position t, of width w:
    alpha_j = softmax(q^T y_j)   for j in [t-w+1, ..., t]     # attention within window
    v_j = v^T y_j                                               # value at each position
    v_bar_t = sum_{j=t-w+1}^{t} alpha_j * v_j / sum alpha_j    # Eq. 10

Final output = max_t v_bar_t                                     # max over all windows
```

The paper uses H=10 heads and w=10 (window width).

Pseudocode:
```python
class RollingAttentionProbe(nn.Module):
    def __init__(self, d_model, d_hidden=100, n_heads=10, window_size=10):
        super().__init__()
        self.mlp = TransformMLP(d_model, d_hidden)
        self.q = nn.Parameter(torch.randn(n_heads, d_hidden))  # query vectors
        self.v = nn.Parameter(torch.randn(n_heads, d_hidden))  # value vectors
        self.window_size = window_size

    def forward(self, x_full, mask):
        """
        x_full: (B, T, d_model)
        mask:   (B, T) bool
        Returns: (B,) logits
        """
        y = self.mlp(x_full)                          # (B, T, d_hidden)

        # Attention scores (for weighting)
        attn_scores = torch.einsum("btd,hd->bth", y, self.q)  # (B, T, H)
        # Value scores (what gets pooled)
        val_scores = torch.einsum("btd,hd->bth", y, self.v)   # (B, T, H)

        B, T, H = attn_scores.shape
        w = self.window_size

        # For each window ending at position t:
        #   1. softmax over attn_scores within window
        #   2. weighted sum of val_scores within window
        #   3. take max over all windows

        # Efficient implementation using unfold or manual loop
        # Pad the beginning so window at t=0 only sees t=0
        # For each window [t-w+1 .. t], compute attention-weighted value

        all_window_scores = []
        for t in range(T):
            start = max(0, t - w + 1)
            # Window slices
            w_attn = attn_scores[:, start:t+1, :]     # (B, w', H)
            w_vals = val_scores[:, start:t+1, :]       # (B, w', H)
            w_mask = mask[:, start:t+1]                 # (B, w')

            # Mask padding
            w_attn = w_attn.masked_fill(~w_mask.unsqueeze(-1), float("-inf"))

            # Softmax within window
            alpha = F.softmax(w_attn, dim=1)            # (B, w', H)
            alpha = alpha.masked_fill(~w_mask.unsqueeze(-1), 0.0)

            # Weighted sum of values within window
            pooled = (alpha * w_vals).sum(dim=1)        # (B, H)
            all_window_scores.append(pooled)

        # Stack all window outputs: (B, T, H)
        window_outputs = torch.stack(all_window_scores, dim=1)

        # Mask invalid positions
        window_outputs = window_outputs.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        # Max over time, per head, then sum over heads
        head_maxes = window_outputs.max(dim=1).values   # (B, H)
        return head_maxes.sum(dim=-1)                    # (B,)
```

**Performance note:** The naive loop over T is O(T*w) per sample. For our data (T ≤ 8189,
w=10), this is fine. For production, Appendix L describes an O(T) streaming update, but
we don't need that for this hackathon.

**An important optimization:** Instead of the Python loop over T, you can use `torch.unfold`:
```python
# Pad beginning with zeros
padded_attn = F.pad(attn_scores, (0, 0, w-1, 0))  # (B, T+w-1, H)
padded_vals = F.pad(val_scores, (0, 0, w-1, 0))
padded_mask = F.pad(mask, (w-1, 0), value=False)

# Unfold to get all windows at once
# Shape: (B, T, w, H)
win_attn = padded_attn.unfold(1, w, 1).permute(0, 1, 3, 2)
win_vals = padded_vals.unfold(1, w, 1).permute(0, 1, 3, 2)
win_mask = padded_mask.unfold(1, w, 1)  # (B, T, w)

# Softmax within each window
win_attn = win_attn.masked_fill(~win_mask.unsqueeze(-1), float("-inf"))
alpha = F.softmax(win_attn, dim=2)      # (B, T, w, H)
alpha = alpha.masked_fill(~win_mask.unsqueeze(-1), 0.0)

# Weighted sum within each window
pooled = (alpha * win_vals).sum(dim=2)   # (B, T, H)

# Max over time, sum over heads
pooled = pooled.masked_fill(~mask.unsqueeze(-1), float("-inf"))
head_maxes = pooled.max(dim=1).values    # (B, H)
logits = head_maxes.sum(dim=-1)          # (B,)
```

This is fully vectorized and much faster than the loop.

### Architecture D: Rolling MultiMax Probe (combination)

**The combination:** Hard `argmax` *within* each window AND `max` across windows. Drops
softmax entirely from the rolling pipeline. The within-window picks the single most
attentive token in each ten-token slice; the across-window max takes the most attentive
slice. This is a more aggressive sparsification than either MultiMax alone (one token
across the whole sequence) or Rolling alone (smooth within window, sharp across windows).

```
For each window ending at t, of width w:
    w_v_t = v_h^T y_argmax(j in window) (q_h^T y_j)    # hard pick within window
Final = max over t of w_v_t                              # max across windows
```

Pseudocode (vectorized via the same `unfold` trick as Rolling):
```python
class RollingMultiMaxProbe(nn.Module):
    def __init__(self, d_model, d_hidden=100, n_heads=10, window_size=10):
        super().__init__()
        self.mlp = TransformMLP(d_model, d_hidden)
        self.q = nn.Parameter(torch.randn(n_heads, d_hidden))
        self.v = nn.Parameter(torch.randn(n_heads, d_hidden))
        self.window_size = window_size

    def forward(self, x_full, mask):
        y = self.mlp(x_full)                          # (B, T, d_hidden)
        attn_scores = torch.einsum("btd,hd->bth", y, self.q)
        val_scores  = torch.einsum("btd,hd->bth", y, self.v)
        B, T, H = attn_scores.shape
        w = self.window_size

        padded_attn = F.pad(attn_scores, (0, 0, w-1, 0))  # (B, T+w-1, H)
        padded_vals = F.pad(val_scores, (0, 0, w-1, 0))
        padded_mask = F.pad(mask, (w-1, 0), value=False)

        win_attn = padded_attn.unfold(1, w, 1).permute(0, 1, 3, 2)  # (B, T, w, H)
        win_vals = padded_vals.unfold(1, w, 1).permute(0, 1, 3, 2)
        win_mask = padded_mask.unfold(1, w, 1)  # (B, T, w)

        # HARD argmax within each window (instead of softmax)
        win_attn = win_attn.masked_fill(~win_mask.unsqueeze(-1), float("-inf"))
        argmax_pos = win_attn.argmax(dim=2)  # (B, T, H)
        win_pooled = torch.gather(
            win_vals, dim=2,
            index=argmax_pos.unsqueeze(2)
        ).squeeze(2)  # (B, T, H)

        # Max over time (across windows), sum over heads
        win_pooled = win_pooled.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        head_maxes = win_pooled.max(dim=1).values   # (B, H)
        return head_maxes.sum(dim=-1)               # (B,)
```

**When this might help vs. plain Rolling**: if the within-window signal is already
spatially concentrated (one informative token per slice), softmax adds noise from
non-informative neighbours. Hard argmax removes it. Empirical question — should be
ablated alongside the other three.

---

## What to Implement

### File: `experiments/16_multimax_probe_omar/probes.py`

Implement these `nn.Module` classes:

1. **`TransformMLP`** — 2-layer ReLU MLP, d_in → d_hidden (default 100)
2. **`AttentionProbeKramar`** — MLP transform + standard softmax attention (Eq. 7-8). Baseline; the starter code's `AttentionProbe` doesn't have the MLP transform, so we re-implement to keep the four architectures comparable.
3. **`MultiMaxProbe`** — MLP transform + hard `argmax` per head, all-tokens scope (Eq. 9).
4. **`RollingAttentionProbe`** — MLP transform + softmax-within-window (`w=10`) + max-over-windows (Eq. 10). The "Selected Probe".
5. **`RollingMultiMaxProbe`** — MLP transform + hard `argmax` within each window + max-over-windows. Combination architecture (paper ablation §3.2.2).

A clean single-file design is to expose all four under one `KramarProbe(arch="...")`
factory plus the shared `TransformMLP`.

All probes must have this interface:
```python
def forward(self, x_full: Tensor, mask: Tensor) -> Tensor:
    """
    Args:
        x_full: (B, T, d_model) — residual activations, one layer, per token
        mask:   (B, T) — bool, True for real tokens, False for padding
    Returns:
        logits: (B,) — one scalar per sample (NOT passed through sigmoid)
    """
```

### File: `experiments/16_multimax_probe_omar/train.py`

Training script. Must support:

```bash
# Train MultiMax on cyber_1 task
python experiments/16_multimax_probe_omar/train.py \
    --extracts_dir ./extracts/gemma4_31b_cyber \
    --dataset cyber \
    --task cyber_1 \
    --arch multimax \
    --out_dir experiments/16_multimax_probe_omar/results

# Train all architectures on all tasks
python experiments/16_multimax_probe_omar/train.py \
    --extracts_dir ./extracts/gemma4_31b_cyber \
    --dataset cyber \
    --task all \
    --arch all \
    --out_dir experiments/16_multimax_probe_omar/results

# Train Rolling Attention on refusal
python experiments/16_multimax_probe_omar/train.py \
    --extracts_dir ./extracts/gemma4_31b_refusal \
    --dataset refusal_gemma4_31b \
    --task refusal_gemma4_31b \
    --arch rolling \
    --out_dir experiments/16_multimax_probe_omar/results
```

#### Data Loading

Use the existing data infrastructure from `experiments/02_extract_activations/data.py`:

```python
sys.path.insert(0, str(REPO_ROOT / "experiments" / "02_extract_activations"))
from data import load_dataset, get_label_for_task
```

Load activation `.pt` files from `--extracts_dir`. Each `.pt` file has:
```python
{
    "residuals": Tensor(n_layers, n_tokens, d_model),  # fp16
    "attention_mask": Tensor(n_tokens),                  # bool
    "label": ...,
    "sample_id": str,
}
```

For single-layer probes, take `residuals[0]` (or the middle layer if multiple).

The train/test split is determined by `hash(sample_id) % 100 < 70 → train`.

#### Training Hyperparameters (from paper Appendix C)

These are the GDM defaults — use them unless you have a reason not to:

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 3e-3 |
| Momentum | β1=0.9, β2=0.999 |
| Training steps | 1000 |
| Batch size | Full batch (all training samples at once) |
| MLP hidden dim | 100 |
| Number of heads | 10 |
| Window size (rolling) | 10 |
| Activation | ReLU |
| Source layer | Middle (halfway through model) |

**Full batch training**: GDM loads all activations into memory and trains for 1000 steps
with full-batch gradient descent. This is feasible because our datasets are small
(878 refusal, 7259 cyber). Do this if memory allows; otherwise use batch_size=32.

**Multi-seed**: Train each architecture with 5 seeds (0-4). Report mean ± std of AUC.
Save the best-seed model weights.

#### Loss Function

Standard BCE with logits:
```python
loss = F.binary_cross_entropy_with_logits(logits, labels)
```

The paper does NOT use the softmax-weighted loss from CC++ for these architectures.
The softmax-weighted loss (from `constitutional_probe.py`) is for the
StreamingLinearProbe only. MultiMax and Rolling probes use plain BCE.

#### Evaluation

Report for each (arch, task, seed):
- AUC (sklearn.metrics.roc_auc_score)
- Accuracy at threshold 0.5
- F1
- Loss

Use Wilson 95% CI for proportions, bootstrap for AUC (as per CLAUDE.md conventions).

#### Output

Save to `--out_dir`:
- `metrics.jsonl` — one line per (arch, task, seed) with all metrics
- `weights/<task>_<arch>_best.pt` — best-seed model state_dict + metadata
- `summary.json` — mean ± std AUC per (arch, task), comparison table

### File: `experiments/16_multimax_probe_omar/notes.md`

Standard experiment notes with Goal, Method, Results, Takeaways.

---

## Key Implementation Details

### 1. Padding and Masking

Prompts have different lengths (15–8189 tokens). When batching:
- Pad all prompts to max length in batch with zeros
- Create boolean mask: True for real tokens, False for padding
- **Critical**: mask padding with `-inf` before softmax (attention) and before `max` (MultiMax)
  so padding never contributes to the output

The existing `pad_full()` function in `starter_code/train_probe.py` does this correctly.
You can reuse it or write your own.

### 2. Memory Considerations

Full-batch on cyber (7259 samples × ~2000 tokens × 3584 dims × fp16) = ~94 GB.
This won't fit in GPU memory. Options:

**Option A (recommended):** Mean-pool or last-token pool the activations BEFORE training,
reducing to (N, d_model). This is what the starter code does with `x_final`. But then
we lose the per-token information that MultiMax needs.

**Option B:** Mini-batch training with batch_size=1 or batch_size=4. Each sample's
activations are loaded from disk on-the-fly. The probe parameters are tiny (~50K params),
so the bottleneck is loading activations.

**Option C:** Pre-extract just the middle layer and store as (n_tokens, d_model) per sample.
This halves memory vs multi-layer extracts.

For this hackathon, **Option B is the most practical**. Load one sample at a time,
compute loss, accumulate gradients, step every K samples.

### 3. The MLP Transform Is Critical

The paper found that adding the MLP transform before attention is essential for
performance. Without it, the attention probe achieves median AUROC 0.944 (logistic
regression baseline). With the MLP, attention probes reach 0.983 (Table 5).

The MLP projects from d_model (3584 for Gemma) to d_hidden (100). This is a massive
dimensionality reduction that:
- Reduces compute for all downstream attention operations
- Acts as a learned feature selector (which of 3584 dims matter?)
- Prevents overfitting (100-dim attention is much more constrained than 3584-dim)

### 4. Number of Heads

The paper uses H=10 heads for attention probes. Each head has a q vector and v vector
in R^100, so total parameters per probe:

```
TransformMLP:  3584 * 100 + 100 + 100 * 100 + 100 = 368,800
Query vectors: 10 * 100 = 1,000
Value vectors: 10 * 100 = 1,000
Total:         ~371K parameters
```

This is small enough to train in seconds on CPU.

### 5. MultiMax at Eval vs Train Time

The paper notes that MultiMax uses hard max "at inference time (though not always during
training)." During training you CAN use softmax with a low temperature (τ → 0) as a
smooth approximation, but in practice straight-through max works fine with AdamW.

For this implementation: **use hard max (`torch.max`) during both training and inference**.
Gradients flow through the argmax position. This is what GDM does.

### 6. Rolling Attention: MultiMax Aggregation at Eval

The paper says: "we explore using [MultiMax] aggregation at evaluation time" for Rolling
Attention probes. This means:

- **During training:** Use the rolling attention with softmax within windows, max over
  windows (as described above)
- **During evaluation:** You can optionally replace the per-window softmax pooling with
  MultiMax (hard max within each window too). The paper reports results for both modes
  (Table 3: "Rolling Agg" vs "MultiMax Agg")

For this implementation: start with standard rolling (softmax within windows), then
try MultiMax aggregation as an ablation.

---

## Existing Code You Can Reuse

| File | What to reuse |
|------|---------------|
| `starter_code/train_probe.py` | `pad_full()`, `evaluate()`, training loop structure, AUC/F1 computation |
| `experiments/02_extract_activations/data.py` | `load_dataset()`, `get_label_for_task()` — uniform data loading |
| `experiments/02_extract_activations/extractor.py` | Only if you need to extract new activations |
| `experiments/03_constitutional_probe/constitutional_probe.py` | `sliding_window_mean()` for reference (but don't use the SWiM loss — use plain BCE) |

The probe module interface should follow the starter code pattern where possible:
```python
# The forward signature that the training loop expects
def forward(self, x_final, x_full, mask):
    # x_final: (B, d) — last-token pooled (unused by MultiMax, but keep for compat)
    # x_full:  (B, T, d) — full per-token activations
    # mask:    (B, T) — bool mask
    # Returns: logits (B,) or (logits, attention_weights)
```

This way the existing training loop in `starter_code/train_probe.py` can run the new
architectures with minimal changes.

---

## Expected Results

Based on the paper's Table 3 and Table 5:

| Architecture | Expected AUC (cyber) | Expected AUC (refusal) |
|-------------|---------------------|----------------------|
| Linear + mean pool (baseline) | 0.95-0.98 (already have) | 0.93 (already have) |
| Attention (MLP transform) | 0.98-0.99 | 0.94-0.96 |
| MultiMax | 0.98-0.99 | 0.94-0.96 |
| Rolling Attention | 0.98-0.99 | 0.94-0.96 |

The big win should be on **long cyber prompts** (some are 8189 tokens). On short prompts,
all methods are similar. The paper's headline: EMA linear probe FNR on long context jumps
to 87.9% while the Selected Probe stays at 3.0% (Figure 4c).

For refusal probes (≤2048 tokens), the improvement may be modest. But it's still worth
running for the ablation table in the submission.

---

## How to Run (Once Implemented)

```bash
# Smoke test (2 samples, CPU)
python experiments/16_multimax_probe_omar/train.py \
    --extracts_dir ./extracts/gemma4_31b_cyber \
    --dataset cyber --task cyber_1 \
    --arch multimax --limit 2

# Full run on cluster
python experiments/16_multimax_probe_omar/train.py \
    --extracts_dir ./extracts/gemma4_31b_cyber \
    --dataset cyber --task all \
    --arch all \
    --out_dir experiments/16_multimax_probe_omar/results

# Refusal probes
python experiments/16_multimax_probe_omar/train.py \
    --extracts_dir ./extracts/gemma4_31b_refusal \
    --dataset refusal_gemma4_31b --task refusal_gemma4_31b \
    --arch all \
    --out_dir experiments/16_multimax_probe_omar/results
```

---

## Summary Table of Architectures

| Name | Transform | Scoring | Aggregation (within) | Aggregation (across) | Params |
|------|-----------|---------|---------------------|---------------------|--------|
| `attention_kramar` | 2-layer ReLU MLP (100) | q^T y (softmax), v^T y (values) | Softmax attention | Sum over H heads | ~371K |
| `multimax` | 2-layer ReLU MLP (100) | v^T y only | **Hard max** per head | Sum over H heads | ~370K |
| `rolling` | 2-layer ReLU MLP (100) | q^T y (softmax within window w=10), v^T y | Softmax within windows, **max over windows** | Sum over H heads | ~371K |
| `rolling_multimax` | 2-layer ReLU MLP (100) | v^T y only | **Hard max** within windows, **max over windows** | Sum over H heads | ~370K |
