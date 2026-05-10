# MultiMax & Rolling Attention Probes — Fast Implementation Guide

**Time budget: 2 hours. Ship working code, not perfect code.**

## What to Build

Two new probe architectures from Kramár et al. 2026 (GDM). Add them to a single file
`experiments/15_multimax_probe_omar/probes.py` and a training script `train.py`.

## Existing Code to Build On

The starter code already has working probes and training. Key files:

- `starter_code/train_probe.py` — has `LinearProbe`, `MLPProbe`, `AttentionProbe`, `MultiHeadAttentionProbe`, data loading (`build_dataset`, `pad_full`), training loop, evaluation (AUC/F1/acc). **Reuse as much as possible.**
- `experiments/02_extract_activations/data.py` — `load_dataset(name, split)`, `get_label_for_task(sample, task)` for uniform data loading across cyber/refusal datasets.
- `experiments/03_constitutional_probe/constitutional_probe.py` — Has `StreamingLinearProbe` with SWiM. Reference only.

Each `.pt` extract file contains:
```python
{
    "residuals": Tensor(n_layers, n_tokens, d_model),  # fp16, d_model=3584 for Gemma
    "attention_mask": Tensor(n_tokens),                  # bool
    "label": ...,
    "sample_id": str,
}
```

Train/test split: `hash(sample_id) % 100 < 70 → train`.

## Architecture 1: MultiMax (paper Eq. 9)

**Concept:** For each of H=10 heads, compute a scalar score at every token, then take the **hard max** (not softmax average). Sum the per-head maxima. This prevents signal dilution on long contexts.

```python
class TransformMLP(nn.Module):
    """2-layer ReLU MLP applied per-token. Projects d_model → d_hidden."""
    def __init__(self, d_in, d_hidden=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)  # (B, T, d_in) → (B, T, d_hidden)


class MultiMaxProbe(nn.Module):
    """Eq. 9: f(S_i) = sum_h max_j [v_h^T y_{i,j}]"""
    def __init__(self, d_model, d_hidden=100, n_heads=10):
        super().__init__()
        self.mlp = TransformMLP(d_model, d_hidden)
        # v_h are learned value vectors, one per head
        self.v = nn.Parameter(torch.randn(n_heads, d_hidden) / d_hidden**0.5)

    def forward(self, x_final, x_full, mask):
        """
        x_final: (B, d) — unused, kept for interface compat with starter code
        x_full:  (B, T, d_model) — per-token residuals
        mask:    (B, T) — bool, True=real token
        Returns: logits (B,)
        """
        y = self.mlp(x_full)                                    # (B, T, d_hidden)
        scores = torch.einsum("btd,hd->bth", y, self.v)        # (B, T, H)
        scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        head_maxes = scores.max(dim=1).values                   # (B, H)
        return head_maxes.sum(dim=-1)                            # (B,)
```

**Key points:**
- No query vectors needed — MultiMax doesn't use attention weights
- `torch.max` gradient flows through the argmax position (straight-through)
- Mask padding with `-inf` before max so padding never wins

## Architecture 2: Rolling Attention (paper Eq. 10)

**Concept:** Softmax attention within sliding windows of width w=10, then take the max score across all windows. Combines local attention with global max pooling.

```python
class RollingAttentionProbe(nn.Module):
    """Eq. 10: attention within windows, max over windows."""
    def __init__(self, d_model, d_hidden=100, n_heads=10, window_size=10):
        super().__init__()
        self.mlp = TransformMLP(d_model, d_hidden)
        self.q = nn.Parameter(torch.randn(n_heads, d_hidden) / d_hidden**0.5)
        self.v = nn.Parameter(torch.randn(n_heads, d_hidden) / d_hidden**0.5)
        self.w = window_size

    def forward(self, x_final, x_full, mask):
        y = self.mlp(x_full)                                    # (B, T, d_hidden)
        B, T, D = y.shape
        H = self.q.shape[0]
        w = self.w

        attn_scores = torch.einsum("btd,hd->bth", y, self.q)   # (B, T, H)
        val_scores  = torch.einsum("btd,hd->bth", y, self.v)   # (B, T, H)

        # Vectorized sliding windows using unfold
        # Pad beginning so early tokens have partial windows
        pad_attn = F.pad(attn_scores, (0, 0, w-1, 0))          # (B, T+w-1, H)
        pad_vals = F.pad(val_scores,  (0, 0, w-1, 0))
        pad_mask = F.pad(mask,        (w-1, 0), value=False)    # (B, T+w-1)

        # unfold: extract all windows of size w along dim=1
        # Result: (B, T, H, w) after permute
        win_attn = pad_attn.unfold(1, w, 1).permute(0, 1, 3, 2)  # (B, T, w, H)
        win_vals = pad_vals.unfold(1, w, 1).permute(0, 1, 3, 2)
        win_mask = pad_mask.unfold(1, w, 1)                       # (B, T, w)

        # Softmax attention within each window
        win_attn = win_attn.masked_fill(~win_mask.unsqueeze(-1), float("-inf"))
        alpha = F.softmax(win_attn, dim=2)                        # (B, T, w, H)
        alpha = alpha.masked_fill(~win_mask.unsqueeze(-1), 0.0)

        # Attention-weighted values within each window
        pooled = (alpha * win_vals).sum(dim=2)                    # (B, T, H)

        # Max over time (across windows), per head
        pooled = pooled.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        head_maxes = pooled.max(dim=1).values                     # (B, H)
        return head_maxes.sum(dim=-1)                              # (B,)
```

## Architecture 3: Attention Probe with MLP (baseline comparison)

Same as starter code's `AttentionProbe` but with the MLP transform added. This lets us
compare: does MultiMax's hard-max beat standard softmax attention when both have the MLP?

```python
class AttentionProbeKramar(nn.Module):
    """Eq. 8: standard softmax attention with MLP transform."""
    def __init__(self, d_model, d_hidden=100, n_heads=10):
        super().__init__()
        self.mlp = TransformMLP(d_model, d_hidden)
        self.q = nn.Parameter(torch.randn(n_heads, d_hidden) / d_hidden**0.5)
        self.v = nn.Parameter(torch.randn(n_heads, d_hidden) / d_hidden**0.5)

    def forward(self, x_final, x_full, mask):
        y = self.mlp(x_full)
        attn_scores = torch.einsum("btd,hd->bth", y, self.q)
        val_scores  = torch.einsum("btd,hd->bth", y, self.v)
        attn_scores = attn_scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        alpha = F.softmax(attn_scores, dim=1)                    # (B, T, H)
        pooled = (alpha * val_scores).sum(dim=1)                  # (B, H)
        return pooled.sum(dim=-1)                                  # (B,)
```

## Training Script

### Hyperparameters (from paper Appendix C — use exactly these)

```python
HPARAMS = {
    "optimizer": "AdamW",
    "lr": 1e-4,
    "weight_decay": 3e-3,
    "betas": (0.9, 0.999),
    "epochs": 1000,        # steps, full-batch = 1000 epochs
    "d_hidden": 100,       # MLP hidden dim
    "n_heads": 10,
    "window_size": 10,     # for Rolling only
    "seeds": [0, 1, 2, 3, 4],
}
```

### Loss: plain BCE

```python
loss = F.binary_cross_entropy_with_logits(logits, labels)
```

Do NOT use the softmax-weighted loss from CC++. These architectures use standard BCE.

### Data loading pattern

```python
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments" / "02_extract_activations"))
sys.path.insert(0, str(REPO_ROOT / "starter_code"))
from data import load_dataset, get_label_for_task

# Load samples
samples = load_dataset("cyber", split="train")  # or "refusal_gemma4_31b"

# Load activations from .pt files
for sample in samples:
    pt = torch.load(f"{extracts_dir}/{sample['sample_id']}.pt", weights_only=False)
    residuals = pt["residuals"][0]    # first layer → (T, d_model)
    mask = pt["attention_mask"]       # (T,) bool
    label = get_label_for_task(sample, "cyber_1")  # 0 or 1
```

### Batching

Prompts have variable length (15–8189 tokens). Use the existing `pad_full()` from
`starter_code/train_probe.py`:

```python
from train_probe import pad_full  # pads to max length in batch, returns (B,T,d), (B,T) mask
```

For memory, use batch_size=4 or batch_size=8 (not full-batch — 7259 cyber samples
× 2000 tokens × 3584 dims won't fit in GPU RAM at once).

### CLI interface

```bash
python experiments/15_multimax_probe_omar/train.py \
    --extracts_dir ./extracts/gemma4_31b_cyber \
    --dataset cyber \
    --task cyber_1 \
    --arch multimax \
    --seeds 5 \
    --out_dir experiments/15_multimax_probe_omar/results

# arch choices: multimax, rolling, attention_kramar, all
# task choices: cyber_1, cyber_2, cyber_3, refusal_gemma4_31b, refusal_qwen36, all
```

### Output

- `results/metrics.jsonl` — one line per (arch, task, seed): `{"arch", "task", "seed", "auc", "acc", "f1", "loss"}`
- `results/weights/<task>_<arch>_best.pt` — best-seed state_dict
- `results/summary.json` — mean±std AUC table

### Evaluation

Use sklearn:
```python
from sklearn.metrics import roc_auc_score, f1_score
auc = roc_auc_score(y_true, probs)
```

Report Wilson 95% CI for proportions. Bootstrap for AUC if time allows, otherwise
just report mean±std across 5 seeds.

## Priority Order (2 hours)

1. **[30 min] `probes.py`** — TransformMLP + MultiMaxProbe + RollingAttentionProbe + AttentionProbeKramar. All 3 architectures, tested with a random tensor smoke test.

2. **[45 min] `train.py`** — Training loop that loads .pt files, trains probe, evaluates, saves metrics. Start with MultiMax on cyber_1 only. Verify AUC > 0.95.

3. **[30 min] Run all tasks** — cyber_1/2/3 + refusal_gemma4_31b with all 3 architectures. 5 seeds each. Save metrics.

4. **[15 min] Summary** — comparison table in `notes.md`. Did MultiMax beat linear? Did Rolling beat MultiMax? How much did the MLP transform help?

## Quick Smoke Test

Before training, verify the probe shapes work:

```python
B, T, d = 2, 50, 3584
x_full = torch.randn(B, T, d)
mask = torch.ones(B, T, dtype=torch.bool)
mask[0, 40:] = False  # simulate padding

for Probe in [MultiMaxProbe, RollingAttentionProbe, AttentionProbeKramar]:
    p = Probe(d_model=d)
    out = p(None, x_full, mask)
    assert out.shape == (B,), f"{Probe.__name__} output shape wrong: {out.shape}"
    out.sum().backward()  # verify gradients flow
    print(f"{Probe.__name__}: OK, output={out.detach()}")
```
