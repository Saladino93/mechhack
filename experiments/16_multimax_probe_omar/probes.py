"""Kramar 2026 Selected Probe — `RollingAttentionProbe` (paper §3.2.2 Eq. 10).

Only the rolling probe is implemented end-to-end here (the Selected Probe / GDM
production pick). The other three architectures from
`IMPLEMENTATION_SPEC.md` are NOT implemented in this scaled-down submission.

Architecture (paper §3.2.2):
  1) `TransformMLP`: 2-layer ReLU MLP, d_model -> d_hidden=100 (paper Eq. 5)
  2) Per-head query and value vectors q_h, v_h ∈ R^{d_hidden}
  3) For each window of width w ending at token t:
        alpha_j = softmax(q_h · y_j) for j in [t-w+1, t]
        v_bar_t = sum_j alpha_j * (v_h · y_j)
  4) head_max_h = max_t v_bar_t (over windows)
  5) logit = sum_h head_max_h

Implementation uses the unfold trick (vectorized over T) — the naive Python
for-loop over T would not finish in time on long-prompt cyber/refusal data.

fp16 sums over long sequences overflow ±65504, so we always cast residuals to
fp32 before reducing (lesson from exp 11 / `experiments/11_refusal_probe_omar/probe.py`).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformMLP(nn.Module):
    """Paper Eq. 5: MLP_M(X) = A_1 · ReLU(A_2 · X), M=2 layers, hidden 100.

    The trailing ReLU keeps the output non-negative (matches the paper). Acts
    as a learnable feature selector that reduces d_model -> 100 before
    attention, sharply cutting both compute and parameters.
    """

    def __init__(self, d_in: int, d_hidden: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_in) or (T, d_in) -> same shape with d_in -> d_hidden
        return self.net(x)


class RollingAttentionProbe(nn.Module):
    """Kramar 2026 Selected Probe (paper §3.2.2, Eq. 10).

    Args:
        d_model: hidden dim of the residual stream (5376 for Gemma-4-31B).
        d_hidden: MLP transform output dim (paper default 100).
        n_heads: number of attention heads (paper default H=10).
        window_size: width of the rolling softmax window (paper default w=10).
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int = 100,
        n_heads: int = 10,
        window_size: int = 10,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.window_size = window_size
        self.mlp = TransformMLP(d_model, d_hidden)
        # Small init so initial softmax over a window is roughly uniform, and
        # values are O(1).
        self.q = nn.Parameter(torch.randn(n_heads, d_hidden) * 0.02)
        self.v = nn.Parameter(torch.randn(n_heads, d_hidden) * 0.02)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x_full: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass over a padded mini-batch.

        Args:
            x_full: (B, T, d_model) per-token residuals (any float dtype, will
                be promoted to fp32 internally).
            mask: (B, T) bool, True for real tokens.

        Returns:
            logits: (B,) one scalar logit per sample (pre-sigmoid).
        """
        # Always operate in fp32 — fp16 overflows on softmax denominators when
        # one window contains a single very-large attention score.
        x_full = x_full.float()
        y = self.mlp(x_full)  # (B, T, d_hidden)

        # Per-head attention scores (q · y) and per-head value scores (v · y).
        attn_scores = torch.einsum("btd,hd->bth", y, self.q)  # (B, T, H)
        val_scores = torch.einsum("btd,hd->bth", y, self.v)   # (B, T, H)

        B, T, H = attn_scores.shape
        w = self.window_size

        # ---- Unfold trick: build all (T) windows of width w in one shot. ----
        # Pad so the window ending at t=0 contains only t=0 (left-pad by w-1).
        # Pad value -inf for attention scores (so padding can never win softmax)
        # and 0 for values (irrelevant, alpha will be 0 there).
        #
        # F.pad on a 3-D tensor with `(left, right)` pads only the LAST dim, so
        # we pass `(0, 0, w-1, 0)` — meaning left-pad the second-to-last dim
        # (the time dim) by w-1.
        neg_inf = torch.tensor(float("-inf"), device=x_full.device, dtype=attn_scores.dtype)
        padded_attn = F.pad(attn_scores, (0, 0, w - 1, 0), value=float("-inf"))  # (B, T+w-1, H)
        padded_vals = F.pad(val_scores, (0, 0, w - 1, 0), value=0.0)              # (B, T+w-1, H)
        padded_mask = F.pad(mask, (w - 1, 0), value=False)                          # (B, T+w-1)

        # Unfold over the time dim: each window is w consecutive positions.
        # Resulting shape: (B, T, H, w) — permute to (B, T, w, H) for clarity.
        win_attn = padded_attn.unfold(1, w, 1).permute(0, 1, 3, 2).contiguous()  # (B, T, w, H)
        win_vals = padded_vals.unfold(1, w, 1).permute(0, 1, 3, 2).contiguous()  # (B, T, w, H)
        win_mask = padded_mask.unfold(1, w, 1).contiguous()                       # (B, T, w)

        # Softmax within each window over the w-axis. Mask-fill -inf where the
        # token is padding (or part of the left-padded prefix).
        win_attn = win_attn.masked_fill(~win_mask.unsqueeze(-1), float("-inf"))
        # All-padding windows would softmax to NaN; clamp by replacing with 0.
        # For real data this only happens in the very first w-1 positions, but
        # we fix it defensively.
        all_invalid = (~win_mask).all(dim=2, keepdim=True)  # (B, T, 1)
        # Replace -inf rows with 0 so softmax produces a uniform-but-zeroed-out
        # row; then we'll mask the output with `mask` anyway.
        win_attn = torch.where(
            all_invalid.unsqueeze(-1).expand_as(win_attn),
            torch.zeros_like(win_attn),
            win_attn,
        )
        alpha = F.softmax(win_attn, dim=2)  # (B, T, w, H)
        # Zero out padded positions (so they contribute nothing to the value sum).
        alpha = alpha.masked_fill(~win_mask.unsqueeze(-1), 0.0)

        # Weighted sum of values within each window -> one vector per (B, t, H).
        pooled = (alpha * win_vals).sum(dim=2)  # (B, T, H)

        # Mask invalid output positions (padding tokens t) so they cannot
        # win the across-window max.
        pooled = pooled.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        # Max over time, per head; sum over heads -> one scalar logit per sample.
        head_maxes = pooled.max(dim=1).values  # (B, H)
        logits = head_maxes.sum(dim=-1) + self.bias  # (B,)
        return logits
