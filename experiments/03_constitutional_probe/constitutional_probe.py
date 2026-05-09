"""Constitutional Classifier++ style streaming linear probe.

This module implements the probe objective from "Constitutional Classifiers++":

* per-token linear logits over residual-stream activations
* Sliding Window Mean (SWiM) logit smoothing during training
* softmax-weighted per-token BCE loss
* max-over-time / EMA sequence scoring for streaming use

The implementation is deliberately lightweight: a multi-layer probe is represented as
one weight vector per selected layer, so it avoids explicitly concatenating all layer
activations into a giant `(tokens, layers * d_model)` matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProbeConfig:
    n_layers: int
    d_model: int
    window_size: int = 16
    tau: float = 1.0
    score_mode: str = "max_swim"  # one of: max_swim, max_ema, final_swim
    ema_decay: float | None = None
    layer_indices: tuple[int, ...] | None = None


class StreamingLinearProbe(nn.Module):
    """Linear probe over one or more residual layers.

    Input residuals are expected to have shape:
      * `(B, L, T, D)` for batches
      * `(L, T, D)` for a single example
      * `(T, D)` for a single-layer single example

    The logit at token t is `sum_l dot(W_l, residual[l, t]) + b`.
    """

    def __init__(self, config: ProbeConfig):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.zeros(config.n_layers, config.d_model))
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / (config.d_model ** 0.5))

    def forward_token_logits(self, residuals: torch.Tensor) -> torch.Tensor:
        """Return unsmoothed logits with shape `(B, T)`."""
        if residuals.dim() == 2:  # (T, D)
            residuals = residuals.unsqueeze(0).unsqueeze(0)
        elif residuals.dim() == 3:  # (L, T, D)
            residuals = residuals.unsqueeze(0)
        if residuals.dim() != 4:
            raise ValueError(f"residuals must be (B,L,T,D), (L,T,D), or (T,D); got {tuple(residuals.shape)}")
        if residuals.shape[1] != self.weight.shape[0] or residuals.shape[-1] != self.weight.shape[-1]:
            raise ValueError(
                "residual shape does not match probe weights: "
                f"residuals={tuple(residuals.shape)}, weights={tuple(self.weight.shape)}"
            )
        return torch.einsum("bltd,ld->bt", residuals.float(), self.weight.float()) + self.bias.float()

    def sequence_logits(self, residuals: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return one sequence-level logit per batch element."""
        token_logits = self.forward_token_logits(residuals)
        if attention_mask is None:
            attention_mask = torch.ones(token_logits.shape, dtype=torch.bool, device=token_logits.device)
        elif attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        attention_mask = attention_mask.to(device=token_logits.device, dtype=torch.bool)
        if self.config.score_mode == "max_ema":
            return ema_sequence_score(token_logits, attention_mask, self.config.window_size, self.config.ema_decay)
        zbar, score_mask = sliding_window_mean(token_logits, attention_mask, self.config.window_size)
        if self.config.score_mode == "final_swim":
            idx = score_mask.long().sum(dim=1).sub(1).clamp_min(0)
            compact = [zbar[b, score_mask[b]] for b in range(zbar.shape[0])]
            return torch.stack([row[i] for row, i in zip(compact, idx.tolist())])
        if self.config.score_mode == "max_swim":
            return zbar.masked_fill(~score_mask, -torch.inf).max(dim=1).values
        raise ValueError(f"unknown score_mode: {self.config.score_mode}")


def resolve_layer_positions(n_input_layers: int, layer_mode: str) -> list[int]:
    """Select layer positions relative to the residual tensor on disk."""
    if n_input_layers <= 0:
        raise ValueError("n_input_layers must be positive")
    mode = str(layer_mode).strip().lower()
    if mode == "all":
        return list(range(n_input_layers))
    if mode in {"middle", "mid"}:
        return [n_input_layers // 2]
    if mode == "early":
        return [n_input_layers // 4]
    if mode == "late":
        return [3 * n_input_layers // 4]
    if mode.startswith("every"):
        step_text = mode.replace("every", "").replace("_", "").strip()
        step = int(step_text or "1")
        return list(range(0, n_input_layers, max(step, 1)))
    if ":" in mode:
        parts = [int(x) if x else None for x in mode.split(":")]
        if len(parts) == 2:
            start, stop = parts; step = None
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError(f"bad layer range: {layer_mode}")
        return list(range(start or 0, stop if stop is not None else n_input_layers, step or 1))
    if "," in mode:
        return [int(x.strip()) for x in mode.split(",") if x.strip()]
    return [int(mode)]


def select_layers(residuals: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
    """Return residuals restricted to selected layer positions."""
    if residuals.dim() == 2:
        residuals = residuals.unsqueeze(0)
    if residuals.dim() != 3:
        raise ValueError(f"single-example residuals must be (L,T,D) or (T,D); got {tuple(residuals.shape)}")
    pos = torch.as_tensor(list(positions), dtype=torch.long, device=residuals.device)
    return residuals.index_select(0, pos)


def sliding_window_mean(token_logits: torch.Tensor, mask: torch.Tensor, window_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute causal SWiM-smoothed logits and the positions eligible for loss/scoring.

    For sequences with length >= M, eligible positions are M-1..T-1. For shorter
    sequences, only the final position is eligible and uses the average of available
    tokens. This matches the paper's "exclude t < M except short sequences" rule.
    """
    if token_logits.dim() != 2:
        raise ValueError("token_logits must be (B,T)")
    mask = mask.to(device=token_logits.device, dtype=torch.bool)
    B, T = token_logits.shape
    M = max(int(window_size), 1)
    z = token_logits.masked_fill(~mask, 0.0)
    counts_raw = mask.float()
    csum = F.pad(torch.cumsum(z, dim=1), (1, 0))
    ccnt = F.pad(torch.cumsum(counts_raw, dim=1), (1, 0))

    ends = torch.arange(T, device=token_logits.device) + 1
    starts = (ends - M).clamp_min(0)
    sums = csum[:, ends] - csum[:, starts]
    counts = (ccnt[:, ends] - ccnt[:, starts]).clamp_min(1.0)
    zbar = sums / counts

    lengths = mask.long().sum(dim=1)
    pos = torch.arange(T, device=token_logits.device).unsqueeze(0).expand(B, T)
    full_window_positions = pos >= (M - 1)
    final_positions_for_short = pos == (lengths.clamp_min(1) - 1).unsqueeze(1)
    score_mask = mask & torch.where(lengths.unsqueeze(1) >= M, full_window_positions, final_positions_for_short)
    return zbar, score_mask


def softmax_weighted_bce_loss(
    token_logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    window_size: int = 16,
    tau: float = 1.0,
) -> torch.Tensor:
    """SWiM + softmax-weighted binary cross entropy over token positions."""
    zbar, score_mask = sliding_window_mean(token_logits, attention_mask, window_size)
    labels = labels.float().to(device=token_logits.device).view(-1, 1).expand_as(zbar)
    per_token = F.binary_cross_entropy_with_logits(zbar, labels, reduction="none")
    tau = max(float(tau), 1e-6)
    weight_logits = (zbar / tau).masked_fill(~score_mask, -torch.inf)
    weights = F.softmax(weight_logits, dim=1).masked_fill(~score_mask, 0.0)
    return (weights * per_token).sum(dim=1).mean()


def ema_sequence_score(
    token_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    window_size: int = 16,
    ema_decay: float | None = None,
) -> torch.Tensor:
    """Streaming-friendly max-over-time EMA score.

    This stores one scalar state per sequence. The default decay `(M-1)/M` gives an
    EMA timescale roughly comparable to an M-token moving average.
    """
    if token_logits.dim() != 2:
        raise ValueError("token_logits must be (B,T)")
    mask = attention_mask.to(device=token_logits.device, dtype=torch.bool)
    B, T = token_logits.shape
    decay = float(ema_decay) if ema_decay is not None else (max(int(window_size), 1) - 1) / max(int(window_size), 1)
    decay = min(max(decay, 0.0), 0.9999)
    ema = torch.zeros(B, device=token_logits.device, dtype=token_logits.dtype)
    seen = torch.zeros(B, device=token_logits.device, dtype=torch.bool)
    best = torch.full((B,), -torch.inf, device=token_logits.device, dtype=token_logits.dtype)
    for t in range(T):
        active = mask[:, t]
        z_t = token_logits[:, t]
        ema = torch.where(seen, decay * ema + (1.0 - decay) * z_t, z_t)
        seen = seen | active
        best = torch.where(active, torch.maximum(best, ema), best)
    return best


def pad_residual_batch(examples: Iterable[tuple[torch.Tensor, torch.Tensor]], layer_positions: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of `(residuals, mask)` examples into `(B,L,T,D), (B,T)`."""
    selected = []
    masks = []
    for residuals, mask in examples:
        r = select_layers(residuals, layer_positions).float()
        m = mask.bool()
        selected.append(r)
        masks.append(m[: r.shape[1]])
    B = len(selected)
    L, D = selected[0].shape[0], selected[0].shape[-1]
    T_max = max(r.shape[1] for r in selected)
    x = torch.zeros(B, L, T_max, D, dtype=torch.float32)
    mask_out = torch.zeros(B, T_max, dtype=torch.bool)
    for i, (r, m) in enumerate(zip(selected, masks)):
        T_i = r.shape[1]
        x[i, :, :T_i, :] = r
        mask_out[i, : min(T_i, m.numel())] = m[:T_i]
    return x, mask_out
