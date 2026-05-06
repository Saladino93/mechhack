# Refusal-Probe Dataset — Qwen 3.6-27B

Same schema as `gemma4_31b/`, run through `Qwen/Qwen3.6-27B`. See [`../gemma4_31b/README.md`](../gemma4_31b/README.md) for details.

## Files (parallel structure)

Filtered to prompts ≤ 8192 tokens (12% dropped) — same cap as Gemma side.

| File | Rows | Notes |
|---|---|---|
| `attacks_full.jsonl` | 878 | All rollouts ≤ 8192 tokens |
| `train_split.jsonl`  | 589 | |
| `test_split.jsonl`   | 289 | |
| `attribution_eval.jsonl` | 62 | Refusal-only, ≤2048 tokens, test split |

## Notable difference from Gemma

Qwen tends to refuse fewer of these prompts, which gives a smaller pool of refusal rollouts for Level-2 evaluation. The refusal probe also tends to be harder to fit on Qwen than on Gemma — possibly because Qwen's hybrid architecture diffuses the refusal signal differently. **Open question for participants**: is Qwen's refusal signal genuinely weaker in residuals, or are we missing the right features?

## Architecture note

Qwen 3.6-27B is a hybrid attention model:
- 16 full-attention layers
- 48 DeltaNet (linear-attention) layers
- 64 layers total, hidden_size=5120, head_dim=256

The hybrid arch may explain why per-token attribution behaves differently from Gemma's all-attention architecture.

## Suggested experiments

- Probe at different layer depths — Qwen's middle layer (32) is half-way through, but full-attention layers cluster every 4. Try probing only the full-attention layers (4, 8, 12, …, 64).
- Probe specifically on `thinking_content` representations rather than action-text representations.
- Multi-head attention probes — Qwen may need more capacity to extract its diffuse refusal signal.
