# Experiment 14 (omar): Cross-task probe transfer

## Goal

Take each cyber task's best-layer LR probe and predict on the OTHER two
tasks' selections. Tells us whether the harm-tier directions in residual
space are aligned (one direction handles all 3) or orthogonal (each tier needs
its own).

## Method

- For each `train_task ∈ {cyber_1, cyber_2, cyber_3}`:
  - Load mean-pool features at the train_task's best layer (cyber_1 L40,
    cyber_2 L40, cyber_3 L35) from `/home/ubuntu/extracts/cyber_all_omar/`.
  - Use the train_task's `selection.json` to fit
    `LogisticRegression(C=1.0)` on the in-scope subset.
- For each `test_task`, predict on the test_task's selection (with its own
  labels) using the train_task-fitted probe. Report AUC.
- 3 × 3 = 9 cells. Diagonal cells (train == test) are training-fit AUCs (≈
  1.000) and serve as a sanity check.

## Result

| train ↓ \ test → | cyber_1 | cyber_2 | cyber_3 |
|---|---:|---:|---:|
| **cyber_1** | 1.000 | 0.886 | 0.713 |
| **cyber_2** | 0.913 | 1.000 | 0.794 |
| **cyber_3** | 0.803 | 0.733 | 1.000 |

(diagonal is the training fit — sanity check, not generalization)

## Implications

1. **Off-diagonal AUCs are 0.71–0.91, all above chance** — the harm-tier
   directions are RELATED. A probe trained on cyber_1 carries real signal
   about cyber_2 (AUC 0.886, ≈ 1 σ below cyber_2's own peak 0.946).
2. **The prohibited tier (cyber_3) is the most different.** The two cells
   involving cyber_3 (0.713 cyber_1→cyber_3 and 0.733 cyber_2→cyber_3) are
   the weakest. Prohibited content is qualitatively different from dual-use
   in residual space, not just "more harmful".
3. **Asymmetry**: cyber_2 → cyber_1 (0.913) > cyber_1 → cyber_2 (0.886). The
   "high-risk direction" generalizes back to dual-use better than vice versa.
   Plausible reading: cyber_2's positives include the upper end of
   "dual_use-flavored" content, so the direction it learns is partly aligned
   with cyber_1's. The reverse — using only dual_use_vs_benign signal to
   predict hrdu vs (du ∪ ben) — has less to work with.
4. **No single linear direction** explains all three. A multi-layer / multi-task
   probe (or one of CC++ Head A's multi-layer concat formulations) would
   probably do better cross-task than any single-task probe.

## Outputs

- `results.json` — 3×3 AUC matrix + per-pair sample sizes + best layers used.
- `matrix.png` — heatmap visualization with AUC values overlaid.

## Caveats

- Each task's "best layer" was chosen on its own data. The OFF-diagonal
  evaluations re-use that task's best-layer features, so we're really
  asking "does the cyber_1 probe generalize to cyber_2/3 at L40?" rather
  than "does cyber_1's probe generalize at the optimal layer for the
  target task?".
- LR with `C=1.0` and no inner-CV. A cleaner ablation would also show the
  effect of refitting at a different C or with a sparser regularisation,
  but for cross-task the current setting is fine.

## When to look at this

After cross-task finishes (`tail experiments/14_cross_task_omar/cross_task.log`),
the headline is the 3×3 matrix in `results.json` and `matrix.png`. If
off-diagonals land >0.85, we have strong evidence for a shared cyber-harm
direction; <0.7 means the tiers are genuinely separate.
