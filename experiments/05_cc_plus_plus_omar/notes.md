# Experiment 05 (omar): Constitutional Classifiers++ probe heads

## Goal

Reproduce and head-to-head compare three probe-training techniques from the
Constitutional Classifiers++ paper (Cunningham et al. 2026, summary in
`papers/SUMMARY.md` entry 8) against a simple linear+mean-pool baseline, on
identical data. The three techniques are:

1. **Multi-layer concat probe (head A).** Mean-pool every captured layer, concat
   into a single feature vector, train one linear classifier. Paper claim:
   "probes using all layers (concatenated) beat single-layer probes."
2. **SWiM (sliding-window-mean then max) probe (head B).** Train a per-token
   linear probe; at inference, smooth its per-token logits with a centred
   sliding-window mean (window=16) and aggregate via max-over-tokens. Designed
   to be robust to token-position noise without losing peak signal.
3. **Softmax-weighted BCE loss (head C).** Same per-token probe, but training
   loss weights per-token BCE by `softmax(token_logits)` so that gradient
   concentrates on positions where the probe is already most confident
   (the paper's hypothesis: "where the probe is most confident is where the
   signal lives"). Reported with both SWiM and plain max aggregations at eval.
4. **Baseline (head D).** Single-layer linear+mean-pool at the layer that won
   the exp 03 sweep (layer 40).

## Method

### Data
- Reuses the exp 03 extracts at `/home/ubuntu/extracts/03_layer_sweep_omar/`:
  Gemma 4-31B-it residuals at layers `[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]`,
  `d_model=5376`, fp16, attention masks included.
- Sample selection: identical to exp 03's `selection.json` — balanced 1000
  cyber_1 samples (500 dual_use + 500 benign, seed 42). Script polls
  the extracts directory until all 1000 `.pt` files are present before
  training, so the upstream extraction job (running on GPU in another
  process) is allowed to finish first.
- Within those 1000: a single `train_test_split(test_size=0.2, random_state=0,
  stratify=y)` → 800 train / 200 test. **Identical split for every head.**

### Heads
- **A (concat).** Per sample: mean-pool each of 13 layers → 13 vectors of
  length 5376 → flatten to 69 888-dim feature. `sklearn.LogisticRegression(solver='lbfgs', max_iter=2000)`,
  C swept over `{0.01, 0.1, 1.0, 10.0}` (best chosen on a 20% inner val split
  of the 800-sample train set), then refit on full 800.
- **D (baseline).** Mean-pool at layer 40 only → 5376-dim feature. Same
  sklearn LR with `C=1.0` (matches exp 03 setup). Single deterministic run.
- **B (SWiM).** PyTorch `nn.Linear(5376, 1)` per-token probe trained on layer-40
  per-token residuals. Loss = `BCEWithLogitsLoss` per token, sample-level label
  broadcast to every masked position, mean over the sample's tokens, mean over
  batch. Optimiser AdamW lr=1e-3, batch size 32, 50 epochs, early-stop on inner
  validation AUC (20% of train, stratified). At inference: per-token logits →
  centred 1D mean filter (window=16, edge padding) → max → sigmoid.
- **C (softmax-weighted BCE).** Same probe and schedule as B, but training loss
  is `sum_t( softmax(logits_t).detach() * BCE_per_token_t )` per sample
  (gradient flows through the BCE, not through the softmax weights). Eval
  reported with both `swim_max` and plain `max` aggregations.

Per-token features for B/C are pre-loaded into RAM as fp32 tensors at layer 40
only (one `(N_tok_i, 5376)` per sample), to keep training fast on CPU.

### Seeds and uncertainty
- B and C run for **5 torch seeds** `0..4` (seeds control torch RNG, batch
  shuffle, and the inner train/val split). Mean ± std reported.
- A and D are deterministic single runs (sklearn lbfgs).
- 1000-resample bootstrap 95% CI on test AUC, **per run**. For B/C the reported
  CI bounds are means across the five per-seed CIs (a quick proxy; not a
  multi-seed posterior).

### Hyperparameters at a glance
| Knob              | Value                                          |
|-------------------|------------------------------------------------|
| Best layer        | 40 (from exp 03)                               |
| Split seed        | 0 (test_size=0.2, stratified)                  |
| C grid (head A)   | {0.01, 0.1, 1.0, 10.0}                         |
| Torch seeds       | {0, 1, 2, 3, 4}                                |
| Torch epochs      | 50                                             |
| Torch batch       | 32                                             |
| Torch optimiser   | AdamW, lr 1e-3                                 |
| SWiM window       | 16 (centred mean, edge padding)                |
| Bootstrap         | 1000 resamples for AUC CI                      |
| Device            | CPU only (GPU is in use by another job)        |

## How to run

```bash
# 1) train & evaluate all heads (polls until extracts are ready)
python experiments/05_cc_plus_plus_omar/train_heads.py
# 2) draw plots from the resulting results.json + swim_traces.npz
python experiments/05_cc_plus_plus_omar/plot.py
```

## Results (partial — run still in progress, captured 2026-05-09)

The run is still mid-execution: Head D and Head A are complete, and seed 0 of
Head B and Head C are complete. Seeds 1–4 of B and C are still pending
(~60-90 min remaining at write-time). The numbers below are from the live
`train.log`; final numbers (with proper 5-seed mean±std) will replace them
when the run finishes.

### Per-head test AUC (200-sample held-out test, same split for every head)

| Head | Method | Test AUC | Test acc | Notes |
|---|---|---:|---:|---|
| **D** (baseline) | linear + mean-pool @ L40 | **0.986** (CI [0.974, 0.995]) | 0.930 | single deterministic run; matches exp 03 |
| **A** (concat) | sklearn LR over all 13 layers concat (69,888-dim), C-sweep | **0.984** (CI [0.966, 0.996]) | 0.945 | C=10 won inner val |
| **B seed 0** (SWiM) | per-token probe @ L40, sliding-window-mean → max | **0.975** (CI [0.956, 0.990]) | 0.570 | seeds 1-4 still running |
| **C seed 0** (softBCE, max\_only agg) | per-token probe @ L40, softmax-weighted BCE | **0.985** | 0.890 | seeds 1-4 still running |
| **C seed 0** (softBCE, swim\_max agg) | same head, SWiM aggregation at eval | **0.983** | 0.920 | seeds 1-4 still running |

### Head A C-sweep (inner val AUC at 20% of train)

| C | val AUC |
|---:|---:|
| 0.01 | 0.9663 |
| 0.1 | 0.9684 |
| 1.0 | 0.9692 |
| 10.0 | **0.9694** ← chosen |

Test AUC differences across C are within ±0.005, so the probe is robust to L2
strength — same finding as exp 09 D2 mean_L30 (which sweeps C on a different
split and lands the same conclusion).

### Cross-head comparison (so far)

- All four head families land within ±0.011 AUC of each other (0.975 to 0.986).
- The simple linear baseline (Head D) is currently the **highest** point estimate.
  This is consistent with the exp 09 D2 finding that the cyber_1 probe doesn't
  need a richer architecture — the limit is data, not capacity.
- Head C's `max_only` aggregation (0.985, seed 0) edges out `swim_max` (0.983)
  on this single seed. This will firm up over the remaining 4 seeds before any
  conclusion about SWiM vs plain-max is fair.
- The accuracy column has more spread than AUC — Head B's seed-0 acc 0.570
  with f1 0.699 looks low at a glance, but the BCE column (0.607) tells the
  story: SWiM's max-aggregation produces high-magnitude logits that drive
  predictions to the extreme of the sigmoid, so the default threshold 0.5 is
  miscalibrated. AUC (threshold-free) is the right metric for these heads.

### Connection to Phase 9c (Kramar MultiMax / Rolling Attention)

Head B (SWiM) and Head C (softmax-weighted BCE) are *related* to Kramar 2026's
MultiMax + Rolling Attention but are not identical:

- SWiM ≈ a 1-D version of Kramar's *Rolling Attention*: both replace mean
  pooling with sliding-window aggregation followed by a cross-window max.
- Softmax-weighted BCE is a training-time twist orthogonal to MultiMax (which
  is an *inference-time* per-head hard-max).
- The full MultiMax + Rolling head (TODO 9c) has not been implemented yet.

If the partial results hold up after seeds 1-4 (i.e. Heads B and C don't
overtake the linear baseline), it weakens the motivation for 9c on cyber_1
specifically — the per-token aggregation tricks don't appear to pay on this
task. They may still pay on long-prompt OOD (the regime Kramar's paper targets,
not our short-prompt CV regime); that's a separate eval and is captured in
the length-stratified analysis (TODO 9a, exp 10).

## Takeaways (partial)

1. **All four heads cluster around 0.98 AUC.** The cyber_1 ceiling under our
   1000-sample, 800-train regime is ~0.985-0.986 regardless of head choice.
2. **Linear baseline is competitive.** Head D (0.986) ≥ Head A (0.984) ≥
   Head C max\_only seed 0 (0.985) ≥ Head C swim\_max (0.983) ≥ Head B (0.975).
   Capacity is not the bottleneck.
3. **Final numbers pending** — wait for seeds 1-4 of Heads B and C before
   making any claim about SWiM vs softBCE or quoting B/C means.

## Caveats

- **Single layer for the per-token heads.** SWiM and softmax-weighted BCE both
  use only layer 40. The paper applies the technique stack on top of a
  multi-layer concat; combining the two ideas is exp 06's job.
- **Inner val split for C selection in head A** uses `random_state=split_seed`,
  so the picked C is mildly correlated with the test split. With four C values
  and 800 samples, the leakage risk is small but not zero.
- **B/C bootstrap CI** is averaged across seeds rather than computed across the
  union of seed×bootstrap predictions. It is a rough envelope, not a strict
  posterior.
- **Deterministic baseline.** Head D is one run; we don't have a multi-seed
  std for it. lbfgs is deterministic given the data, so this is fine.
