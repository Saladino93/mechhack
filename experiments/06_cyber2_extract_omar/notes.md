# Experiment 06 (omar): Cyber-2 Activation Extraction

## Goal

Extract Gemma 4-31B-it residual activations at 13 layers for a balanced 1000-sample
training subset of the **cyber_2** task (high_risk_dual_use vs {dual_use, benign}),
so downstream probe-training (separate experiment) has the data it needs.

This is a pure data-extraction experiment — no probes are trained here.

## Method

### Data and labels
- Dataset: `cyber` train split (`datasets/cyber_probes/train.jsonl`, 5079 samples).
- Task `cyber_2` per `experiments/02_extract_activations/data.py:get_label_for_task`:
  - positive: `high_risk_dual_use`
  - negative: `dual_use` ∪ `benign`
  - other (`prohibited`): excluded.
- Train-set class counts: high_risk_dual_use=1052, dual_use=685, benign=1582.

### Selection (`select.py`, seed `random.Random(42)`)
- 500 positives drawn randomly from 1052 `high_risk_dual_use` (`rng.sample`).
- 500 negatives drawn from the union {dual_use, benign}, with a strong
  preference for `sample_id`s already on disk in `/home/ubuntu/extracts/cyber_all_omar/`
  (1406 dual_use+benign samples already extracted by experiment 03 and symlinked
  in). The pool of 1406 already-extracted negatives is much larger than the 500
  needed, so **all 500 negatives are reused, zero fresh negative extractions
  required**.
- Selection is then `rng.shuffle`'d so positives and negatives interleave.
- Resulting negative breakdown: 243 dual_use + 257 benign (close to the
  ~30/70 dual_use/benign ratio in the original 1406-sample pool, weighted by
  exp 03's selection).

### Why this is balanced and reproducible
- 500 / 500 by binary label, fixed seed.
- Class breakdown of negatives is documented in `selection.json:negative_label_counts`.
- The choice to reuse negatives biases the negative pool toward exp 03's
  shuffle, but exp 03 itself sampled with `random.Random(42)` from the full
  cyber-train {dual_use, benign}, so the resulting negatives remain a random
  subsample of the underlying population.

### Extraction (`extract.py`)
- Same hook-based extractor as `experiments/03_layer_sweep_omar/extract.py`:
  - 13 layers `[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]`
  - `chunked_sdpa_scope()` from `starter_code/chunked_sdpa.py` (required for
    Gemma's wide global-attn `head_dim=512`).
  - Forward-pre-hooks on the requested decoder layers + a forward-hook on the
    last layer to capture index 60.
  - fp16 storage on CPU; full residual stream `(L, N, d)` per `.pt`.
- Output dir: **`/home/ubuntu/extracts/cyber_all_omar/`** (shared across
  cyber_1/2/3). Existing dual_use+benign `.pt` files appear here as symlinks
  to `/home/ubuntu/extracts/03_layer_sweep_omar/` (set up once, manually).
- Resume logic: any sample whose `.pt` already exists is skipped (so the 500
  reused negatives are no-ops).

### Why a shared extracts dir
The disk has ~2.4 TB free, so dedup-by-symlink is more about cleanliness than
storage. With one shared dir, downstream code can simply read
`cyber_all_omar/<sample_id>.pt` regardless of which task selected it. Only
this experiment's `selection.json` decides which IDs belong to cyber_2.

## How to run

```bash
python experiments/06_cyber2_extract_omar/select.py
python experiments/06_cyber2_extract_omar/extract.py 2>&1 | tee experiments/06_cyber2_extract_omar/extract.log
```

## Results

- `selection.json`: 1000 samples (500 high_risk_dual_use + 500 negatives;
  negatives = 243 dual_use + 257 benign).
- New extractions written to `/home/ubuntu/extracts/cyber_all_omar/` —
  500 fresh `high_risk_dual_use` `.pt` files (one per positive). Negatives
  were skipped (already present as symlinks to exp 03's extracts).
- Extraction stats and per-sample timing live in `extraction_metadata.json`
  (n_ok, n_skipped, n_errors, wall_minutes, peak_vram_gb_overall).
- Full extraction log: `extract.log`.

### Final stats
- `n_ok = 500`, `n_skipped = 500` (the reused negatives), `n_errors = 0`.
- Wall time: **8.8 min** for the 500 fresh extractions.
- Peak VRAM: **67.0 GB** (largest prompt cyber_4022 at N=7713 reached 65.9 GB;
  the 60-62 GB plateau matches exp 03's mid-prompt baseline).
- No OOMs. No samples excluded.
- Total `.pt` files in `/home/ubuntu/extracts/cyber_all_omar/` after this run:
  1906 (1406 from exp 03 symlinks + 500 fresh high_risk_dual_use).

## Takeaways

- Sharing a single extracts dir + symlinks let cyber_2 reuse 100% of its
  negatives from exp 03 — only the 500 positive forward passes were new work.
- This pattern generalises to cyber_3 (exp 07).

## Next

- Experiment 07: same recipe for cyber_3 (`prohibited` vs the 3 other classes).
- Probe training across cyber_1/2/3 will be a separate experiment that just
  reads `cyber_all_omar/` and the per-task `selection.json`s.

## Results (post-disconnect rerun, Sat May  9 12:13 UTC 2026)

The original training run was killed mid-sweep (mean-pool reached L25 then died).
Re-launched `train_probes.py` from scratch under `nohup` (CPU only, PID 63218);
ran for ~34 min wall-clock alongside concurrent CPU jobs (exp 05 train_heads,
exp 09 c_sweep), then wrote a complete `results.json` with all 13 mean +
13 last_token layers, plus all three plot PNGs.

### Headline (cyber_2, 500 hrdu vs 500 (243 du + 257 ben), 5-fold stratified CV)

- **Best: last_token, L60 — test AUC 0.955 ± 0.017** (acc 0.896 ± 0.027,
  train AUC 0.983 — first slightly-non-saturated train layer, suggests the
  final-token last-layer probe still has a touch of generalization headroom).
- Mean-pool peak: **L40, AUC 0.946 ± 0.012** (acc 0.892, train AUC 1.000);
  the mean-pool curve is a smooth ramp 0.881 → 0.946 from L0 to L40, then
  drifts slightly down to 0.936 at L60.
- Last_token L0 is **flat at AUC=0.500, acc=0.500** — exactly the same
  degenerate behavior seen in exp 07 cyber_3 last_token L0, and for the
  same reason: the embedding-layer residual at the final masked token
  carries no contextual information that's discriminative for the task,
  so all folds collapse to the constant predictor. By L5 last_token jumps
  to 0.804 and then climbs cleanly to 0.955 at L60.
- Last_token monotonic (with one small dip at L30 to 0.893) and overtakes
  mean-pool only at L60; mean-pool is the safer pick across mid-stack.

### Comparison with neighbouring tasks

| task    | best pooling | best layer | best AUC          |
|---------|--------------|------------|-------------------|
| cyber_1 | mean         | L40        | 0.983 (exp 03)    |
| cyber_2 | last_token   | L60        | **0.955 ± 0.017** |
| cyber_2 | mean         | L40        | 0.946 ± 0.012     |
| cyber_3 | mean         | L35        | 0.955 (exp 07)    |
| cyber_3 | last_token   | L60        | 0.923 (exp 07)    |

- cyber_1 (dual_use vs benign) remains the easiest, clean 0.983 at mean L40.
- cyber_2 (high_risk_dual_use vs {dual_use union benign}) and cyber_3
  (prohibited vs the rest) sit at essentially the same difficulty
  (peak ~0.955), which is interesting because cyber_2 has to discriminate
  *within* the high-risk vs dual_use boundary while cyber_3 separates the
  much-more-distinct `prohibited` class. Apparently both signals are
  cleanly linearly readable from the late residual stream, just at slightly
  different layer/pooling sweet spots: cyber_3 prefers mid-stack mean-pool
  (L35), cyber_2 prefers final-token L60.
- The cyber_2/cyber_3 last_token L0 = 0.5 collapse is reproducible across
  both experiments; not a bug, just the embedding layer carrying no
  task-discriminative info at the prompt-final position.
