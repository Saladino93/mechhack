# Experiment 07 (omar): Cyber-3 Activation Extraction

## Goal

Extract Gemma 4-31B-it residual activations at 13 layers for a balanced 1000-sample
training subset of the **cyber_3** task (`prohibited` vs {high_risk_dual_use,
dual_use, benign}), so downstream probe-training (separate experiment) has
the data it needs.

This is a pure data-extraction experiment — no probes are trained here.

## Method

### Data and labels
- Dataset: `cyber` train split (`datasets/cyber_probes/train.jsonl`, 5079 samples).
- Task `cyber_3` per `experiments/02_extract_activations/data.py:get_label_for_task`:
  - positive: `prohibited`
  - negative: `high_risk_dual_use` ∪ `dual_use` ∪ `benign`
- Train-set class counts: prohibited=1760, high_risk_dual_use=1052, dual_use=685, benign=1582.

### Selection (`select.py`, seed `random.Random(42)`)
- 500 positives drawn randomly from 1760 `prohibited` (`rng.sample`).
- 500 negatives drawn from the union of the 3 negative classes, with a strong
  preference for `sample_id`s already on disk in `/home/ubuntu/extracts/cyber_all_omar/`.
  After exp 06 finishes, that dir contains:
  - 1406 dual_use+benign extracts (symlinks to exp 03's output)
  - 500 high_risk_dual_use extracts (from exp 06)
  → 1906 already-extracted negatives, far more than the 500 needed.
- Selection is then `rng.shuffle`'d so positives and negatives interleave.

**Run order matters:** `experiments/06_cyber2_extract_omar/extract.py` must
finish before `experiments/07_cyber3_extract_omar/select.py` runs, otherwise
the negative pool will only contain dual_use+benign and miss high_risk_dual_use.
We document the resulting per-class breakdown in
`selection.json:negative_label_counts`.

### Why this is balanced and reproducible
- 500 / 500 by binary label, fixed seed.
- Class breakdown of negatives is documented in `selection.json:negative_label_counts`.
- The choice to reuse extracts biases the negative pool toward the union of
  exp 03 and exp 06 selections, but each was already a `random.Random(42)`
  draw from the underlying population, so the resulting negatives remain a
  random subsample.

### Extraction (`extract.py`)
- Identical to exp 06's `extract.py` apart from the task tag — same
  hook-based extractor, same 13 layers `[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]`,
  same `chunked_sdpa_scope()`, same fp16 storage, same `(L, N, d)` per-`.pt`
  schema as exp 03.
- Output dir: **`/home/ubuntu/extracts/cyber_all_omar/`** (shared across
  cyber_1/2/3).
- Resume logic: any sample whose `.pt` already exists is skipped, so all
  reused negatives are no-ops.

### Why a shared extracts dir
Disk has ~2.4 TB free, so this is mostly about cleanliness rather than space:
downstream code reads `cyber_all_omar/<sample_id>.pt` regardless of which
task selected it; only this folder's `selection.json` decides which IDs
belong to cyber_3.

## How to run

```bash
# Must run AFTER exp 06's extract.py finishes
python experiments/07_cyber3_extract_omar/select.py
python experiments/07_cyber3_extract_omar/extract.py 2>&1 | tee experiments/07_cyber3_extract_omar/extract.log
```

## Results

- `selection.json`: 1000 samples (500 prohibited + 500 negatives drawn from
  the 3 already-extracted negative classes; per-class counts logged in the
  JSON).
- New extractions written to `/home/ubuntu/extracts/cyber_all_omar/` —
  500 fresh `prohibited` `.pt` files (one per positive). Negatives skipped
  (already present from exp 03 + exp 06).
- Extraction stats and per-sample timing live in `extraction_metadata.json`.
- Full extraction log: `extract.log`.

### Final stats
- `selection.json` negative breakdown: 138 high_risk_dual_use + 176 dual_use +
  186 benign (sum = 500). All 500 negatives reused from exp 03 + exp 06.
- `n_ok = 500`, `n_skipped = 500`, `n_errors = 0`.
- Wall time: **6.2 min** for the 500 fresh extractions (faster than cyber_2
  because prohibited prompts run shorter on average).
- Peak VRAM: **66.8 GB**.
- No OOMs. No samples excluded.
- Total `.pt` files in `/home/ubuntu/extracts/cyber_all_omar/` after this
  run: **2406** (1406 dual_use+benign symlinks from exp 03 + 500
  high_risk_dual_use from exp 06 + 500 prohibited from exp 07).

## Takeaways

- Sharing a single extracts dir + symlinks across cyber_1/2/3 lets each
  successive task reuse all earlier extracts as negatives, so only the new
  positive class triggers fresh GPU work.

## Next

- Probe training across cyber_1/2/3 will be a separate experiment that just
  reads `cyber_all_omar/` and the per-task `selection.json`s.

## Results (post-disconnect rerun, Sat May  9 11:42:48 UTC 2026)

Probe training (`train_probes.py`) finished before the disconnect (final line
of `train.log`: "Results written to .../results.json"). After reconnecting I
re-ran only `plot.py` (CPU-only, GPU left to other agents); results.json was
verified complete with `tasks.cyber_3.per_pooling.{mean,last_token}` each
holding 13 layer entries (0,5,…,60) on 500 / 500 stratified 5-fold CV.

**Headline.** Best probe is mean-pool at **layer 35**, test ROC-AUC =
**0.955 ± 0.016** (acc 0.886 ± —, train AUC 1.000). Last-token pooling peaks
much later at layer 60 with AUC 0.923 ± 0.012; mean-pool dominates last-token
across every middle layer. cyber_3 is meaningfully harder than the cyber_1
baseline from exp 03 (mean-pool peak **0.983** at layer 40) — the
prohibited-vs-mixed-negatives boundary loses ~0.03 AUC versus the cleaner
prohibited-vs-benign split. It comes in slightly above exp 06's cyber_2
(mean-pool peak **0.938** at layer 25), suggesting prohibited content is more
linearly separable from the broad negative pool than `high_risk_dual_use` is
from `{dual_use, benign}`. Peak layer drifts later as the task gets harder
(cyber_2 → 25, cyber_3 → 35, cyber_1 → 40).

**Plot files produced** (all in this dir):
- `auc_vs_layer.png` — ROC-AUC vs layer with train overlay
- `acc_vs_layer.png` — accuracy vs layer with train overlay
- `metrics_vs_layer.png` — combined two-panel figure

**Caveats.**
- `plot.py` titles still read "cyber_1" (cosmetic; the data plotted is cyber_3
  via the `tasks` dict). I did not patch this — runs cleanly, labels in the
  legend correctly say `cyber_3`.
- exp 06 `results.json` only contains mean-pool, so the cyber_2 comparison
  here is mean-pool only.
- Layer 0 last-token AUC is exactly 0.5 (degenerate — likely all-equal
  features at the embedding for that pooling); does not affect any other
  layer.
- `plot.log` captured stdout/stderr from the rerun.
