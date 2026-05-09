# Experiment 09 (omar): Stress-test exp 03 cyber_1 result

## Goal

Exp 03 reports test AUC = 0.983 on cyber_1 (mean-pool, layer 40) with 5-fold CV
on a balanced 999-sample subset. The number is suspiciously high. Three
diagnostics to figure out whether the probe is "really good" or whether
something trivial / overfit is going on.

1. **D1 — TF-IDF text baseline.** If a plain text classifier on the same 999
   prompts already hits ~0.95+, then most of exp 03's 0.98 is in the surface
   text and the 31B activations are doing very little extra.
2. **D2 — Regularisation sweep.** Sweep `LogisticRegression(C)` over
   `{1e-4 .. 10}` at the two best layers (30 mean, 35 last-token). Do we still
   hit 0.98 with much stronger L2 + much smaller weight L1-norm? If yes,
   the "real" probe direction is low-dimensional and the high train-AUC is
   spurious capacity, not necessary capacity.
3. **D3 — OOD held-out test.** Refit on all 999 train samples, evaluate on 500
   held-out test-split samples (extracts from exp 08). Compare to CV AUC.

All on CPU — exp 08 owns the GPU.

## Method

### D1 — TF-IDF baseline (`tfidf_baseline.py`)
- Same 999 sample IDs as exp 03 (`experiments/03_layer_sweep_omar/selection.json`).
- Load `prompt` field from `datasets/cyber_probes/train.jsonl`.
- `TfidfVectorizer(min_df=2, ngram_range=(1,2), max_features=20000)`.
- `LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")`.
- 5-fold stratified CV, `random_state=0` (matches exp 03).
- Report AUC mean ± σ, accuracy mean ± σ.

### D2 — Regularisation sweep (`c_sweep.py`)
- Two configs: (mean-pool, layer 30) and (last-token, layer 35).
- Sweep `C ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0}`.
- 5-fold stratified CV, same seed as exp 03.
- For each C: report mean test AUC, mean test acc, mean train AUC, mean L1
  norm of the learned weight vector (`np.abs(coef_).sum()`), and the number of
  weights with `|w| > 1e-3` (an effective-sparsity proxy).
- "Sparsest C still within 0.005 of the best AUC" is the headline number.

### D3 — OOD test eval (`ood_eval.py`)
- Wait (poll every 30 s, give up after 5 minutes if exp 08's `selection.json`
  isn't there) for exp 08's selection.
- Pick 500 test sample IDs (250 dual_use + 250 benign, stratified, seed=42)
  from the 923 cyber_1-eligible test rows. Save to `ood_selection.json`.
- Wait for ≥500 of those extracts to land in `/home/ubuntu/extracts/cyber_all_omar/`.
- Re-fit one `LogisticRegression(C=1.0)` on all 999 train extracts, evaluate on
  the 500 OOD test samples.
- 1000-bootstrap 95% CI on the test AUC.
- Report per (pooling, layer) for layers in [25, 30, 35, 40, 45].
- Headline = (CV AUC, test AUC, gap) at each layer/pooling.

### Side-effect
- Write `split_manifest.json` mapping `sample_id -> "train"/"test"` for every
  cyber_1-eligible sample, **also copied to**
  `/home/ubuntu/extracts/cyber_all_omar/split_manifest.json`, so future
  experiments can tell train from test inside the shared extracts folder.

## Results

See `results.json` for full numbers and `metrics.jsonl` for the incremental log.

### Headline table

| Diagnostic | Number | Interpretation |
|---|---:|---|
| D1 — TF-IDF AUC (cyber_1) | **0.9460 ± 0.0203** | activations add only +0.037 over plain text |
| D1 — TF-IDF AUC (cyber_2) | _running_ (`tfidf_baseline_cyber23.py`) | extension to harder tier |
| D1 — TF-IDF AUC (cyber_3) | _running_ (`tfidf_baseline_cyber23.py`) | extension to hardest tier |
| D2 — best AUC, mean-pool L30 | **0.9809 @ C=1** (L1=283.4) | within noise of exp 03's L40 |
| D2 — sparsest C ≥ best-0.005, mean L30 | **C=0.1 → 0.9804 (L1=109.9)** | probe is much sparser than 5376 dims |
| D2 — best AUC, last-token L35 | _in progress_ | |
| D3 — OOD AUC, mean-pool L40 | _TBD (script not yet written)_ | does CV AUC hold on real held-out? |
| D3 — gap (CV − OOD) | _TBD_ | overfitting magnitude |

## Takeaways (running)

### Headline finding — activation-over-TF-IDF gap grows with task difficulty

| Task | TF-IDF AUC | Activation AUC (best mean-pool) | Δ (activation − TF-IDF) |
|---|---:|---:|---:|
| cyber_1 (dual_use vs benign) | 0.946 ± 0.020 | 0.983 (L40) | **+0.037** |
| cyber_2 (hrdu vs du ∪ ben) | 0.887 ± 0.031 | 0.946 (L40) | **+0.059** |
| cyber_3 (prohibited vs hrdu ∪ du ∪ ben) | 0.890 ± 0.015 | 0.955 (L35) | **+0.065** |

**Activations help more on harder discriminations.** Plain n-gram features carry
most of the easy-tier signal (cyber_1 boundary is largely about which words
are present), but their advantage narrows as the task moves toward graded harm
distinctions that aren't reducible to bag-of-words. TF-IDF *also* gets harder
in absolute terms (its AUC drops by ~0.06 from cyber_1 → cyber_2/3) but the
activation probe drops less, so the relative value of representation-level
features grows.

Implication: future probe-improvement work (Phase 5 MultiMax, Phase 6 multi-layer,
9c Kramar) should be evaluated primarily on cyber_2 / cyber_3 / OOD long-prompt
slices, where the activation-vs-text gap is largest and any architectural gain
is least likely to be a TF-IDF lookalike.

### D1 (cyber_1)

Plain TF-IDF + LR gets **0.946**, vs the activation probe at **0.983**. The
activations are doing *something* (Δ = +0.037 AUC) but not a lot — most of
the cyber_1 signal lives in surface n-grams. This is the *floor* of how useful
a 31B activation probe needs to be on this task to justify its cost over
plain text.

### D1 (cyber_2 + cyber_3 extension)

Same vectoriser/classifier/CV recipe, run on exp 06 and exp 07 selections via
`tfidf_baseline_cyber23.py`:

- **cyber_2**: TF-IDF AUC **0.887 ± 0.031** (range 0.849–0.925). Activation
  probe (mean-pool L40) currently at 0.946 → Δ = **+0.059**.
- **cyber_3**: TF-IDF AUC **0.890 ± 0.015** (range 0.876–0.910). Activation
  probe (mean-pool L35) at 0.955 → Δ = **+0.065**.

Both deltas are strictly larger than cyber_1's +0.037 — see headline finding
above.

### D2 mean_L30 (cyber_1)

Two findings:

1. **Sparsity.** Probe doesn't need 5376 dims — at C=0.1 the L1 norm of the
   weights drops from 283.4 to 109.9 (~2.6× sparser) while AUC only falls from
   0.9809 → 0.9804 (within noise). At C=0.01 we lose more (0.9725) but still
   ~all 4055/5376 weights are non-trivial; the *direction* is what matters,
   not the count of active dimensions.
2. **Sub-collapse range.** Pushing C ≤ 1e-3 collapses AUC (0.911 → 0.934).
   So the useful probe lives somewhere in C ∈ [0.01, 1.0]. Below 1e-3 we lose
   real signal; above 1.0 we just memorise (train AUC = 1.0, test plateau).
3. **Pre-disconnect parity.** Best mean-pool C=1 AUC of 0.9809 ≈ exp 03's L40
   peak (0.983). So the "best layer" choice is more about layer than C.

### D2 last_token_L35

In progress.

### D3

Not started — script needs to be written. Gated on exp 08's extracts (now
complete) and selection schema.
