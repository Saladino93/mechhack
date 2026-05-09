# Experiment 03 (omar): Layer Sweep on Cyber Probes

## Goal
Find which layer of Gemma 4-31B-it gives the best linear-probe AUC for the cyber tasks
(starting with cyber_1, dual_use vs benign). Output: an AUC-vs-layer plot per task.

## Method
- **Model:** Gemma 4-31B-it (downloaded to `/home/ubuntu/models/Gemma-4-31B-it`)
- **Sample selection:** **First 1000 cyber-train samples** (from `datasets/cyber_probes/train.jsonl`)
  filtered to category ∈ {dual_use, benign}. **This is a biased subset** — the
  file is roughly grouped by category, so the first 1000 eligible rows pick up
  *all* 685 dual_use samples + only the first 315 benign ones. The remaining
  1267 benign samples in cyber-train were dropped. Result: 685 / 315 (≈68/32).
- **No positives for cyber_2 / cyber_3** in this subset, so those tasks were
  skipped (will be addressed in `05_full_cyber_omar`).
- **Layers extracted:** `[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]`
  — 13 indices into the residual stream (0 = embedding output, k>0 = output of
  block k-1; index 60 = output of last block).
- **Pooling:** Mean over attention-masked tokens, per layer → one feature vector
  of shape (d_model,) per (sample, layer). d_model = 5376.
- **Probe:** `sklearn.linear_model.LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')`,
  one model per (layer, task). Single seed (`random_state=0`).
- **Evaluation split:** **Within those 1000 train samples**, sklearn does an
  80/20 stratified split → 800 fit / 200 evaluate. **The official `test.jsonl`
  is never touched** — these AUC/accuracy numbers are an in-sample holdout, not
  the project's official benchmark.
- **Uncertainty:** 1000-sample bootstrap of test predictions for the AUC 95% CI.

## Caveats / known issues
1. **Sample selection is non-random** (see above). Any conclusion about absolute
   AUC level is conditional on this subset. The shape of the curve (best layer ≈
   40) should be more robust.
2. **No official test-set evaluation.** The 200-sample "test" is part of the
   official train split. To get the project metric, we'd need to extract
   activations on `test.jsonl` too and evaluate there.
3. **Single seed, no probe-hyperparameter sweep.** L2 strength fixed at C=1.0.
4. **Mean-pooling only.** Other aggregations (last-token, MultiMax, multi-layer
   concat, logit-lens, polynomial lift) are exp 04's job.
5. **Class imbalance: 68/32.** Majority-class accuracy = 0.685; AUC is the
   more honest headline metric.

## How to run

```bash
# 0) one-time: download Gemma (~62GB, 5-15 min on a fast link)
HF_TOKEN=... python starter_code/download_models.py --out_dir /home/ubuntu/models --only gemma

# 1) extract 13 layers of residuals for 1000 cyber samples
python experiments/03_layer_sweep_omar/extract.py

# 2) train per-layer probes for cyber_1 (cyber_2/cyber_3 currently no-op for this subset)
python experiments/03_layer_sweep_omar/train_probes.py

# 3) plot AUC vs layer
python experiments/03_layer_sweep_omar/plot.py
```

Outputs (all in this folder):
- `extracts/` — symlinked or written to `/home/ubuntu/extracts/03_layer_sweep_omar/`
- `results.json` — per-(task, layer) AUC + bootstrap CI
- `auc_vs_layer.png` — the headline plot

## Results

### Final run (balanced 999, 5-fold stratified CV, both poolings)

Extracted Gemma 4-31B-it residuals at 13 layers for the balanced cyber_1
selection (500 dual_use + 499 benign — `cyber_6469`, a 25K-char binary blob,
OOM'd on the 80 GB H100 even after the hook-based rewrite and was excluded).
Peak VRAM held at 58–62 GB across the run.

Logistic-regression probe (`sklearn.LogisticRegression`, `C=1.0`, `lbfgs`),
5-fold stratified CV (`shuffle=True, random_state=0`). Reported as
test mean ± std across folds, with min/max range. Train AUC reported alongside
to expose overfitting.

| Pooling | Best layer | Test AUC (mean ± σ) | Test acc | Train AUC |
|---|---:|---:|---:|---:|
| **mean-pool** | **40** | **0.983 ± 0.006** (range 0.976–0.989) | 0.928 ± 0.015 | 1.000 |
| last-token    | 35     | 0.975 ± 0.005 (range 0.970–0.981)     | 0.915 ± 0.013 | 1.000 |

See `auc_vs_layer.png` and `acc_vs_layer.png`.

### Earlier biased run (kept for the record)

Initial run used "first 1000 cyber_1-eligible train samples" (685 / 315) and a
single 80/20 stratified split. Reported AUC 0.932 at layer 40. **Discarded**
because the file ordering put all dual_use early, so the 1000 first rows hit
all 685 dual_use + only the first 315 benigns — the 1267 remaining benigns
in train were never seen. Replaced with the balanced + CV run above.

| Layer | Test AUC | 95% CI         | Acc   |
|------:|---------:|----------------|------:|
|  0    | 0.767    | [0.695, 0.834] | 0.720 |
|  5    | 0.794    | [0.725, 0.859] | 0.750 |
| 10    | 0.827    | [0.768, 0.883] | 0.755 |
| 15    | 0.845    | [0.787, 0.895] | 0.785 |
| 20    | 0.877    | [0.821, 0.921] | 0.795 |
| 25    | 0.900    | [0.854, 0.939] | 0.815 |
| 30    | 0.915    | [0.868, 0.953] | 0.865 |
| 35    | 0.913    | [0.869, 0.949] | 0.855 |
| **40**| **0.932**| [0.892, 0.963] | 0.870 |
| 45    | 0.902    | [0.857, 0.941] | 0.825 |
| 50    | 0.879    | [0.828, 0.925] | 0.815 |
| 55    | 0.874    | [0.824, 0.920] | 0.800 |
| 60    | 0.867    | [0.813, 0.918] | 0.815 |

See `auc_vs_layer.png` for the curve with CI band.

## The most important observation — why does layer 0 already get 0.95?

Layer 0 is **before any transformer computation** — raw token embeddings + positional
encodings, no attention, no MLP, no contextual processing. Yet mean-pool AUC ≈ 0.95.

**The answer: it's a bag-of-words classifier.**

Mean-pooling the embedding layer computes the average word embedding across the prompt.
That is equivalent to a very expensive bag-of-words. It works because dual_use and benign
draw from very different vocabulary distributions:

| benign vocabulary                          | dual_use vocabulary                       |
|--------------------------------------------|-------------------------------------------|
| "Raspberry Pi", "IoT", "class project"     | "vulnerability detection", "RBAC", "IDS"  |
| "bug reports", "accessibility", "sysadmin" | "Kubernetes", "CAN-bus", "secure-coding"  |

A linear probe on the mean embedding is essentially learning: *"does this prompt contain
security/exploitation vocabulary?"* — and that alone gets 0.95 AUC.

This is both **encouraging** and a **warning**:

- **Encouraging:** the signal is strong and easy to detect.
- **Warning:** the probe might be learning topic/vocabulary, not the model's deeper
  understanding of harm level. A prompt with security vocabulary but genuinely benign
  intent could get misclassified.

Notice that **last-token at layer 0 ≈ 0.50 (pure chance)**. That confirms the bag-of-words
reading: one token's embedding carries no information. It's the *aggregate* vocabulary
that's informative.

### What this means for strategy

1. The **~0.03 AUC gain from layers 10–50 over layer 0** is the interesting part — that's
   what the transformer adds beyond bag-of-words. Those layers build contextual
   understanding of intent vs. capability.
2. For **Level 2 attribution**, prefer a mid-layer probe (~30–40), not the embedding
   probe. Embedding-probe attribution would point to "words that sound security-related";
   mid-layer probe attribution points to "tokens that contextually signal dual-use intent".
   The latter is more causally relevant for crafting minimal edits that change behaviour
   rather than vocabulary.
3. **Validation in flight:** experiment 09 (`09_exp03_extra_omar`) D1 is running a pure
   TF-IDF + logistic regression baseline on raw prompt text right now. If it also gets
   ~0.95, the embedding probe isn't doing anything the model's internals don't already
   have for free. The **delta between TF-IDF and the mid-layer probe** is the real value
   added by Gemma's residual stream.

### Slide-worthy framing

> "Layer 0 mean-pool gets 0.95 AUC — the model barely needs to think to distinguish
> dual_use from benign. But the 0.03 AUC gain from deeper layers is where contextual
> understanding lives, and that's what matters for Level 2 attribution."

This shows we understand what the probe is actually learning, not just reporting a number.

## Takeaways

- **Best layer for cyber_1 = ~30–40** (~half-to-2/3 through Gemma's 60 blocks).
  Test AUC plateau ≈ 0.98 across that range with σ ≈ 0.005–0.01 across 5 folds —
  tight and consistent.
- **Yes, the probes overfit on train** (train AUC ≈ 1.000 at every non-trivial
  layer with 5376-d features and ~800 fit points). **But the test signal is real**:
  held-out folds still hit ~0.98 with very low variance. So the activations
  genuinely separate dual_use from benign — the probe just also happens to
  memorise the trainset on top of that.
- **Mean-pool ≥ last-token** at every layer here (both approach 0.98 at the peak,
  but mean-pool wins a narrow margin everywhere). Last-token at layer 0 is
  exactly chance (the final chat-template token is identical for every prompt).
- Layer 60 is the worst non-zero layer for both poolings (test AUC drops to
  0.93 mean / 0.90 last-token), echoing the "tuned lens" finding (Belrose 2023):
  the very last block commits to next-token specifics and becomes less
  informative for a coarse semantic classifier.
- The earlier "first 1000" biased subset gave 0.93 single-split AUC. Switching
  to balanced + CV moved this to **0.98 ± 0.006**. The bigger lesson is that
  a single split sat well inside the across-fold noise band — never report a
  single number without CV (or at minimum bootstrap on multiple splits).

**Engineering notes:**
- The original `extractor.ActivationExtractor` from exp 02 can't handle Gemma
  4-31B at >7k tokens because it requests `output_hidden_states=True`, which
  materialises all 61 hidden-state tensors at once. Switched to forward-hook
  extraction (only the 13 requested layers held on GPU at a time). Same .pt schema.
- One sample (`cyber_6469`, benign, 25K chars including a binary blob) OOM'd
  even with the hook-based extractor. Excluded from `selection.json`.
- `train_probes.py` writes `metrics.jsonl` (one line per fold) and
  rewrites `results.json` atomically after every layer, so a crash mid-run
  preserves work-so-far for inspection.

## Next

- `04_probe_heads_omar`: at the best layer (40), compare probe heads — mean,
  last-token, MultiMax (Kramar 2026), multi-layer concat (CC++ 2026), logit-lens
  trajectory (Belrose 2023), polynomial lift (Pleshkov 2026).
- `05_full_cyber_omar`: stratified re-extract so cyber_2 (vs high_risk_dual_use)
  and cyber_3 (vs prohibited) have positives. Apply best head to all three tasks.
- `06_refusal_omar`: same pipeline on Gemma + Qwen refusal probes.
