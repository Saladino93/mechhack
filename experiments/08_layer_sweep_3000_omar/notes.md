# Experiment 08 (omar): Layer Sweep on the Full 3190-Sample cyber_1 Pool

## Goal
Re-run the experiment 03 layer sweep at scale to test whether its 0.983 ± 0.006
test AUC is genuinely the activation signal or partly an artifact of the small
(N=999, 50/50 balanced) subset and inside-train-fold evaluation.

Concretely: does the held-out **test-split** AUC (the project's official
benchmark) match exp 03's CV AUC, or does it materially regress?

## Method

- **Model:** Gemma 4-31B-it (`/home/ubuntu/models/Gemma-4-31B-it`)
- **Sample selection:** **all 3190 cyber_1-eligible samples** (dual_use vs
  benign) across both `train.jsonl` and `test.jsonl`. No balancing — class
  ratio is the natural ~30/70 split (952 dual_use vs 2238 benign).
  - train: 685 dual_use + 1582 benign = 2267
  - test:  267 dual_use +  656 benign =  923
  - Sorted by `sample_id` for reproducibility.
- **Layers:** `[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]` — same as exp 03.
- **Pooling:** `mean` (over masked tokens) and `last_token`.
- **Probe:** `sklearn.linear_model.LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')`,
  single seed (`random_state=0`), no probe-hyperparam sweep.
- **Two evaluation regimes per (pooling, layer):**
  1. **5-fold stratified CV** on the 2267 train samples — direct comparison
     point with exp 03's CV.
  2. **Held-out test eval:** fit on all 2267 train samples, score on all 923
     test samples. Bootstrap (1000-resample) 95% CI on the test AUC.
- **Extraction:** hook-based residual extractor (`chunked_sdpa_scope` from
  `starter_code/chunked_sdpa.py`), fp16 storage, identical schema to exp 03.
  Wrapped in try/except so OOMs are logged (no .pt written) and the run
  continues.
- **Reuse strategy:** all 1406 cyber_1 extracts already on disk in
  `/home/ubuntu/extracts/cyber_all_omar/` (symlinked from
  `/home/ubuntu/extracts/03_layer_sweep_omar/` by the previous agent) are
  re-used as-is. Only the missing 1784 samples were freshly extracted.

## Data

- **Selection file:** `selection.json` — 3190 entries, each
  `{sample_id, label, split}`.
- **Extracts dir:** `/home/ubuntu/extracts/cyber_all_omar/` (shared across
  exps 06/07/08). Output schema per `.pt`:
  - `residuals` `(13, N_tok, 5376)` fp16
  - `input_ids` int32, `attention_mask` bool
  - `n_tokens, fwd_seconds, peak_vram_gb, label, sample_id, layer_idxs`

## Results

_(Filled in after `train_probes.py` completes — see `results.json`)_

### Headline table

| Pooling | Best layer (CV) | CV AUC ±σ | Best layer (test) | Held-out test AUC | Test 95% CI | exp 03 CV AUC |
|---|---:|---:|---:|---:|---:|---:|
| mean       | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | 0.983 ± 0.006 |
| last_token | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | 0.975 ± 0.005 |

See `auc_vs_layer.png`, `acc_vs_layer.png`, `metrics_vs_layer.png`.

### Extraction stats
- New extractions this run: _TBD_
- Reused (symlinked) from exp 03: 1406
- OOM-excluded: _TBD_  (sample_ids: _TBD_)
- Wall time (extract): _TBD_  Peak VRAM: _TBD_ GB
- Wall time (probe):   _TBD_

## Takeaways

_(Filled in after results)_

## How to run
```bash
python experiments/08_layer_sweep_3000_omar/select.py
python experiments/08_layer_sweep_3000_omar/extract.py    # GPU
CUDA_VISIBLE_DEVICES="" python experiments/08_layer_sweep_3000_omar/train_probes.py
python experiments/08_layer_sweep_3000_omar/plot.py
```
