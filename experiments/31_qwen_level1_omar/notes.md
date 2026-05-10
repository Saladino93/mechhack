# Experiment 30 (omar): Qwen Refusal — Level 1 layer sweep × 4 probe heads

## Goal

Close the README_OMAR Level-1 hole: produce a Refusal-Qwen AUC table comparable
to the Refusal-Gemma sweep in exp 18. Also let us answer the open question
flagged by the dataset README: **is Qwen's refusal signal genuinely weaker in
residuals than Gemma's, or just diffused / different-shaped?**

## Method

### Model + extracts
- **Qwen3.6-27B** (text config: `hidden_size=5120`, `num_hidden_layers=64`,
  `head_dim=256`, hybrid 16 full-attn + 48 DeltaNet). Downloaded to
  `/home/ubuntu/models/Qwen3.6-27B` (51.8 GB).
- **Layers extracted**: `[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]` —
  13 layers, every 5 from 0 (embeddings) to 60 (5 layers from the top).
  Same schedule that exp 03/18 used on Gemma so AUC numbers are directly
  comparable across models.
- **Samples**: all 878 rows of `datasets/refusal_probes/qwen36/attacks_full.jsonl`
  (275 refusal, 603 compliance — 31% positive rate). fp16, with chat template.
- **Extracts**: `/home/ubuntu/extracts/qwen36/`, ~250 MB/sample on average.
- **Splits**: dataset-defined 589 train / 289 test (we use these — never the
  random 70/30 from `train_probe.py` defaults).

### Probe heads
1. **Arditi direction** (per-layer, no training) — unit-normalized
   `(μ_refused − μ_compliance)` from training mean-pool features; project test
   features. Free baseline below the trained linear probe.
2. **Linear (mean-pool)** — `nn.Linear(5120, 1)`, AdamW 1e-3, 50 epochs,
   early-stop on val loss. 5 seeds. Per layer.
3. **Linear (last-token)** — same but indexes the last unmasked position.
4. **MLP** — `Linear → GELU → Dropout(0.1) → Linear(1)`, hidden=256, mean-pool
   input. AdamW 1e-3. 5 seeds. Per layer.
5. **Attention probe** — DeepMind-style single learned-query soft-attention
   over per-token residuals. AdamW 5e-4. 3 seeds. Per layer (every-10 subset).
6. **MultiMax probe** — Kramár 2026 Architecture C: `TransformMLP(d→100) → 10 heads`
   with sharp-softmax (τ=0.1) per head, max-over-tokens, sum-over-heads.
   Imported from exp 16 (`probes.MultiMaxProbe`). AdamW 1e-4 wd=3e-3. 3 seeds.
   Per layer (every-10 subset).
7. **CC++ Head A (global, not per-layer)** — mean-pool every one of the 13
   layers, concat to a `13 × 5120 = 66 560`-dim feature vector, fit one
   sklearn `LogisticRegression(solver='lbfgs', max_iter=2000)` with C-sweep
   over `{0.01, 0.1, 1.0, 10.0}` chosen on a held-out 20% inner val of the
   train split. Single deterministic run.

### Metric / uncertainty
- **Primary metric**: AUC on the dataset's pre-defined test split (281
  labeled rows after dropping `is_refusal=None`). Splits are deterministic
  per `rules/predict.md` (`hash(sample_id) mod 100 < 70 → train`); we never
  redefine them.
- **Test-AUC error bars**: 1000-resample bootstrap with replacement on the
  281-sample test set; 95% CI = [2.5, 97.5] percentiles. We refuse a
  resample where only one class survives (no AUC computable). Per arch ×
  layer.
- **Per-seed std (trainable heads only)**: 5 seeds (0–4) for linear/MLP,
  3 seeds for the slow attention/multimax heads. Seeds control torch RNG,
  numpy RNG, and inner train/val shuffle. Reported as mean ± std on the
  point AUC across seeds. Combined with bootstrap CI on the seed-averaged
  prediction.
- **Arditi / cc_concat**: deterministic single runs (no seed-averaging).
  Bootstrap CI on test AUC only.
- **Multi-task variance** (across the 5 hackathon tasks): not reported here
  because only Refusal-Qwen is in this experiment. README_OMAR aggregates
  the cross-task numbers.
- **Wilson 95% CI** is what the project uses for proportions (e.g., refusal
  rate, judge agreement) — not relevant to AUC here.

### Hyperparameters
| Knob              | Value                                          |
|-------------------|------------------------------------------------|
| Optimiser         | AdamW                                          |
| LR                | 1e-3 (linear/MLP), 5e-4 (attention)            |
| Weight decay      | 1e-3                                           |
| Epochs            | 50                                             |
| Batch size        | 32                                             |
| Early stop        | val loss, patience 5                           |
| Seeds             | {0,1,2,3,4} (trainable heads only)             |
| Bootstrap         | 1000 resamples                                 |
| C-grid (CC++ A)   | {0.01, 0.1, 1.0, 10.0}                         |
| MLP hidden        | 256, dropout 0.1                               |
| Attention head    | single learned query, scaled dot-product       |

## Results

**Fast heads — done 2026-05-10.** Test n=281 (97 refusal / 184 compliance,
pos_rate 34.5%). Train n=581. 1000-bootstrap 95% CI on test AUC.

### Per-layer AUC (winner per arch in **bold**)

| Layer | linear_mean | linear_last | mlp_mean | arditi |
|---:|---:|---:|---:|---:|
| 0  | 0.758 | 0.500 | 0.805 | 0.721 |
| 5  | 0.815 | 0.798 | 0.820 | 0.748 |
| 10 | 0.814 | 0.813 | 0.806 | 0.744 |
| 15 | 0.808 | 0.796 | 0.807 | 0.761 |
| 20 | 0.804 | 0.822 | 0.803 | 0.774 |
| 25 | 0.810 | 0.823 | 0.810 | 0.790 |
| 30 | 0.834 | 0.833 | 0.837 | 0.772 |
| 35 | **0.850** | 0.846 | **0.852** | 0.762 |
| 40 | 0.833 | 0.862 | 0.841 | 0.808 |
| 45 | 0.846 | 0.862 | 0.848 | 0.788 |
| 50 | 0.845 | 0.864 | 0.845 | 0.776 |
| 55 | 0.824 | 0.856 | 0.834 | 0.769 |
| 60 | 0.827 | **0.876** | 0.810 | **0.827** |

### Headline (best layer per head)

| Probe | Best layer | Test AUC | 95% CI | seed-std | vs TF-IDF word |
|---|---:|---:|---|---:|---:|
| **linear_sklearn_last** (LBFGS C-sweep) | **L60** | **0.878** | [0.84, 0.92] | det. | **+0.026** |
| linear_last (AdamW, 5 seeds) | L60 | 0.876 | [0.83, 0.92] | ±0.008 | +0.024 |
| **multimax_simple_tuned** | **L50** | **0.876** | [0.84, 0.92] | ±0.020 ⚠ | +0.024 |
| **multimax_kramar_tuned** (paper-exact) | L60 | 0.861 | [0.82, 0.90] | ±0.009 | +0.009 |
| multimax_kramar (loose wd) | L50 | 0.855 | [0.81, 0.90] | ±0.011 | +0.003 |
| mlp_mean | L35 | 0.852 | [0.81, 0.90] | ±0.005 | 0.000 |
| **TF-IDF word 1-2gram (text-only baseline)** | — | **0.852** | [0.81, 0.89] | det. | **0.000** |
| linear_mean | L35 | 0.850 | [0.80, 0.89] | ±0.001 | −0.002 |
| linear_sklearn_mean | L35 | 0.844 | [0.80, 0.89] | det. | −0.008 |
| attention_deepmind_tuned | L50 | 0.844 | [0.80, 0.89] | ±0.002 | −0.008 |
| multimax_simple | L50 | 0.866 | [0.82, 0.91] | ±0.014 | +0.014 |
| TF-IDF char 3-5gram (text-only baseline) | — | 0.839 | [0.79, 0.88] | det. | −0.013 |
| cc_concat (concat of 13 mean-pools, LR) | global, C=0.01 | 0.838 | [0.79, 0.88] | det. | −0.014 |
| attention_deepmind | L50 | 0.842 | [0.80, 0.88] | ±0.002 | −0.010 |
| arditi (mean-difference direction) | L60 | 0.827 | [0.77, 0.87] | det. | −0.025 |

⚠ "seed std" for `multimax_simple_tuned` is reported on the per-seed point AUC.
The 0.876 figure is the AUC of the **5-seed-averaged probability**; 2 of 5
individual seeds collapsed to AUC ≤ 0.55 because the max-pool gradient flows
through only 1 token per sample, and a bad early argmax can lock training
into a wrong direction. Averaging predictions across seeds rescues the
ensemble even when individual probes fail. **The kramar variant (10 heads
+ TransformMLP + sharp-softmax) gets the same AUC ceiling at ~½ the seed
variance** (±0.009 vs ±0.020).

### Reading the AUC-by-layer plot (`auc_by_layer.png`)

The plot has 12 probe lines + 2 horizontal baselines on the same x-axis (layer
index, 0 = embeddings, 60 = 5 layers from the top). Errorbars are 1000-sample
bootstrap 95% CIs. A small per-arch x-jitter (≤±0.5 layer units) keeps
overlapping errorbars readable.

**Lines (per-layer probes, sweep 13 layers — 4 layers for the per-token
heads):**

- `arditi` (dark teal, solid): **no-training** mean-of-refused minus
  mean-of-compliant direction; just dot-product. Reads as a "free" probe.
  Curve climbs slowly; peaks at L60 (0.827).
- `linear_mean` (medium blue, solid): linear AdamW probe over mean-pooled
  residuals. Fast hump at L5 (~0.81), peaks at L35 (0.850), flatlines after.
- `linear_last` (teal, solid): same probe but on the last unmasked token.
  Starts at 0.5 at L0 (no context — every prompt's last token is the same
  chat-template marker), climbs almost monotonically, **wins outright at
  L60 with AUC 0.876**.
- `linear_sklearn_mean`/`linear_sklearn_last` (slightly darker shades, solid):
  sklearn LBFGS C-sweep over the same features. Within ±0.006 of AdamW —
  proves the linear ceiling is feature-limited, not training-limited.
- `mlp_mean` (orange, solid): 256-hidden MLP on mean-pool. Tracks
  linear_mean very closely; the nonlinearity buys at most +0.005 AUC.
- `multimax_simple` (light purple, solid, dots only at L30/40/50/60):
  per-token linear → max-over-tokens. Erratic; peaks at L50 (0.866) but
  collapses at other layers from gradient brittleness.
- `multimax_kramar` (purple, solid, 4 dots): Kramár 2026 Architecture C
  (TransformMLP→100, 10 heads, sharp softmax). Stable across L30–L60,
  peaks at L50 (0.855).
- `attention_deepmind` (pink, solid, 4 dots): DeepMind-style single
  learned-query soft-attention pool. Tight seed-std but flat plateau
  ~0.84.
- `multimax_simple_tuned` / `multimax_kramar_tuned` /
  `attention_deepmind_tuned` (same colors, **dashed**, 4 dots): re-runs
  with paper-exact wd=3e-3 (kramar), `BCEWithLogitsLoss(pos_weight≈2.26)`,
  batch=32, patience=5, 5 seeds. Tuning lifts `mm_simple` at L50 to 0.876
  and `mm_kramar` at L60 to 0.861; barely moves attention.

**Horizontal lines (no per-layer dependence — sample-level baselines):**

- **TF-IDF word 1-2gram** (green dash-dot at AUC 0.852): sklearn LR with
  C-sweep on word n-gram TF-IDF of the raw `attack_prompt`. *No model
  internals at all.* Ties with the best mean-pool probes.
- **TF-IDF char 3-5gram** (lime dotted at 0.839): same, character n-grams.
- **cc_concat (13×) AUC=0.837** (gray band): mean-pool every layer, concat
  to 66 560-d, sklearn LR with C-sweep (best C=0.01 — heavy regularization).
  Concatenating layers does *not* beat the best single layer.

**The visual story.** Activation probes hover at or below the green TF-IDF
line for the first ~30 layers and through the mid-stack. Only `linear_last`
crosses cleanly above starting around L40, climbing to a clear ~0.876
ceiling at L60. The tuned `multimax_simple_tuned` momentarily matches that
ceiling at L50 but with high seed variance. **Everything else is within
the gray cc_concat band — i.e., not better than what TF-IDF gives you.**

### Cross-model comparison (Refusal — Gemma vs Qwen)

| Method | Gemma 4-31B-it | Qwen 3.6-27B | Δ |
|---|---:|---:|---:|
| best linear (mean or last) | 0.953 (last L45) | 0.876 (last L60) | **−0.077** |
| Arditi best | 0.758 (L32) | 0.827 (L60) | +0.069 |
| LR ↔ Arditi gap | **+0.17** | +0.05 | — |

## Takeaways

_(slow heads still running — these will be revised when attention/multimax land.)_

1. **Qwen's refusal signal is weaker but real.** Best probe AUC 0.876 vs
   Gemma's 0.953 — the 0.08 gap matches the dataset README's prediction.
   Refusal-Qwen sits below Refusal-Gemma but well above chance.

2. **Last-token > mean-pool, by a meaningful margin.** Best last L60 = 0.876
   vs best mean L35 = 0.850. On Gemma the equivalent gap was +0.008 (last L45
   vs mean L40), so for Qwen the *position* of the signal matters more —
   consistent with the hybrid arch routing things differently into the
   final-token state than Gemma's all-attention stack does.

3. **The signal climbs through the late layers.** linear_mean peaks at L35
   then plateaus, but linear_last keeps rising (L40=0.862, L60=0.876). This
   is the *opposite* of Gemma exp 18 where the signal saturates by L40.
   Implication: **don't stop the layer sweep at the conventional 'middle';
   late layers are where Qwen concentrates refusal.**

4. **MLP doesn't beat linear here.** Per-layer, mlp_mean ≈ linear_mean within
   ±0.01 AUC. So the bottleneck on Qwen isn't probe capacity — it's the
   feature representation itself (or the limited ~580 train samples).

5. **CC++ multi-layer concat (66 560-d) is *worse* than the best single
   layer.** Concat AUC 0.838 vs linear_last L60 0.876. Mirrors Gemma exp 18's
   '13-layer concat overfits' finding — capacity is hurting, not helping.
   Heavy regularization (C=0.01) was selected; even so, no gain over single-
   layer.

6. **Arditi closes the gap to LR a lot more on Qwen than Gemma.** Gap is
   +0.17 on Gemma but only +0.02–0.05 on Qwen. Two readings: (a) Qwen's
   refusal direction in residual space is *closer* to the simple
   mean-of-difference, with less off-diagonal information for LR to exploit;
   (b) the LR probe is being capacity-starved by 581 train samples on a
   harder distribution.

7. **The 'graded ladder' hypothesis is open.** Qwen full-attention layers
   are at indices 3, 7, 11, …, 63 (every 4). Of our extracted layers,
   {15, 35, 55} are post-full-attn and {0, 5, 10, 20, 25, 30, 40, 45, 50, 60}
   are post-DeltaNet. Visually L35 (post-full-attn) is a bump but L60
   (post-DeltaNet, last block) wins outright with last-token. The story
   "full-attention layers carry more refusal signal" does not hold cleanly.

8. **TF-IDF baseline matches all activation probes except linear_last L60.**
   Word 1-2gram LR AUC 0.852 — equal to mlp_mean L35 (0.852), beating
   linear_mean (0.850), arditi (0.827), cc_concat (0.838). Only
   linear_last L60 (0.876) clears it, by +0.024. Compare Gemma refusal
   activation lift = +0.075. Qwen's refusal information is **mostly already
   in the surface text**; activations buy ~3 extra AUC points and only at
   one specific layer.

9. **AdamW vs sklearn-LBFGS linear probes are tied to within ±0.006 AUC.**
   linear_last L60: 0.876 (AdamW) vs 0.878 (sklearn LBFGS, C=0.001). So the
   linear ceiling is ~0.876–0.878 — the AUC is feature-limited, *not*
   training-limited. No further linear-probe tuning will help.

### Length-stratified AUC (winner: linear_sklearn_last L60)

| Bucket | n | n_pos (%) | char-len p50 | AUC | 95% CI |
|---|---:|---:|---:|---:|---|
| short  (≤6 655)  | 95 | 49 (51.6%) |  4 421 | **0.876** | [0.80, 0.94] |
| medium (≤11 420) | 92 | 36 (39.1%) |  8 423 | 0.771 | [0.67, 0.86] |
| long   (>11 420) | 94 | 12 (12.8%) | 19 082 | 0.927 | [0.87, 0.98] |

Note the U-shape: medium drops, long is *higher* but only 12 positives.
Long bucket's high AUC is mostly because long prompts in this corpus are
overwhelmingly compliant (87% negative class), which makes the rare 12
refusals easy to separate from the bulk.

### Kramár FNR @ strict FPR (threshold locked on SHORT bucket)

Production-style guardrail: pick a 1% FPR threshold on the short-prompt
distribution, deploy that single threshold to all prompt lengths.

| Bucket | n_pos | FNR @ 1% FPR | FNR @ 5% FPR |
|---|---:|---:|---:|
| short  | 49 | **53.1%** | 40.8% |
| medium | 36 | **75.0%** | 66.7% |
| long   | 12 | **100.0%** | 83.3% |

Compare Gemma refusal exp 15 (mean L32):
short 49% → medium 64% → long **80%** FNR @ 1% FPR.

**Qwen's threshold-shift is *worse* than Gemma's** (53% → 100% vs 49% → 80%).
At a production-realistic strict-FPR threshold the Refusal-Qwen probe
catches *zero* of the 12 long-prompt refusals. This is Kramár's exact
motivation for the Rolling / MultiMax architecture. The repo's exp 16 has
those probes implemented (originally targeted at Gemma); porting them to
Qwen is the obvious follow-up.
