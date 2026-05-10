# Qwen / Gemma cross-model story — figures

## `qwen_gemma_refusal_auc_by_layer.png` — side-by-side Level-1

Gemma 4-31B-it (left, 60 layers) and Qwen 3.6-27B (right, 64 layers) on the
**same 878 refusal prompts**. Same probe heads, same protocol.

| Probe | Gemma best | Qwen best | Δ (Gemma − Qwen) |
|---|---:|---:|---:|
| linear_last | **L40 = 0.952** | **L60 = 0.876** | **+0.076** |
| linear_mean | L40 = 0.919 | L35 = 0.850 | +0.069 |
| mlp_mean    | L40 = 0.920 | L35 = 0.852 | +0.068 |
| arditi      | L55 = 0.828 | L60 = 0.827 | +0.001 |
| cc_concat   | C=0.1, 0.920 | C=0.01, 0.838 | +0.082 |
| **TF-IDF word baseline** | _not run_ | 0.852 | — |

**Two stories on one slide:**
1. **Gemma's refusal signal is materially stronger (+0.07–0.08 AUC).** It also peaks earlier (L40 vs L60). Linear_last L40 cleanly clears the cc_concat band; on Qwen, only linear_last L60 squeaks above the TF-IDF line.
2. **Arditi is essentially equal across the two models** (0.828 vs 0.827). The directional signal is the same; the LR probe pulls more juice out of Gemma than out of Qwen.



Selected plots from this session's Refusal-Qwen Level-1 sweep + cross-model
direction agreement.

## `cross_model_direction.png` — **the headline result**

Two panels.

**Left:** per-prompt scatter at L40. For each of 281 test prompts, x = Gemma's
Arditi-projection score, y = Qwen's. Points coloured by joint label
(both refused, both complied, only Gemma refused, only Qwen refused).
The two models score the same prompts almost identically along their
respective refusal directions: cosine 0.877 of the score vectors,
Pearson 0.860, sample size 281.

**Right:** four curves vs layer (0 → 60).
- **Cosine of the per-prompt score vectors** (blue line). Stays at **+0.99–1.00 from L10 to L55** — the two models' refusal-projection rankings of the same 878 prompts match almost perfectly through almost the entire stack. Drops at L00 (embedding) and L60 (last block).
- **Pearson r** (purple). Same picture, slightly looser, peaks ~0.95.
- **AUC: Gemma direction → Qwen labels** (red dashed). 0.69–0.79, peaking around L40–L55. Compare to Qwen's own direction reaching 0.83 — Gemma's direction transfers at ~93% of Qwen's own performance.
- **AUC: Qwen direction → Gemma labels** (orange dashed). 0.72–0.83, mirror image.

**Reading:** the "refusal direction" is essentially **the same direction in two completely different model families** (Gemma 4-31B, 60-layer full-attn vs Qwen 3.6-27B, 64-layer hybrid DeltaNet) once you measure it in mid-stack. The L60 dip is consistent with model-specific final-layer specialization.

## `qwen_refusal_auc_by_layer.png` — Refusal-Qwen Level-1 sweep

12 probe heads × 13 layers. Errorbars = 1000-bootstrap 95% CI.

**Per-layer probes** (lines): linear_mean, linear_last, linear_sklearn_*, mlp_mean, arditi, plus the per-token heads (multimax_simple/kramar, attention_deepmind, untuned + tuned variants — short 4-point lines at L30/40/50/60).

**Horizontal baselines**:
- TF-IDF word 1-2gram LR (green dash-dot, AUC 0.852) — text-only floor.
- TF-IDF char 3-5gram LR (lime dotted, AUC 0.839).
- cc_concat (gray band, AUC 0.838) — sklearn LR on concat of 13 mean-pools.

**Headline:** only `linear_last` clears the TF-IDF baseline cleanly, peaking at L60 with AUC **0.876** (sklearn-LBFGS variant: 0.878). Activation lift over TF-IDF is **+0.024** — much smaller than Gemma's +0.075. Qwen's refusal signal sits very close to surface text features.

## `qwen_refusal_length_strat_auc.png` — AUC by prompt-length tertile

Same 12 probes scored separately on short (≤6 655 chars), medium, long
(>11 420) test buckets. Top-6 probes shown.

**Pattern:** AUC dips on the medium bucket (~0.77 for the L60 winner) and
recovers on long (~0.93 — but only 12 positives, dominated by easy
discrimination since long prompts are 87% non-refusals).

## `qwen_refusal_kramar_fnr.png` — Kramár-style FNR @ 1% FPR

Threshold *locked on the short bucket* (the production setup: pick FPR
target on a calibration set, deploy a single threshold). Then evaluate FNR
per length bucket.

**Headline:** the L60 winner goes 53% → 75% → **100% FNR** as prompts get
longer. Gemma's same metric (exp 15) was 49% → 64% → **80%**. **Qwen is
worse than Gemma on the production-deployable threshold-shift problem** —
direct motivation for Kramár Rolling/MultiMax probes.
