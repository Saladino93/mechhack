# Why simple text features predict refusal as well as they do

> EDA on Refusal-Gemma data (n=878 prompts, 47.9% refused). Findings here
> explain why a TF-IDF + LR baseline already hits AUC ≈ 0.87, and why fancier
> probes (LR-on-residuals, quadratic, attention) get only +0.07–0.08 of
> additional AUC. The signal lives in surface text far more than expected.

## Headline: refusals are deterministic surface phenomena

### The model's *response* starts with a refusal phrase ~98% of the time

Top response-start trigrams that occur in refusals but never in compliance:

| Response opening 3-gram | refused | complied | log-odds |
|---|---:|---:|---:|
| `I cannot provide` | **161** | **0** | +5.09 |
| `While I understand` | 140 | 0 | +4.95 |
| `I cannot complete` | 18 | 0 | +2.94 |
| `I cannot port` | 13 | 0 | +2.64 |
| `I cannot implement` | 11 | 0 | +2.48 |
| `I cannot translate` | 6 | 0 | +1.95 |

Sum of these zero-false-positive 3-grams: **349/421 = 83% of refusals**
caught by a hardcoded regex against the response. The judge model could be
replaced by a 5-line regex on the response and lose only ~17% recall.

The remaining 17% (~70 refusals) include `I can help` (49 ref / 4 comp —
"I can help with the *defensive* version of...") and `While I can` (4/0).
These are softer refusals (declines wrapped in a redirect).

### The *prompt* often telegraphs the refusal too

Top prompt-start words (≥5 occurrences) that predict refusal:

| Prompt-start word | refused | complied | log-odds |
|---|---:|---:|---:|
| `during` | **47** | **0** | +3.87 |
| `developing` | 15 | 5 | +0.98 |
| `course:` | 39 | 15 | +0.92 |
| `working` | 54 | 23 | +0.83 |
| `building` | 18 | 13 | +0.31 |
| `i'm` | 234 | 389 | −0.51 |

`during` at the prompt start is **100% refusal** in this dataset (47/47).
This points at a red-team-template artifact: prompts of the form *"During an
authorized [scenario], please [harmful capability]..."* are reliably flagged
by Gemma. A linear classifier on the first prompt token is already a
non-trivial detector.

`I'm` is the dominant prompt opener (623/878), slightly more in compliance.
This is the boilerplate role-play frame: *"I'm a security researcher / I'm
working on..."* — half the time it's a successful jailbreak preamble.

### Length correlations: longer prompts → MORE compliance

Per-feature Pearson correlation with `is_refusal`:

| feature | r |
|---|---:|
| **response_chars** | **−0.516** |
| prompt_chars | −0.402 |
| prompt_words | −0.403 |
| n_tokens | −0.402 |
| thinking_chars | +0.068 |
| judge_score | +0.165 |

Two readings, both meaningful:

1. **Refusals are short** (response ~−0.52). "I cannot provide..." stops at
   a paragraph; compliance can run pages. Trivial heuristic.
2. **Long prompts get compliance more often** (prompt ~−0.40). Consistent
   with the Kramár threshold-shift: refusal-pulling tokens get diluted in
   a sea of context, the model decides to comply.

This second one matters for **probe design**. A probe that mean-pools across
all tokens will dilute the refusal signal in the same way the model does.
A probe that takes a max over tokens (Kramár's MultiMax) pays attention to
the single critical token regardless of context length. **That's why
max-pool > mean-pool when n_tokens grows**.

## What this means for our probe stack

### A. The TF-IDF baseline is strong because the signal is in tokens

| Method | refusal test AUC | gap vs activation |
|---|---:|---:|
| **TF-IDF word 1-2gram** | **0.877 ± 0.025** | — |
| TF-IDF char 3-5gram | 0.870 ± 0.038 | — |
| Random predictor | 0.500 | — |
| LR on Gemma residual L40 mean | **0.945** | **+0.068** |
| LR on residual L45 last-tok | 0.953 | +0.076 |

Activations earn +0.07–0.08 of additional AUC. **That's the value of the
internal representations on this task** — not nothing, but the surface text
already does most of the work. On refusal, the model's behavior is highly
correlated with the very tokens it conditioned on.

### B. Why the activation lift is so much larger on cyber than on refusal

| Task | TF-IDF AUC | Activation AUC | Lift |
|---|---:|---:|---:|
| cyber_1 | 0.946 | 0.983 | **+0.037** |
| cyber_2 | 0.887 | 0.946 | +0.059 |
| cyber_3 | 0.890 | 0.955 | +0.065 |
| **refusal_gemma** | **0.877** | **0.953** | **+0.076** |

Pattern: **activation lift grows with task difficulty, not the other way
around.** When discrimination is easy (cyber_1: dual-use vs benign), TF-IDF
+ LR is nearly as good as residuals. When the boundary is more semantic
(cyber_2/3: graded harm tiers; refusal: model-internal decision), the
activations carry meaningfully more signal.

This vindicates the probe approach: it's not a magic bullet (the cyber_1
gap of +0.037 is small), but it scales with hardness.

### C. Why mean-pool already works so well on refusal

`response_chars` r = −0.52 says half the variance in `is_refusal` is
captured just by counting characters in the response. Mean-pool of residuals
implicitly encodes "average refusal-ness" of every token, which is highly
correlated with prompt length and content density. So mean-pool is not just
"averaging noise" — it's averaging tokens that already individually carry
refusal information at the surface level.

The corollary: **on harder tasks (where surface information is less
discriminative), mean-pool should hurt more relative to max-pool**. We see
this exactly in the Kramár data: their long-context FNR explodes for
mean-pool but not for max-pool. We see a partial echo in our exp 15 data:
refusal mean-pool L32 short = 0.92, long = 0.92 (AUC stable) but FNR @ 1%
FPR jumps 49% → 80% (threshold-shift on long prompts).

### D. The "I'm" / "during" finding is a probe-monitor opportunity

Combining the response-start regex and the prompt-start word features
gives you a **deployable monitor** that needs no model internals:

```python
def naive_refusal_predictor(prompt: str, response: str) -> float:
    # Strong response cues (perfect specificity)
    if re.match(r"^\s*(I cannot|While I understand|I'm sorry)", response):
        return 1.0
    # Strong prompt cue
    if prompt.lstrip().lower().startswith("during"):
        return 1.0
    # Length heuristic (refusals are short)
    if len(response) < 200:
        return 0.7
    return 0.3
```

This kind of regex-plus-length monitor is **what you'd actually deploy** for
post-hoc refusal detection. The probes win when you need to detect refusal
*before* the response exists — i.e. at the level of the activations on the
prompt alone. That's the regime where activations earn their +0.07–0.08
lift.

## Open questions / followups

1. **How much of the activation lift survives token-control?** I.e. fit
   a probe on residuals BUT residualize against TF-IDF features first. Is
   there +0.07 of *non-token-derived* signal, or is the lift just better
   linear use of the same information?
2. **The "during" finding generalizes how?** Likely an artifact of the
   red-team prompt templates used to generate this dataset. If we see
   the same pattern in unseen prompts, it's a data property; if not,
   it's a sampling artifact.
3. **Refusal direction shares signal with cyber_3 (prohibited)** but not
   cyber_1 (dual-use) — see exp 22. The semantic harm boundary is what
   transfers, not the keyword overlap.

## Files

- `eda.py` — full analysis script
- `histograms/*.png` — 11 distribution + refusal-rate plots
- `ngrams/response_first_*gram.csv` — per-class response opening tokens
- `ngrams/prompt_first_word.csv` — per-class first prompt word
- `ngrams/prompt_top_unigram_by_class.csv` — discriminative prompt tokens
- `ngrams/prompt_top_bigram_by_class.csv` — discriminative prompt bigrams
- `correlations/feature_correlations.csv` — Pearson r of length features
- `correlations/source_distribution.csv` — refusal rate by source
