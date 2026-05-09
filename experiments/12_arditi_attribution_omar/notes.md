# Experiment 12 (omar): Per-token Arditi attribution

## Goal

For each of the 81 prompts in `datasets/refusal_probes/gemma4_31b/attribution_eval.jsonl`,
compute per-token contribution to the refusal direction:
`contribution[i] = residual[i] @ arditi_direction`. The high-scoring tokens
become the `[POS|SCORE]token` markup the editor LLM in
`starter_code/iterative_edit_agent.py` expects, and they tell us where in the
prompt the refusal signal concentrates.

## Method

- Reuses `arditi_direction.npy` from exp 11 (unit-norm refusal direction at
  L32 mean-pool features).
- For every `.pt` in the 81-sample eval set: load fp16 residuals (single
  layer 32, shape `(1, n_tokens, 5376)`), squeeze, dot every token with the
  Arditi direction → 1 score per token.
- Token strings come from `tokenizer.decode([id])` on the saved `input_ids`
  (Gemma tokenizer, trust_remote_code).
- Top-K (K=8) tokens per prompt by absolute score; build a
  `[POS|SCORE]token` marked prompt for downstream PRE consumption.

## Headline

| Statistic | Value |
|---|---:|
| samples processed | 81 / 81 (0 skipped) |
| max per-prompt attribution (mean across prompts) | **+142.3** |
| min per-prompt attribution (mean across prompts) | −4.80 |
| mean absolute attribution (across all attended tokens) | 106.4 |

The asymmetry (max ≈ +142 vs min ≈ −4.8) is striking and a methodology note:
the −4.8 floor is essentially constant across all 80/81 prompts → it's a
single boilerplate token (BOS / chat-template start) that always gets the
same residual. Not a meaningful "compliance pull". Real compliance-side
attribution from the Arditi direction is essentially zero in practice on this
eval set — the direction is one-sided.

## Top recurring tokens (across the 81 prompts, top-K=8 per prompt)

| Token | # prompts where it's top-8 | Mean Arditi score |
|---|---:|---:|
| ` detection` | 36 (44%) | +143 |
| ` isolated` | 29 (36%) | +144 |
| ` test` | 26 (32%) | +143 |
| ` environment` | 15 | +146 |
| ` realistic` | 13 | +143 |
| ` forensic` | 11 | +142 |
| ` legitimate` | 8 | +143 |
| ` authorized` | 7 | +139 |
| ` controlled` | 6 | +143 |
| ` simulated` | 6 | +142 |
| ` educational` | 6 | +141 |

These are exactly the **safety-washing / sandbox-framing** vocabulary the
attacks use to disguise harmful asks. *Zero* concrete harmful keywords
("keylogger", "ransomware", "exploit", "malware", "phish", "backdoor",
"rootkit", "trojan", "C2", …) appear in any prompt's top-8 across all 81
prompts.

## Spatial pattern

`per_prompt_attribution_strip.png`: the top-8 tokens per prompt cluster
*tightly* in one region (sparse outside), and there's a clear **vertical
band of hot tokens at the very end of every prompt** (positions 0.95-1.0).
Probably the chat-template / generation-marker tokens carrying the model's
"decision-to-refuse" right before generation, OR the actual ask landing at
the prompt's tail after framing. Either way, end-of-prompt edits look like
the cheapest-leverage rewrites.

## Implications

1. **The Arditi direction is detecting safety-washing vocabulary, not harmful
   content.** On the safety-washed-attack benchmark we have, this is exactly
   what jailbreak-pattern detection looks like.
2. **Caveat on the strong claim**: the direction was *trained* on refused vs.
   complied examples, and refused training data is heavy with safety-washed
   attacks. So of course the direction picks up safety-washing — that's its
   training distribution. We can't conclude "Gemma's refusal is fundamentally
   surface-level" without a control set of plain harmful asks. We *can*
   conclude "rewrites that strip safety-washing should flip the probe" —
   verified empirically by exp 13.
3. **One-sided direction in practice.** Attribution scores are essentially
   non-negative on this eval (apart from the constant template-token floor).
   Future work: re-derive the direction on a more balanced refused/complied
   training set and check whether a real compliance side appears.

## Outputs

- `attributions.jsonl` — per-prompt: `top_tokens` (8 entries with position,
  score, token), `marked_prompt` (with `[POS|SCORE]token` markers), score
  range, mean abs.
- `top_tokens.tsv` — flat 648-row TSV (sample_id, position, score, token).
- `summary.json` — aggregate stats.
- Plots: `per_prompt_attribution_strip.png`, `score_distribution.png`,
  `top_tokens_overall.png`, `rank_vs_score.png`.

## Cosmetic note

The strip plot was originally labelled with an x-axis claiming "normalised
[0,1]" but actually displaying bucket index 0-200; that was fixed and the
PNG regenerated. The current `per_prompt_attribution_strip.png` correctly
shows token position normalised to [0, 1].
