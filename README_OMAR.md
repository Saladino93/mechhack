# README_OMAR — Omar's experiments + headline findings

> One-stop summary of every `*_omar/` experiment in this repo, the headline
> numbers from each, and where to look for plots / scripts. Use this as the
> reading guide for the slides.

Status as of **2026-05-09 16:00 UTC** (Saturday). Submission deadline Sunday
14:00 UTC. Some long-running jobs (cyber length-strat, cross-task, MultiMax
implementation) were still in flight when this file was last updated — see the
"in flight" markers below.

---

## Headline (slide 1 candidate)

### Level 1 — Probe AUC across the 5 tasks (4 of 5 done; Qwen skipped)

| Task | Definition | AUC (mean-pool) | AUC (last-token) | Best layer | Source |
|---|---|---:|---:|---:|---|
| Cyber Probe-1 | dual_use vs benign | **0.983** (1k CV) → **0.988** (3k held-out) | 0.97 | mean L40 | exp 03 / exp 08 |
| Cyber Probe-2 | high_risk_dual_use vs (du ∪ ben) | 0.946 | **0.955** | mean L40 / last L60 | exp 06 |
| Cyber Probe-3 | prohibited vs (hrdu ∪ du ∪ ben) | **0.955** | 0.92 | mean L35 | exp 07 |
| Refusal-Gemma | refusal vs compliance | **0.927** | 0.917 | L32 (only one extracted) | exp 11 |
| Refusal-Qwen | — | — | — | — | **skipped** (model not downloaded) |

### Level 2 — Behaviour-flip results, n=81 attribution_eval prompts, Wilson 95% CIs

Pipeline: per-token Arditi attribution (exp 12) → user-supplied k=7 rewrites
(`rewrites_k7.json`) → score each rewrite with both Arditi projection AND a
fully-fitted LR probe (exp 13) → roll Gemma on the LR-best rewrite → judge with
DeepSeek V4 Pro using the canonical `REFUSAL_JUDGE_SYSTEM`.

| Metric | k/n | Point | 95% CI |
|---|---:|---:|---|
| **Pr(f flipped \| edit)** (cheap proxy) | 29/81 | **0.358** | [0.262, 0.467] |
| **Pr(model flipped \| edit)** *(README headline)* | 22/71 | **0.310** | [0.214, 0.425] |
| Pr(model flipped \| f flipped) (concordance) | 9/29 | 0.310 | [0.173, 0.492] |
| **Pr(model flipped \| f flipped, orig=refusal)** *(stricter, causal)* | 9/20 | **0.450** | [0.258, 0.658] |

Judge ↔ heuristic agreement: **98.9%** (209/211). 0 judge errors.

### Two non-obvious findings worth a slide each

1. **The activation-over-TF-IDF gap grows with task difficulty.**
   cyber_1 +0.037, cyber_2 +0.059, cyber_3 +0.065 (exp 09). Cheap baseline
   establishes the floor; activations earn more bits as the discrimination
   gets harder. Suggests evaluating any new probe head primarily on the
   harder tiers.

2. **The refusal probe suffers the Kramár threshold-shift on long prompts.**
   AUC barely budges (short 0.920 → long 0.918), but FNR at 1% FPR (production
   guardrail) jumps from **49% → 80%** (exp 15 Kramár eval). Direct
   motivation for the Rolling/MultiMax probe (exp 16). Cyber_1 does NOT
   suffer this — long cyber prompts are EASIER, not harder (exp 10). The
   threshold-shift story is refusal-specific.

---

## Per-experiment summary

> Layout convention: each `experiments/NN_<topic>_omar/` is self-contained
> with a `notes.md`, scripts, log files, plots, and a `results.json`.

### exp 03 — Cyber_1 layer sweep (1,000 samples)

- **Goal**: train Gemma 4-31B-it residual probes for cyber_1 (dual_use vs
  benign), sweep over 13 layers × {mean, last-token} pooling, find the best
  layer.
- **Result**: peak test AUC **0.983 ± 0.011** at mean-pool layer 40, 5-fold
  CV. Best last-token AUC also at L40 (~0.97). Plots:
  `auc_vs_layer.png`, `metrics_vs_layer.png`.
- **Implications**: layer 40 is the canonical "best" for cyber_1.

### exp 05 — Constitutional Classifiers++ probe heads

- **Goal**: head-to-head compare 4 probe-training techniques on the same
  cyber_1 selection: (D) linear+mean-pool baseline at L40, (A) multi-layer
  concat + sklearn LR with C-sweep, (B) SWiM (sliding-window-mean → max),
  (C) softmax-weighted BCE.
- **Result (partial — stopped at user request to refocus on Level 2 work)**:
  - Head **D**: AUC 0.986, CI [0.974, 0.995], acc 0.930.
  - Head **A**: AUC 0.984, CI [0.966, 0.996], C=10 chosen.
  - Head **B** seeds 0-1 mean: ≈ 0.975.
  - Head **C** seeds 0-1: max_only 0.985, swim_max 0.983.
  - All four cluster within ±0.011 AUC of 0.985. **Linear baseline (Head D)
    is the highest point estimate** — capacity isn't the bottleneck on
    cyber_1.
- **Implications**: C-sweep on Head A landed C=10 (almost no L2). Head B/C
  multi-seed (5 seeds each) was not finished — see notes.md for partial
  per-seed numbers. The CC++ stack does not beat the linear baseline on
  cyber_1 in our 1,000-sample, 800-train regime.

### exp 06 — Cyber_2 extract + probes (1,000 samples)

- **Goal**: extract Gemma residuals for the cyber_2 task (high_risk_dual_use
  vs dual_use ∪ benign), train layer sweep.
- **Result**: peak AUC **0.955 ± 0.017 at last-token L60** (acc 0.896). Mean
  pool peak **0.946 ± 0.012 at L40**. All 13 layers × 2 poolings.
- **Implications**: cyber_2's harm-tier discrimination is harder than
  cyber_1 (≈ 0.05 AUC drop) — and last-token pooling beats mean-pool here,
  which is interesting.

### exp 07 — Cyber_3 extract + probes (1,000 samples)

- **Goal**: cyber_3 = prohibited vs hrdu ∪ du ∪ ben.
- **Result**: peak AUC **0.955 ± 0.016 at mean-pool L35** (acc 0.886).
  last-token peaks at L60 (0.923). Layer 0 last-token is degenerate at 0.5
  (embedding has no contextual information).
- **Implications**: cyber_2 and cyber_3 tied at AUC 0.955 but at different
  sweet spots — cyber_2 prefers final-token L60, cyber_3 prefers mid-stack
  mean-pool L35.

### exp 08 — Cyber_1 scale-up to 3,000 samples + held-out test

- **Goal**: take exp 03's 1,000-sample probe and scale to 2,267 train + 923
  held-out cyber_1-eligible test samples.
- **Result**: held-out test **AUC 0.988 [0.982, 0.992] @ mean L40**. CV AUC
  0.984 (5-fold on train). **Gap CV − held-out = −0.005** (probe
  generalizes; held-out is *higher* than CV, no overfit).
- **Implications**: cyber_1 probe scales cleanly — exp 03's 0.983 holds up
  with more data and on a real held-out split. This is the strongest
  generalization signal in the project.

### exp 09 — Stress-tests for exp 03 (D1 / D2 / D3)

- **D1 — TF-IDF baseline** for cyber_1, cyber_2, cyber_3:
  | Task | TF-IDF AUC | Activation peak | Δ (act − TF-IDF) |
  |---|---:|---:|---:|
  | cyber_1 | 0.946 ± 0.020 | 0.983 | **+0.037** |
  | cyber_2 | 0.887 ± 0.031 | 0.946 | **+0.059** |
  | cyber_3 | 0.890 ± 0.015 | 0.955 | **+0.065** |
  → activations earn MORE marginal bits on harder tiers.
- **D2 — Regularisation sweep** at mean L30: peak AUC 0.9809 @ C=1 with L1
  norm 283; **C=0.1 matches within 0.005 AUC at L1=109.9** (~2.6× sparser).
  Probe doesn't need 5,376 dimensions. (Last-token L35 sweep was killed at
  user request.)
- **D3 — OOD held-out** ran but crashed at the LR-fit step due to fp16
  pooling overflow. Effectively redundant with exp 08's 923-sample held-out
  AUC (0.988), so deferred.

### exp 10 — Length-stratified AUC for cyber_1/2/3 (in flight)

- **Goal**: tertile-bucket prompts by char length, report AUC per bucket at
  each task's best layer.
- **cyber_1 result (done)**: short 0.979 / medium 0.980 / **long 0.991**
  — long is *easier*, not harder. Opposite of Kramár's prediction. Probably
  because long cyber prompts are more elaborately dual-use (more signal
  tokens to mean-pool).
- **cyber_2/cyber_3**: still running at write time.
- **Implications**: Kramár MultiMax/Rolling motivation is REFUSAL-specific
  (see exp 15) — not a cyber concern.

### exp 11 — Refusal probe + Arditi direction (Gemma)

- **Goal**: train an LR refusal probe on Gemma at L32 (single-layer extract);
  also compute the Arditi 2024 unit-norm refusal direction
  `(mean(refused) − mean(complied))/||·||`.
- **Result**:
  - Mean-pool 5-fold CV **AUC 0.9265 ± 0.0134**.
  - Last-token CV **AUC 0.9170 ± 0.0157**.
  - Arditi-direction projection AUC (no LR fit) **0.758** — i.e. the LR
    probe adds +0.17 AUC over the bare 1-D direction.
- **Outputs**: `arditi_direction.npy` (a unit vector in R^5376), `lr_probe.npz`
  (LR coefficients), `auc_by_pooling.png`, `fold_auc.png`.
- **Implications**: the Arditi direction is real signal (well above chance)
  but the full LR probe is materially stronger. For Level 2 attribution
  (which-token-to-edit) Arditi works. For *scoring* candidate edits, use the
  LR probe — the "two different `f`s" architectural choice from
  `experiments/extra_omar/notes.md`.

### exp 12 — Per-token Arditi attribution

- **Goal**: project each token's L32 residual onto the Arditi direction to
  rank refusal-pulling tokens; produce the `[POS|SCORE]token` markup the
  starter `iterative_edit_agent.py` expects as input.
- **Result on the 81 attribution_eval prompts**:
  - All 81 processed, 0 skipped.
  - Per-prompt **max attribution ~ +142** (range 131–149); per-prompt min ~
    −4.80 (basically a constant template token, not a meaningful "compliance
    pull" — see notes).
  - **No harmful keywords** ("keylogger", "ransomware", "exploit", "malware",
    "phish", "backdoor", "rootkit", "trojan", "C2", …) appear in *any*
    prompt's top-8 across all 81 prompts.
  - Most-frequent top-tokens across the 81 prompts: `detection` (38%),
    `isolated` (33%), `test` (28%), `environment` (21%), `authorized` (15%),
    `forensic` (12%), `simulated` (11%), `legitimate` (10%), `educational`
    (6%) — all *safety-washing* / framing tokens, not harmful-content tokens.
- **Plots**: `per_prompt_attribution_strip.png`, `score_distribution.png`,
  `top_tokens_overall.png`, `rank_vs_score.png`. The strip plot has the
  x-axis fixed to normalised [0,1] (correct now after a label bug-fix).
- **Implications (interpretability)**: on safety-washed attacks the Arditi
  direction picks up the *cover-story vocabulary*. We can't conclude from
  this alone that Gemma's refusal is fundamentally surface-level — there's
  no control set of plain harmful asks in this 81-prompt eval. But it does
  predict that rewrites stripping safety-washing should flip the probe.

### exp 13 — PRE rewrite scoring + Gemma rollouts + LLM judge

- **Goal**: full Probe-Rewrite-Evaluate loop on the 81 attribution_eval
  prompts using the user-supplied `rewrites_k7.json` (7 paraphrases per
  prompt, pre-generated by DeepSeek). Score each rewrite with both the LR
  probe and the Arditi direction; pick the best per scorer; roll Gemma on
  the best rewrite; judge with DeepSeek V4 Pro using
  `REFUSAL_JUDGE_SYSTEM`.
- **Rewrite stats**: TF-IDF cosine to original mean **0.30**; word-Jaccard
  mean **0.26**; **length ratio mean 0.34** — DeepSeek didn't paraphrase, it
  *stripped* (~⅔ of length removed), exactly the "safety-washing removal"
  pattern predicted by exp 12.
- **Probe-flip rate**: 29/81 (35.8%) prompts have a rewrite with `lr_prob <
  0.5`. Arditi-direction-best and LR-best agree on only 32/81 (40%) of
  prompts → the two scorers genuinely disagree.
- **Headline (Wilson 95% CIs)** — see top of this file.
- **Side-finding**: 10/81 originals were judged COMPLIANCE despite the
  source corpus pre-labelling them all `is_refusal=True`. Gemma actually
  answered some of them — the judge was honest about it. Headline n drops
  to 71 because of this.
- **Outputs**: `scores.jsonl`, `best_rewrites.json`, `rollouts.jsonl`,
  `judgements.jsonl`, `level2_metrics.json`, `level2_summary.md`.

### exp 14 — Cross-task probe transfer

- **Goal**: take each of cyber_1 / cyber_2 / cyber_3's best-layer LR probe,
  predict on the OTHER two tasks.
- **Result**: 3×3 cross-task AUC matrix (rows = train task, cols = test task,
  diagonal is the training fit):
  | train ↓ \\ test → | cyber_1 | cyber_2 | cyber_3 |
  |---|---:|---:|---:|
  | cyber_1 | 1.000 | **0.886** | 0.713 |
  | cyber_2 | **0.913** | 1.000 | 0.794 |
  | cyber_3 | 0.803 | 0.733 | 1.000 |
- **Implications**: all off-diagonals are above chance — the tiers share a
  related direction. **cyber_2 ↔ cyber_1 transfer best** (0.886 / 0.913),
  while **cyber_3 generalizes worst** in both directions (0.713-0.803).
  Prohibited content is qualitatively different from dual-use in residual
  space, not just "more harmful". No single linear direction explains all
  three; a multi-task probe would help.

### exp 15 — Refusal length-stratified + Kramár FNR @ strict FPR

- **AUC by length tertile** (5-fold CV, mean L32):
  | Bucket | n samples | AUC | Std |
  |---|---:|---:|---:|
  | short (≤ 6,490 chars) | 277 | **0.9274 ± 0.016** | tightest |
  | medium | 277 | 0.9135 ± 0.048 | 3× std |
  | long (> 11,253 chars) | 278 | 0.9100 ± 0.048 | 3× std |

- **Kramár-style FNR @ strict FPR** (out-of-fold predictions; threshold
  locked on the SHORT bucket, applied to all):
  | Bucket | AUC | n_pos | FNR @ 1% FPR | FNR @ 5% FPR |
  |---|---:|---:|---:|---:|
  | short | 0.920 | 174 | **49.4%** [42, 57] | **27.6%** [22, 35] |
  | medium | 0.913 | 162 | 64.2% [57, 71] | 37.0% [30, 45] |
  | **long** | 0.918 | 83 | **79.5%** [70, 87] | **62.7%** [52, 72] |

- **Implications**: AUCs look essentially flat across buckets but the FNR
  number a production guardrail actually deploys (1% FPR threshold) jumps
  from 49% → 80% short → long. Direct motivation for the Kramár Selected
  Probe (rolling attention) on refusal data — exp 16. Long bucket is also
  class-skewed (only 30% refusal vs 63% short) — long prompts in this
  corpus tend to be legitimate technical asks, but the few that *are*
  refusals get badly missed at the strict threshold.

### exp 16 — Kramár MultiMax / Rolling Attention probes (in flight)

- **Spec**: `IMPLEMENTATION_SPEC.md` — 4 architectures (attention_kramar,
  multimax, rolling, rolling_multimax). Background agent currently
  implementing the Rolling probe (Eq. 10, paper §3.2.2 — the "Selected
  Probe") on cyber_1/2/3 and Refusal-Gemma. Hyperparameters from Kramár
  Appendix C: AdamW, lr=1e-4, wd=3e-3, 1000 steps, d_hidden=100, H=10,
  w=10. CPU-only training.
- **Why this matters**: exp 15 Kramár FNR shows the production threshold-shift
  vulnerability is real on refusal. Rolling/MultiMax is the paper's fix.
  Goal is to narrow the 49% → 80% FNR gap on long prompts.
- **Pending**: training results per task + per-length-bucket FNR for refusal
  to claim the gap closure directly.

---

## What's NOT here / known gaps

1. **Refusal-Qwen** (5th task in the README's table). Qwen 3.6-27B not
   downloaded locally (~55 GB). Skip with a "compute-cost decision" note in
   the slides.
2. **Cyber-tier scale-up** (TODO 9b). cyber_2/cyber_3 still at 1,000
   samples. The shared extract dir already has all needed dual_use + benign;
   only positive-class (hrdu, prohibited) extraction would cost GPU time.
   Deferred to focus on Level 2.
3. **CC++ exp 05 final seeds**. Only seeds 0-1 of Heads B/C done. Killed at
   user request.
4. **Plots for some experiments**. exp 08, exp 13, exp 14 missing dedicated
   plots (data is in `results.json`).
5. **Kramár Rolling probe results**. Implementation in flight (exp 16).

---

## File / experiment cross-reference

| Slide section | Pull from |
|---|---|
| Level 1 AUC table | exp 03, 06, 07, 08, 11 |
| Activations vs TF-IDF | exp 09 D1 + D1 cyber23 extension |
| Probe sparsity (C-sweep) | exp 09 D2 mean_L30 |
| Length-stratified — cyber | exp 10 (cyber_1 done; cyber_2/3 in flight) |
| Length-stratified — refusal | exp 15 |
| Kramár threshold-shift FNR | exp 15 `kramar_results.json` |
| Per-token Arditi attribution | exp 12 (4 plots) |
| Rewrite similarity / strip stats | exp 13 `similarity_summary.json` |
| PRE Level 2 metrics | exp 13 `level2_metrics.json`, `level2_summary.md` |
| Worked examples (probe flip + behavior flip / miss) | exp 13 `level2_summary.md` |
| Selected Probe (Kramár Rolling) | exp 16 (in flight) |

---

## Open / pending tasks (TaskList snapshot)

See `TODO.md` Phase 9 + Phase 10 for the full backlog. The biggest movers
left for the slide deck:

- finish exp 16 Rolling probe (in-flight agent)
- finish exp 14 cross-task (in-flight)
- finish exp 10 cyber length-strat (in-flight)
- write the "what surprised us / what didn't work" slide
- pick 2-3 worked examples from `level2_summary.md` for the appendix
