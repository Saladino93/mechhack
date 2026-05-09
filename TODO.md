# Level 1 — TODO List

## Phase 0: Extract residual activations on GPU cluster

> **Prerequisite for everything. Must run on A100/H200.**

### 0a. Extract Gemma refusal residuals (layer 32)
- [ ] SSH into cluster, get a pod: `runai submit dev-pod --image registry.rcp.epfl.ch/mlo-protsenk/redteam-mechinterp:v9 --gpu 1 --pvc hackathon-mechhack-scratch-gNN:/scratch --pvc hackathon-mechhack-shared-ro:/data --command -- sleep infinity`
- [ ] Clone repo onto pod: `runai exec -it dev-pod -- bash`, then `cd /scratch && git clone ...`
- [ ] Set AIaaS key: `export AIAAS_KEY=sk--...`
- [ ] Run extraction:
  ```bash
  python starter_code/extract_residuals.py \
      --model_key gemma4_31b \
      --layers middle \
      --samples_file datasets/refusal_probes/gemma4_31b/attacks_full.jsonl \
      --out_dir ./extracts/gemma_refusal_L32
  ```
- [ ] Verify: 878 `.pt` files created, each with `residuals: (1, n_tokens, 8192)`
- [ ] Verify: `extraction_metadata.json` exists and lists all samples
- [ ] Spot-check: load one `.pt`, confirm keys `residuals`, `input_ids`, `attention_mask`, `label`, `layer_idxs`

### 0b. Extract Gemma cyber residuals (layer 32)
- [ ] Run on train split:
  ```bash
  python starter_code/extract_residuals.py \
      --model_key gemma4_31b \
      --layers middle \
      --samples_file datasets/cyber_probes/train.jsonl \
      --out_dir ./extracts/gemma_cyber_L32_train
  ```
- [ ] Run on test split:
  ```bash
  python starter_code/extract_residuals.py \
      --model_key gemma4_31b \
      --layers middle \
      --samples_file datasets/cyber_probes/test.jsonl \
      --out_dir ./extracts/gemma_cyber_L32_test
  ```
- [ ] Verify: 5,079 train + 2,180 test `.pt` files
- [ ] Note: cyber dataset uses field `"prompt"` not `"attack_prompt"` — check if `extract_residuals.py` handles both or needs a flag

### 0c. Extract Qwen refusal residuals (layer 32)
- [ ] Run extraction:
  ```bash
  python starter_code/extract_residuals.py \
      --model_key qwen36 \
      --layers middle \
      --samples_file datasets/refusal_probes/qwen36/attacks_full.jsonl \
      --out_dir ./extracts/qwen_refusal_L32
  ```
- [ ] Verify: 878 `.pt` files, each with `residuals: (1, n_tokens, 5120)`
- [ ] Note: Qwen is hybrid arch (16 full-attention + 48 DeltaNet). Layer 32 falls in DeltaNet region. Consider also trying layer 8 or 16 (full-attention region) if results are weak.

### 0d. (Stretch) Extract multi-layer residuals
- [ ] Only do this after single-layer AUC is known from Phases 2-3
- [ ] Run:
  ```bash
  python starter_code/extract_residuals.py \
      --model_key gemma4_31b \
      --layers "16,24,32,40,48" \
      --samples_file datasets/refusal_probes/gemma4_31b/attacks_full.jsonl \
      --out_dir ./extracts/gemma_refusal_multi
  ```
- [ ] Output: residuals shape `(5, n_tokens, 8192)` per sample — 5x more data per file
- [ ] Repeat for cyber if needed

**Estimated time: ~1-2h total for 0a-0c on A100**

---

## Phase 1: Build data loading utilities

> **Write once, use everywhere. All probes depend on this.**

- [ ] Create `load_extracts.py` with a function:
  ```python
  def load_dataset(
      extracts_dir: str,
      task: str,           # "refusal_gemma", "refusal_qwen", "cyber_1", "cyber_2", "cyber_3"
      position: str,       # "last", "mean", "all"
      layer_idx: int = 0,  # index into extracted layers array
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
      """Returns X_train, y_train, X_test, y_test"""
  ```

- [ ] Implement token position strategies:
  - `"last"`: `residuals[layer_idx, attention_mask.nonzero().max(), :]` — shape `(d_model,)`
  - `"mean"`: `residuals[layer_idx, :n_tokens, :].mean(dim=0)` — shape `(d_model,)`
  - `"all"`: `residuals[layer_idx, :n_tokens, :]` — shape `(n_tokens, d_model)` (for attention probes)

- [ ] Implement label construction matching `scoring/score_probes.py`:
  - Refusal tasks: `label = int(sample["is_refusal"])`
  - Cyber Probe-1: positive = `dual_use`, negative = `benign` (skip `high_risk_dual_use` and `prohibited`)
  - Cyber Probe-2: positive = `high_risk_dual_use`, negative = `dual_use ∪ benign` (skip `prohibited`)
  - Cyber Probe-3: positive = `prohibited`, negative = everything else

- [ ] Handle train/test splits:
  - Refusal: match sample_ids against `train_split.jsonl` / `test_split.jsonl`
  - Cyber: use `train.jsonl` / `test.jsonl` directly

- [ ] Verification checks:
  - [ ] Shapes correct: `X_train.shape == (n_train, d_model)` for last/mean
  - [ ] Label distributions match expected:
    - Gemma refusal: ~281 pos / 308 neg train, ~140 pos / 149 neg test
    - Qwen refusal: ~178 pos / 411 neg train, ~97 pos / 192 neg test
    - Cyber Probe-1: 685 dual_use (pos) / 1582 benign (neg) in train
    - Cyber Probe-2: 1052 hrdu (pos) / 2267 du+benign (neg) in train
    - Cyber Probe-3: 1760 prohibited (pos) / 3319 rest (neg) in train
  - [ ] No NaN/Inf in residuals
  - [ ] No sample_id overlap between train and test

**Estimated time: ~30 min**

---

## Phase 2: Difference-in-means baseline (Arditi's method)

> **Simplest possible probe. ~10 lines. Establishes the AUC floor.**

- [ ] Implement:
  ```python
  def difference_in_means_probe(X_train, y_train, X_test, y_test):
      mean_pos = X_train[y_train == 1].mean(axis=0)
      mean_neg = X_train[y_train == 0].mean(axis=0)
      direction = mean_pos - mean_neg
      direction /= np.linalg.norm(direction)
      scores = X_test @ direction
      auc = roc_auc_score(y_test, scores)
      return auc, direction
  ```

- [ ] Run for all 5 tasks with `position="last"`
- [ ] Run for all 5 tasks with `position="mean"`
- [ ] Pick the better position per task

- [ ] Record results:
  ```
  Task              | last-token AUC | mean-token AUC | Best
  refusal_gemma     |                |                |
  refusal_qwen      |                |                |
  cyber_probe_1     |                |                |
  cyber_probe_2     |                |                |
  cyber_probe_3     |                |                |
  ```

- [ ] Save direction vectors: `probes/dim_direction_{task}.npy`
- [ ] Save results: `probes/dim_results.json`

- [ ] **Interpret**: if AUC > 0.85, signal is strongly linear — remaining phases are polish. If AUC < 0.7, need nonlinear or multi-layer methods.

- [ ] **Note**: the saved direction vectors ARE the refusal directions (Arditi). Reusable for Level 2 attribution.

**Estimated time: ~15 min**

---

## Phase 3: Logistic regression probe

> **One step up. Optimizes the decision boundary directly.**

- [ ] Implement:
  ```python
  from sklearn.linear_model import LogisticRegression
  from sklearn.preprocessing import StandardScaler

  scaler = StandardScaler()
  X_train_s = scaler.fit_transform(X_train)
  X_test_s = scaler.transform(X_test)

  best_auc = 0
  for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
      clf = LogisticRegression(C=C, max_iter=2000, solver='lbfgs')
      clf.fit(X_train_s, y_train)
      scores = clf.predict_proba(X_test_s)[:, 1]
      auc = roc_auc_score(y_test, scores)
      if auc > best_auc:
          best_auc, best_C = auc, C
  ```

- [ ] Run for all 5 tasks using best position from Phase 2
- [ ] Record best C and AUC per task
- [ ] Compare to difference-in-means: how much does optimization help?

- [ ] **Compute error bars** (required for submission):
  ```python
  def bootstrap_auc(y_true, y_score, n_bootstrap=2000):
      rng = np.random.default_rng(42)
      aucs = []
      for _ in range(n_bootstrap):
          idx = rng.choice(len(y_true), len(y_true), replace=True)
          if len(np.unique(y_true[idx])) < 2:
              continue
          aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
      lo, hi = np.percentile(aucs, [2.5, 97.5])
      return np.mean(aucs), lo, hi
  ```
- [ ] Report: `AUC = 0.XX [0.XX, 0.XX] 95% CI` per task

- [ ] Save: `probes/logreg_{task}.pkl` (model + scaler)
- [ ] Save: `probes/logreg_results.json`

**Estimated time: ~15 min**

---

## Phase 4: Attention-pooling probe

> **Uses full token sequence. Learns WHERE in the prompt to look.**

- [ ] Use `AttentionProbe` from `starter_code/train_probe.py`:
  ```
  Input: x ∈ R^(n_tokens, d_model)
  1. Learned query q ∈ R^(d_model,)
  2. Attention weights: a = softmax(x @ q / sqrt(d))
  3. Pooled: z = a^T @ x
  4. Output: score = w^T @ z + b
  ```

- [ ] Load data with `position="all"` (full sequence per sample)
- [ ] Handle variable-length sequences: pad + attention mask
- [ ] Split train → train/val (80/20) for early stopping
- [ ] Training config:
  - Loss: `BCEWithLogitsLoss`
  - Optimizer: `AdamW`, lr=5e-4, weight_decay=1e-3
  - Epochs: up to 50, early stopping patience=5
  - Run 5 seeds (0-4)

- [ ] Also try `MultiHeadAttentionProbe` (K=4 heads)
- [ ] Compare AUC to logistic regression baseline
- [ ] Report: mean AUC ± std across seeds per task

- [ ] Save: `probes/attn_{task}_seed{s}.pt`
- [ ] Save: `probes/attn_results.json`

**Estimated time: ~30 min**

---

## Phase 5: MultiMax aggregation probe (Kramar's method)

> **Hard-max per head. Prevents signal dilution on long prompts.**

- [ ] Implement:
  ```python
  class MultiMaxProbe(nn.Module):
      def __init__(self, d_model, n_heads=4, d_hidden=256):
          self.mlp = nn.Sequential(nn.Linear(d_model, d_hidden), nn.GELU())
          self.v = nn.Parameter(torch.randn(n_heads, d_hidden))
          self.output = nn.Linear(n_heads, 1)

      def forward(self, x, attention_mask=None):
          h = self.mlp(x)                                    # (B, T, d_hidden)
          scores = torch.einsum('bnd,hd->bnh', h, self.v)    # (B, T, H)
          if attention_mask is not None:
              scores = scores.masked_fill(~attention_mask.unsqueeze(-1), float('-inf'))
          pooled = scores.max(dim=1).values                   # (B, H)
          return self.output(pooled).squeeze(-1)               # (B,)
  ```

- [ ] Train same config as Phase 4
- [ ] Try n_heads ∈ {2, 4, 8}
- [ ] Compare AUC vs attention probe, especially on long prompts (cyber dataset)
- [ ] Subset analysis: AUC on prompts >2048 tokens vs <512 tokens

- [ ] (Optional) Also try Rolling Attention:
  - Sliding windows of width w=10
  - Attention within each window, then max across windows

- [ ] Save: `probes/multimax_{task}_h{n}_seed{s}.pt`
- [ ] Save: `probes/multimax_results.json`

**Estimated time: ~30 min**

---

## Phase 6: Multi-layer probe (CC++ style)

> **Concatenate activations from multiple layers. Requires Phase 0d extraction.**

- [ ] **Prerequisite**: Phase 0d must be done (extract layers 16,24,32,40,48)
- [ ] Concatenate last-token activations across layers:
  ```python
  x = np.concatenate([residuals[i, last_pos, :] for i in range(n_layers)])
  # Shape: (n_layers * d_model,) = (5 * 8192,) = (40960,)
  ```

- [ ] Handle high dimensionality (40960 dims, ~300-600 train samples):
  - [ ] Try PCA → 256-512 dims → LogisticRegression
  - [ ] Try heavy L2 regularization: LogisticRegression(C=0.001)
  - [ ] Try small MLP: 40960 → 256 → 1 with dropout=0.3

- [ ] (Optional) Implement SWiM smoothing from CC++:
  ```python
  def swim_smooth(per_token_logits, window=16):
      kernel = torch.ones(window) / window
      smoothed = F.conv1d(logits.unsqueeze(0).unsqueeze(0),
                          kernel.unsqueeze(0).unsqueeze(0),
                          padding=window // 2).squeeze()
      return smoothed.max()
  ```

- [ ] (Optional) Implement softmax-weighted loss from CC++:
  ```python
  def softmax_weighted_bce(logits, label, tau=1.0):
      weights = F.softmax(logits / tau, dim=0)
      bce = F.binary_cross_entropy_with_logits(logits, label.expand_as(logits), reduction='none')
      return (weights * bce).sum()
  ```

- [ ] Compare AUC: single-layer vs multi-layer per task
- [ ] Save: `probes/multilayer_{task}.pt`
- [ ] Save: `probes/multilayer_results.json`

**Estimated time: ~1h (including extraction)**

---

## Phase 7: Final evaluation + submission packaging

> **Pick best method per task. Compute final numbers. Package for scoring.**

- [ ] Compile comparison table across all phases per task
- [ ] Select best method per task (may differ across tasks)
- [ ] Compute bootstrap 95% CI for each final AUC (n_bootstrap=2000)
- [ ] Compute headline metric: `mean_auc = mean(5 AUCs)`, `std = std(5 AUCs)`

- [ ] Write `submission/predict.py`:
  ```python
  def load_predictor() -> Any:
      """Load best probe weights per task."""

  def predict(predictor, residuals: np.ndarray, attention_mask=None) -> float:
      """Returns float in [0, 1] — probability of positive class."""
  ```

- [ ] Run official scorer:
  ```bash
  python scoring/score_probes.py --submission_dir ./submission/
  ```
- [ ] Verify output matches our computed AUCs

- [ ] Save: `results/final_aucs.json`, `results/comparison_table.md`

**Estimated time: ~30 min**

---

## Phase 8: Prepare presentation (Level 1 portion)

> **10 min talk. Methodology → results → insights.**

- [ ] Slide 1: What we built — `f: model_internals → predicted_label`, method hierarchy
- [ ] Slide 2: Results table — 5 AUCs with 95% CIs, method comparison
- [ ] Slide 3: What the probe learned — PCA visualization, weight interpretation
- [ ] Slide 4: Layer analysis (if multi-layer done) — AUC per layer curve
- [ ] Slide 5: Surprises / what didn't work
- [ ] Slide 6: Connection to production (Mythos, CC++, Kramar)
- [ ] Last slide: Code link for reproducibility

**Estimated time: ~1h**

---

## Dependency graph

```
Phase 0 (extract on cluster)
  │
Phase 1 (data loader)
  │
  ├── Phase 2 (diff-in-means)     ─┐
  ├── Phase 3 (logistic reg)       │
  ├── Phase 4 (attention probe)    ├── can run in parallel
  ├── Phase 5 (MultiMax)           │
  └── Phase 6 (multi-layer) ──────┘ (needs Phase 0d first)
        │
Phase 7 (final eval + submission)
  │
Phase 8 (presentation)
```

## Time budget

| Phase | Task | Est. Time | Cumulative |
|-------|------|-----------|------------|
| 0 | Extract residuals | 1-2h | 2h |
| 1 | Data loader | 30min | 2.5h |
| 2 | Diff-in-means | 15min | 2.75h |
| 3 | Logistic regression | 15min | 3h |
| 4 | Attention probe | 30min | 3.5h |
| 5 | MultiMax | 30min | 4h |
| 6 | Multi-layer | 1h | 5h |
| 7 | Final eval | 30min | 5.5h |
| 8 | Slides | 1h | 6.5h |

**Phases 2-5 run in parallel after Phase 1, so wall-clock is ~4-5h total.**

---

## Phase 9: Post-disconnect follow-ups (added 2026-05-09)

> Items raised during exp 03→09 retros. Captured here so they aren't lost. Ordering reflects user-stated priority.

### 9a. Length-stratified AUC + accuracy (exp 10)
- [ ] Gated on exp 06 (cyber_2 retrain) finishing
- [ ] Bucket prompt lengths into tertiles (or fixed quantiles) using `selection.json` for each task
- [ ] Re-score existing 5-fold CV folds from exp 03 (cyber_1), exp 06 (cyber_2), exp 07 (cyber_3) per bucket at the best layer × pooling
- [ ] Output: AUC + acc per length bucket per task per pooling, plus a plot
- [ ] Goal: identify if "long prompt" is the dominant failure mode — directly motivates Phase 5 (MultiMax) and 9c (Kramar head)
- [ ] Lives at `experiments/10_length_stratified_omar/`

### 9b. Scale-up cyber_2 + cyber_3 selections
- [ ] Current selections: 500/500 each. Extracts dir `/home/ubuntu/extracts/cyber_all_omar/` already has ~3500 cached extracts → most extraction is reuse.
- [ ] Note (per user, 2026-05-09): cyber_2 / cyber_3 are progressively *broader* than cyber_1 — they include cyber_1 cases plus more. Selection logic should reflect this hierarchy.
- [ ] Update `experiments/06_cyber2_extract_omar/select.py` and `07_cyber3_extract_omar/select.py` to draw a larger balanced selection (target: 1500/1500 or as data permits)
- [ ] Re-run extract.py incrementally — only extract missing samples, do not re-extract the 1000 already done
- [ ] Re-run train_probes.py + plot.py for each
- [ ] Update notes.md with new sample sizes and headline AUCs

### 9c. GDM Kramar-2026 MultiMax + Rolling Attention head
- [ ] Reference: `papers/kramar2026_production_ready_probes_gemini.pdf`. Detailed sketch already in `experiments/extra_omar/notes.md` Tier 2 §4.
- [ ] Already partially scoped as Phase 5 above — this entry is the *production* version with both MultiMax (hard-max per attention head) **and** Rolling Attention (sliding windows of width 10, max across windows)
- [ ] Goal: long-prompt + OOD robustness. Kramar's headline: long-context FNR drops from 87.9% (EMA probe) to 3.0% (Selected = MultiMax + Rolling)
- [ ] CPU-only. Probably extends `experiments/05_cc_plus_plus_omar/train_heads.py` or a new sibling experiment
- [ ] Eval on length-stratified folds from 9a so we can directly cite the long-bucket FNR improvement
- [ ] **Full implementation spec at `experiments/16_multimax_probe_omar/IMPLEMENTATION_SPEC.md`** (4 architectures, hyperparams from Appendix C, vectorised pseudocode, CLI examples). Tracks 9c at the implementation level.

### 9c-impl. Implement Kramar 4-architecture probe family (exp 16)
- [ ] Spec: `experiments/16_multimax_probe_omar/IMPLEMENTATION_SPEC.md`. Architectures:
  1. `attention_kramar` — Eq. 7-8 baseline (softmax + 2-layer ReLU MLP transform)
  2. `multimax` — Eq. 9 (hard argmax per head, all-tokens scope)
  3. `rolling` — Eq. 10 (softmax within w=10 windows, max across windows) — the "Selected Probe"
  4. `rolling_multimax` — combination (hard argmax within window + max across windows)
- [ ] Hyperparameters from Kramar Appendix C: AdamW, lr=1e-4, wd=3e-3, 1000 steps, d_hidden=100, H=10, w=10.
- [ ] Build files: `probes.py` (4 nn.Module classes + factory), `train.py` (CLI loader + 5-fold CV + held-out test), `notes.md` (results write-up).
- [ ] **Level 1 use** — train all 4 architectures on cyber_1, cyber_2, cyber_3, and Gemma refusal. Compare to current LR mean-pool (cyber: 0.946-0.988; refusal: 0.927). Especially target the long-prompt regime where exp 15 showed AUC 0.927 → 0.910 on refusal.
- [ ] **Level 2 use** — swap the trained `rolling` (or `rolling_multimax`) probe in as the scoring `f` in the PRE pipeline (exp 13). Keep Arditi for attribution; use Kramar probe for picking-best-of-7. The "two different `f`s" architecture from `experiments/extra_omar/notes.md`.
- [ ] Output: per-task AUC table, per-architecture × per-length-bucket AUC (re-uses exp 10 / exp 15 length tertiles).

### 9d. Quadratic / polynomial probe ablation (Pleshkov 2026)
- [ ] From `experiments/extra_omar/notes.md` Tier 2 §5 — not yet in main TODO
- [ ] PCA → degree-2 polynomial lift → ridge regression. d ≤ 32 for cyber to keep features < N
- [ ] Goal: ablation. If quadratic AUC >> linear → evidence against the linear-representation hypothesis on cyber. If ≈ same → confirms linear is sufficient.
- [ ] Cyber-only (refusal is single direction per Arditi)
- [ ] Lives at a new experiment dir TBD

### 9e. Level 2 pipeline (Arditi attribution + Xiong PRE + Boppana scorer)
- [ ] From `experiments/extra_omar/notes.md` Tier 2 §6 + Recommended Pipeline section. Not yet in main TODO.
- [ ] Per-token attribution: refusal-direction projection (Arditi)
- [ ] Edit candidates: k=7 LLM rewrites targeting attributed tokens (Xiong PRE)
- [ ] Candidate scoring: attention-pooling probe (Boppana, `starter_code/train_probe.py`)
- [ ] Verification: re-roll Gemma on edited prompt; report Pr(f|edit), Pr(model|edit), Pr(model|f)
- [ ] Distinct `f`s for attribution vs scoring is the design point — see "Two Different `f`s" section in extra_omar notes

---

## Phase 10: Deferred (paused 2026-05-09 to focus on Arditi / Level 2)

> Items previously in flight or planned that we deliberately paused to free CPU/GPU and attention for the Level 2 pivot. Retain so they can be resumed without re-discovery.

### 10a. exp 05 CC++ — finish seeds 2–4 of Heads B and C
- [ ] Killed mid-run after 4 of 10 head×seed pairs completed. Heads D and A are deterministic single runs and are final (D=0.986, A=0.984). Heads B and C have only seeds 0-1 documented (≈0.971 and ≈0.980 respectively).
- [ ] To finish: relaunch `train_heads.py` (it already supports resuming would need a small flag; otherwise re-run all seeds and discard 0-1 duplicates) → ~60 min CPU.
- [ ] Lower priority: results are already enough to claim "all heads cluster at ~0.98 AUC on cyber_1, linear baseline is competitive" without additional seeds.

### 10b. cyber_2 / cyber_3 scale-up (was Phase 9b)
- [ ] Same as 9b. Marginal gain on top of current 1000-sample results is small now that TF-IDF / activation gap is established. Pick up after Level 2 wraps if there is time.

### 10c. Length-stratified AUC (was 9a)
- [ ] Cheap (~5 min CPU). Deferred until after Level 2. Sub-bucket the existing exp 03/06/07 CV folds by prompt length and report per-bucket AUC.

### 10d. Qwen refusal extraction + probes
- [ ] Qwen3.6-27B not downloaded locally (~55 GB pull). Refusal probes only; not strictly needed for Level 2 if we stick to Gemma.

### 10e. exp 09 D2 last_token L35 sweep
- [ ] Killed before completing. Mean-pool L30 sweep is the more useful one and is already saved. Skip unless we revisit C-sensitivity for the last-token pooling specifically.

### 10f. exp 09 D3 OOD eval
- [ ] In progress (PID 92721, currently at 600/999 features). Will finish naturally — *not* deferred, but listed here for visibility into the resource picture.

