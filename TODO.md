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
