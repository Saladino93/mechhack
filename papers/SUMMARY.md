# Paper Summaries & Hackathon Strategy

## Table of Contents
1. [Individual Paper Summaries](#individual-paper-summaries)
2. [Cross-Paper Synthesis](#cross-paper-synthesis)
3. [What We Can Do From Here](#what-we-can-do-from-here)

---

## Individual Paper Summaries

### 1. Sparse Autoencoders Find Highly Interpretable Features in Language Models
**Cunningham et al. 2023** | [arXiv:2309.08600](https://arxiv.org/abs/2309.08600)

SAEs decompose LLM activations into sparse, monosemantic features that are 2-3x more interpretable than neurons. Trained as overcomplete autoencoders with L1 sparsity penalty on residual stream activations. Features are causally relevant (not just correlated) -- patching a few SAE features shifts model outputs. Enables fine-grained circuit analysis: e.g., a "closing parenthesis" circuit spanning layers 0-5 with interpretable nodes. Key limitation: nonzero reconstruction loss means SAEs miss some information.

**Relevance to hackathon:** SAE features are a candidate feature space for probes (Level 1) and for attribution back to input tokens (Level 2). More monosemantic than neurons, so probes built on them should have cleaner separation.

---

### 2. Towards Automated Circuit Discovery for Mechanistic Interpretability (ACDC)
**Conmy et al. 2023** | [arXiv:2304.14997](https://arxiv.org/abs/2304.14997)

ACDC automates the manual circuit-discovery workflow: given a behavior, it iteratively prunes edges from the model's computational graph using activation patching, keeping only edges whose removal degrades task performance. Successfully recovers known circuits (e.g., IOI in GPT-2 Small -- 68 of 32,000 edges). KL divergence is the most stable metric. Sensitive to corrupted distribution choice and threshold hyperparameters. Misses "negative" components (heads that suppress alternatives).

**Relevance to hackathon:** Could find the refusal circuit in Gemma/Qwen. Need: contrastive datasets (refusal vs. non-refusal prompts), KL divergence metric, and careful corrupted distribution design. Scale concern: ACDC was shown on GPT-2 Small; may need coarser initial screening for 27-31B models.

---

### 3. Sparse Feature Circuits
**Marks et al. 2024** | [arXiv:2403.19647](https://arxiv.org/abs/2403.19647)

Combines SAEs with circuit discovery: finds causal subnetworks of interpretable SAE features. Uses attribution patching (first-order Taylor expansion -- cheap) and integrated gradients for precision. Small circuits (~100 features) explain the bulk of model behavior. Demonstrates SHIFT: inspect features, ablate task-irrelevant ones to debias classifiers (gender accuracy drops from 87% to 52% while profession accuracy rises from 76% to 95% on Gemma-2-2B).

**Relevance to hackathon:** The attribution patching method (IE_atp) is directly applicable for Level 2: identify which SAE features at which token positions drive refusal, then edit those tokens. Position matters -- same feature at different positions plays different causal roles.

---

### 4. Eliciting Latent Predictions with the Tuned Lens
**Belrose et al. 2023** | [arXiv:2303.08112](https://arxiv.org/abs/2303.08112)

The tuned lens trains a lightweight affine map per layer to decode intermediate hidden states into vocabulary distributions, fixing the logit lens's unreliability. Early/middle layers can be *more accurate* than the final layer (the model "overthinks" and changes its mind). Prediction trajectories detect anomalous inputs (near-perfect AUROC for prompt injection on 5/9 tasks). Transfers from base models to fine-tuned variants. Training is fast (<1 hour on 8xA40).

**Relevance to hackathon:** The prediction trajectory across layers could serve as a feature vector for probes. "Prediction depth" (when the model commits to refuse vs. comply) could be a discriminative signal. Cheap to train, transfers to instruction-tuned models.

---

### 5. Refusal in Language Models Is Mediated by a Single Direction
**Arditi et al. 2024** | [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)

**The most directly relevant paper.** Refusal in chat LLMs is mediated by a single linear direction in residual stream activation space. Found via difference-in-means on 128 harmful vs. 128 harmless prompts. Removing this direction (directional ablation or weight orthogonalization) eliminates refusal; adding it induces refusal on harmless prompts. Works across 13 models (1.8B-72B). Adversarial suffixes work by redirecting attention away from harmful tokens, suppressing the refusal direction. Minimal capability degradation (<1% on MMLU/ARC/GSM8K).

**Relevance to hackathon:**
- **Level 1:** The refusal direction *is* a probe -- project activations onto it and threshold.
- **Level 2:** Per-token contribution to the refusal direction gives direct attribution. The top-8 attention heads that write to this direction attend to specific input tokens -- those are the tokens to edit.
- **Key insight:** Only 128+128 examples needed. Extraction is trivial (difference-in-means).

---

### 6. Probe-Rewrite-Evaluate (PRE)
**Xiong et al. 2025** | [arXiv:2509.00591](https://arxiv.org/abs/2509.00591)

**Closest analog to the Level-2 loop.** A training-free pipeline: (1) linear probe scores how "test-like" vs "deployment-like" a prompt looks, (2) LLM rewrites prompts to look more deployment-like, (3) evaluate behavior change. Using k=7 candidate rewrites scored by probe, they reduce deception by 25-35% across GPT-4.1, Claude, Gemini, Llama. Probe score changes are monotonically predictive of behavioral transitions. The probe transfers cross-architecture (trained on Llama, works on Claude/GPT/Gemini).

**Relevance to hackathon:** The k-best-of-n with probe scoring is exactly the Level-2 inner loop: generate candidate edits, score with probe, pick best. The rewriter system prompt is the key engineering artifact. Their transition matrix + Awareness Elasticity metric gives a principled evaluation framework.

---

### 7. Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought
**Boppana et al. 2026** | [arXiv:2603.05488](https://arxiv.org/abs/2603.05488)

Reasoning models often commit to their final answer early in their internal representations, then produce "performative" CoT that doesn't reflect genuine deliberation. Attention-pooling probes on residual stream activations decode the answer from the first tokens of reasoning (87.98% accuracy on DeepSeek-R1 671B MMLU), while text-only monitors remain near chance. Performativity is task-difficulty-dependent (high on easy recall, low on hard multi-hop). Probe confidence is well-calibrated and enables early exit (68% token savings at 95% confidence).

**Relevance to hackathon:** Validates that activation probes fire earlier than CoT monitors -- exactly what the hackathon asks for ("monitors that fire as early as possible"). Use attention-pooling probes (not per-token linear probes) for long sequences. Target later layers (second half of the network).

---

### 8. Constitutional Classifiers++
**Cunningham et al. 2026** | [arXiv:2601.04603](https://arxiv.org/abs/2601.04603)

**Anthropic's production jailbreak-defense stack.** Key innovations:
- **Exchange classifiers** (evaluate outputs in context of inputs) defeat reconstruction and obfuscation attacks.
- **Two-stage cascade:** cheap Stage 1 screens all traffic; expensive Stage 2 evaluates only flagged ~10%.
- **Linear activation probes** with SWiM logit smoothing (window=16) and softmax-weighted loss (concentrates gradient on informative token positions). Probes cost ~377K FLOPs/token vs ~8B for a small classifier.
- **Probe+classifier ensemble:** probes and classifiers capture complementary failure modes; ensembling outperforms either alone.
- Result: 40x cheaper than baseline, 0.05% false-positive rate, only 1 high-risk vulnerability found in 1,736 hours of red-teaming.

**Relevance to hackathon:**
- **Level 1:** SWiM smoothing and softmax-weighted loss are directly applicable probe training techniques.
- The cascade architecture (probe first, classifier on uncertain cases) is the production deployment pattern.
- Probes using all layers (concatenated) beat single-layer probes.

---

### 9. Building Production-Ready Probes For Gemini
**Kramar et al. 2026** | [arXiv:2601.11516](https://arxiv.org/abs/2601.11516)

**The key production probes paper.** Probes fail on long-context inputs due to signal dilution in mean-pooling/softmax. Solutions:
- **MultiMax:** hard max per attention head (prevents dilution).
- **Rolling Attention:** attention within sliding windows of width 10, then max across windows.
- **AlphaEvolve:** automated architecture search rediscovered MultiMax-like solutions.
- Probes match Gemini 2.5 Flash at >10,000x lower cost. Long-context FNR drops from 87.9% (EMA probe) to 3.0% (Selected Probe).
- Cascading probe + LLM (using LLM only ~6-8% of the time) achieves lower error than LLM alone.
- Adaptive red teaming remains challenging: all methods have high FNR (20-50%).

**Relevance to hackathon:**
- **Level 1:** MultiMax and Rolling aggregation are state-of-the-art probe architectures. Architecture choice matters more than seed selection or data scaling.
- The weighted error metric (50x weight on overtriggering, 5x on FNR) is the production-relevant evaluation setup.
- Don't expect adversarial robustness -- probes are one layer of defense.

---

### 10. Claude Mythos Preview System Card
**Anthropic, April 2026**

Mythos is Anthropic's most capable frontier LLM, with autonomous zero-day exploitation capability. Not publicly released; offered only to vetted defensive cybersecurity partners (Project Glasswing). The cyber taxonomy (`benign / dual_use / high_risk_dual_use / prohibited`) drives real-time probe classifiers that monitor and block conversations. For general release: block prohibited always, often block high_risk_dual_use. The probe pattern (from Constitutional Classifiers) is reused across domains (cyber, CBRN).

**Relevance to hackathon:** This is the production system the hackathon's probes mirror. The 4-tier cyber taxonomy defines the Level 1 classification tasks. Probe classifiers run in real-time on conversational context, not just individual prompts.

---

### 11. Interpreting Language Model Parameters (VPD)
**Bushnaq, Braun, Sharkey et al. 2026 (Goodfire)** | [goodfire.ai/research](https://www.goodfire.ai/research/interpreting-lm-parameters)

Introduces **adversarial Parameter Decomposition (VPD)**, which decomposes model *weights* (not activations) into rank-one subcomponents, each implementing a small, interpretable part of the model's algorithm. Key innovation: adversarial ablation masks (gradient ascent to find worst-case ablations) ensure mechanistic faithfulness. Applied to a 67M-param, 4-layer model: 38,912 rank-one subcomponents, only ~10,000 alive, only ~205 active per token position. Pareto-dominates transcoders (PLTs/CLTs) at equivalent sparsity. **No feature splitting** -- component count stays stable as capacity increases (unlike SAEs/transcoders which scale linearly).

Critical finding: **attention computations are distributed across heads.** VPD reveals that "previous token attention" is implemented by shared rank-one Q/K subcomponents spanning *all* heads -- proving attention heads are not the right unit of analysis. Also demonstrates surgical model editing (gendered pronouns, bracket closing) via identified subcomponents.

**Relevance to hackathon:** Could decompose the attention heads that Arditi identified as writing to the refusal direction, revealing *how* they compute the refusal signal from input tokens. The adversarial faithfulness guarantee means if VPD says a subcomponent is unimportant for refusal, it truly is -- even under worst-case conditions. **Current limitation: only demonstrated at 67M scale**, not yet applicable to Gemma-31B/Qwen-27B. More of a "where the field is heading" reference than an immediately usable tool.

---

### 12. Polynomial Autoencoder (blog post)
**Ivan Pleshkov, May 2026 (Qdrant)** | [ivanpleshkov.dev](https://ivanpleshkov.dev/blog/polynomial-autoencoder) | [code](https://github.com/IvanPleshkov/poly-autoencoder)

A closed-form nonlinear compression method: PCA encoder + degree-2 polynomial decoder fit via Ridge regression. No SGD, no hyperparameters -- one `np.linalg.solve`. Captures the "cone effect" (nonlinear structure) in transformer embeddings that PCA misses. On BEIR/FiQA: 4x compression at only -0.85 p.p. NDCG vs raw (PCA loses -3.58 p.p.). The quadratic lift `[1, p_i, p_i*p_j]` enables a linear regression to fit quadratic structure.

**Relevance to hackathon (speculative but interesting):**
- **Quadratic probes:** Instead of a linear probe `y = w^T * activations`, use `y = w^T * polynomial_lift(activations)`. The quadratic terms `p_i * p_j` capture feature interactions (e.g., "negation" x "harmful" = "not harmful"). Still interpretable -- you can read off which interaction terms matter.
- **Detecting manifold leakage:** If a quadratic probe significantly beats a linear one, that's evidence the model encodes the concept via feature interaction rather than a simple direction -- challenging the Linear Representation Hypothesis.
- **Better SAE decoders:** Current SAEs use linear decoders. A polynomial decoder could achieve higher reconstruction fidelity with fewer features.
- **Caveat:** Need N >> M = (d+1)(d+2)/2 to avoid overfitting. At d=128, M=8385, so need ~50K+ training examples. The hackathon datasets are small (~1000-6000), so this works best on low-d projections of activations.

---

## Cross-Paper Synthesis

### The Big Picture

These papers form a coherent pipeline from **representation** to **detection** to **attribution** to **intervention**:

```
SAEs (Cunningham'23)          →  Interpretable feature space
  ↓
Circuits (Conmy'23, Marks'24) →  Which features/heads implement refusal
  ↓
Refusal Direction (Arditi'24) →  The single direction controlling refuse/comply
  ↓
Probes (Kramar'26, CC++'26)  →  Production-grade detectors using activations
  ↓
Tuned Lens (Belrose'23)      →  Early-layer prediction extraction
Reasoning Theater (Boppana'26)→  Probes fire before CoT monitors
  ↓
PRE (Xiong'25)               →  Probe-guided prompt rewriting to flip behavior
  ↓
Mythos System Card           →  The production deployment this all feeds into
```

### Key Convergent Findings

1. **Linear probes on activations are surprisingly powerful.** Arditi finds a single linear direction for refusal. CC++ deploys linear probes at ~377K FLOPs/token. Kramar shows probes match Gemini 2.5 Flash at 10,000x lower cost. You don't need complex nonlinear classifiers. *However*, Pleshkov's polynomial lift suggests a principled middle ground: quadratic probes that capture feature interactions while remaining closed-form and interpretable.

2. **Refusal is localized, not distributed.** A single direction (Arditi), a handful of attention heads, a small circuit. This concentration makes both detection (Level 1) and flipping (Level 2) tractable.

3. **Attribution back to input tokens is feasible.** Marks' attribution patching, Arditi's per-token refusal-direction contribution, and Xiong's probe-score-guided rewriting all converge on the same idea: you can identify *which input tokens* drive the probe signal, then edit them.

4. **Probes fire earlier than text-based monitors.** Boppana shows this for reasoning models; Belrose shows intermediate layers can be more informative than the final layer. The hackathon explicitly asks for early-firing monitors.

5. **Architecture matters more than data scaling for probes.** Kramar's MultiMax/Rolling aggregation, CC++'s SWiM smoothing, and Boppana's attention-pooling probes all show that the right aggregation mechanism dramatically outperforms naive approaches.

6. **The probe-rewrite loop works.** Xiong's PRE pipeline proves that optimizing prompts against a probe score causally changes model behavior, with the effect transferring across architectures.

---

## What We Can Do From Here

### Level 1 Strategy: Predict behavior from internals

**Recommended approach (fast, high-ceiling):**

1. **Extract the refusal direction** (Arditi's method) for Gemma 4-31B-it:
   - Collect ~128 refused + ~128 complied prompts from the provided dataset
   - Compute mean residual stream activations at each layer for each class
   - The difference-in-means vector is your probe
   - Evaluate: project held-out activations onto this direction, threshold, report AUC

2. **Train a proper linear probe** on residual stream activations:
   - Use the CC++ recipe: concatenate activations from multiple layers, SWiM smoothing (window=16), softmax-weighted BCE loss
   - Or use Kramar's Rolling Attention probe if sequence lengths vary
   - Train on the provided datasets (cyber probes + refusal probes)

3. **For the 5 tasks:**
   - Cyber Probes 1-3: train one probe per binary task on the cyber dataset
   - Refusal-Gemma / Refusal-Qwen: train per-model refusal probes

4. **Stretch: SAE features as probe inputs**
   - If SAEs are available for Gemma/Qwen, use SAE feature activations as the input to a linear classifier
   - This gives interpretable features (which ones fire for prohibited vs. benign?)

5. **Stretch: Quadratic (polynomial) probes** (Pleshkov's idea adapted)
   - Apply PCA to reduce activation dimensionality, then polynomial lift + Ridge regression
   - The quadratic terms `p_i * p_j` capture feature interactions that linear probes miss
   - If the quadratic probe beats linear significantly, you've found evidence of nonlinear concept encoding
   - Caveat: only works with enough data (N >> M). Use low-d PCA projections (d=32-64) to keep M manageable with the hackathon's dataset sizes

**Key implementation details:**
- Use `starter_code/extract_residuals.py` to get activations
- Focus on middle-to-later layers (Arditi: refusal direction peaks there; Boppana: later layers carry more decodable info)
- Report AUC with bootstrap 95% CI (n is small, ~60-80 for refusal)

### Level 2 Strategy: Flip behavior with minimal edits

**Recommended approach (the PRE loop adapted for refusal):**

1. **Attribution** -- find which input tokens contribute most to the refusal signal:
   - **Option A (fast):** Compute per-token projection onto the refusal direction. Tokens with highest projection are driving refusal.
   - **Option B (precise):** Attribution patching (Marks' IE_atp) -- one forward + one backward pass per prompt.
   - **Option C (if SAEs available):** Identify which SAE features fire on which tokens, cross-reference with the refusal circuit.

2. **Edit generation** -- propose minimal natural-language edits:
   - Use an LLM (MiniMax via AIaaS) as a rewriter, targeting the attributed tokens/phrases
   - Generate k=5-7 candidate rewrites per prompt
   - System prompt should instruct: preserve the technical request, change only the flagged tokens/phrases, maintain naturalness

3. **Scoring** -- rank candidates with the probe:
   - Score each candidate with your Level-1 probe
   - Select the one that maximally flips the probe score

4. **Verification** -- confirm the model actually flips:
   - Run the selected edit through the actual model (Gemma 4-31B-it)
   - Check if behavior flipped (refusal → compliance)
   - Report all three quantities: Pr(f flipped | edit), Pr(model flipped | edit), Pr(model flipped | f flipped)

5. **Iterate** -- up to 5 iterations per prompt (reference scaffold):
   - If the probe flipped but the model didn't, your probe is being gamed -- re-attribute and try different edits
   - If neither flipped, try broader edits or different attribution method

**Key engineering decisions:**
- Focus on **Gemma 4-31B-it** (100% reproducibility on cluster, 93% on AIaaS)
- Keep edits ≤25 token changes (the "minimal edit" budget)
- Use the intent judge (MiniMax, score ≥ 7/10) to ensure edits preserve the original ask
- A perplexity filter catches gibberish/GCG-style attacks

### Time Budget Suggestion (for the hackathon)

| Block | Task | Hours |
|-------|------|-------|
| 1 | Extract residuals, train refusal-direction probe, get baseline AUCs | 2-3h |
| 2 | Train proper linear probes (CC++ style) for all 5 tasks | 2-3h |
| 3 | Implement attribution (per-token refusal contribution) | 1-2h |
| 4 | Build the edit loop (rewriter + probe scoring + verification) | 3-4h |
| 5 | Run Level-2 evaluation, collect metrics, error bars | 2-3h |
| 6 | Prepare slides with worked examples | 1-2h |

### What Would Make a Strong Submission

1. **High AUC on all 5 tasks** with proper error bars (Wilson 95% CI)
2. **A clear causal story**: attribution → edit → probe flip → model flip, with concordance (Pr(model|f)) significantly above chance
3. **Worked examples**: show original prompt, attributed tokens, edited prompt, before/after probe scores, before/after model behavior
4. **Honesty about limitations**: if your probe is gameable (high Pr(f|edit) but low Pr(model|edit)), say so -- it's a finding, not a failure
5. **Connecting to production**: reference the Mythos system card, CC++, Kramar -- your probe is a toy version of what's deployed
