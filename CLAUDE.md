# CLAUDE.md — Mechhack Project Guide

## What is this?

Red-team mechanistic interpretability hackathon. Two levels:
- **Level 1 (Predict):** Build `f: model_internals -> label` for 5 binary tasks (3 cyber-harm probes + 2 refusal probes). Metric: AUC.
- **Level 2 (Flip):** Use Level-1 `f` to attribute signal to input tokens, make minimal natural-language edits that flip the model's behavior. Report: Pr(f|edit), Pr(model|edit), Pr(model|f).

Target models: Gemma 4-31B-it (primary, trusted), Qwen 3.6-27B (secondary).

## Experiment workflow

All work goes in `experiments/NN_name/` folders, numbered sequentially.

### Creating a new experiment

```bash
# Example: creating experiment 03
mkdir -p experiments/03_refusal_direction
```

Each experiment folder must contain:
- **`notes.md`** — Goal, method, results, takeaways (filled in as you go)
- **Script(s) or notebook(s)** — The actual code
- **Output artifacts** — Plots (`.png`), metrics (`.json`), saved weights (`.pt`)

### notes.md template

```markdown
# Experiment NN: Title

## Goal
What question are we answering?

## Method
What approach are we taking and why?

## Results
Key numbers, plots, observations.

## Takeaways
What did we learn? What's next?
```

### Running experiments

Scripts should be runnable from the repo root:
```bash
python experiments/01_data_exploration/explore.py
```

Use relative paths from repo root for datasets:
```python
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS = REPO_ROOT / "datasets"
```

## Key directories

```
datasets/              # DO NOT MODIFY — fixed train/test splits
  cyber_probes/        # 7259 rollouts, 4 categories
  refusal_probes/      # per-model (gemma4_31b, qwen36)
starter_code/          # Provided baselines — read, extend, don't overwrite
  extract_residuals.py # Activation extraction (GPU required)
  train_probe.py       # Probe training (linear/MLP/attention)
  grad_input_baseline.py # Attribution baseline
  iterative_edit_agent.py # Level-2 edit loop baseline
scoring/               # Official scoring scripts
experiments/           # Our work — numbered folders
papers/                # Reference papers + SUMMARY.md with strategy
```

## Build & run

### Dependencies (on-cluster pod)
All deps are pre-installed in the pod image (`v9`). No pip install needed.

### Dependencies (local / off-cluster)
```bash
pip install torch transformers numpy scikit-learn matplotlib seaborn httpx huggingface-hub hf_transfer
```

### Models
- On cluster: pre-staged at `/data/Gemma-4-31B-it` and `/data/Qwen3.6-27B`
- Off cluster: `python starter_code/download_models.py --out_dir ./models` (needs `HF_TOKEN`, ~117 GB)

### Common commands
```bash
# Extract residuals (GPU required)
python starter_code/extract_residuals.py --model_key gemma4_31b --layers middle --sample_limit 10

# Train probes
python starter_code/train_probe.py --extracts_dir ./extracts/gemma4_31b --manifest ./extracts/gemma4_31b/extraction_metadata.json --out_dir ./probes --task refusal_gemma4_31b

# Score Level 1
python scoring/score_probes.py --submission_dir <your_submission>

# Score Level 2
python scoring/score_disrupt.py --submission <your_submission> --model_key gemma4_31b
```

## Conventions

- **Splits are fixed.** `hash(sample_id) mod 100 < 70 -> train`. Never redefine them.
- **Always report error bars.** Wilson 95% CI for proportions, bootstrap for AUC.
- **Gemma is the trusted target** for Level 2 (100% reproducibility on cluster).
- **Plots go in the experiment folder** as `.png` files.
- **Metrics go in `results.json`** inside the experiment folder.
