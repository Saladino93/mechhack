# Experiments

Each experiment gets a numbered folder: `01_name/`, `02_name/`, etc.

Every experiment folder should contain:
- A notebook or script with the experiment code
- A `notes.md` with: goal, method, results, takeaways
- Any output artifacts (plots, metrics, saved weights)

## Index

| # | Name | Goal | Status |
|---|------|------|--------|
| 01 | `01_data_exploration` | Understand dataset distributions, token lengths, class balance | **Done** |
| 02 | `02_extract_residuals` | Extract residual activations from Gemma/Qwen | Pending |
| 03 | `03_refusal_direction` | Find refusal direction via difference-in-means (Arditi method) | Pending |
| 04 | `04_linear_probe` | Train linear probes on residuals for all 5 tasks | Pending |
| 05 | `05_advanced_probes` | SWiM smoothing, attention-pooling, multi-layer concat | Pending |
| 06 | `06_attribution` | Per-token attribution (grad x input, refusal-direction projection) | Pending |
| 07 | `07_edit_loop` | Iterative edit agent: attribute -> rewrite -> re-score | Pending |
| 08 | `08_eval_behavior` | Full Level-2 eval with behavior verification | Pending |
