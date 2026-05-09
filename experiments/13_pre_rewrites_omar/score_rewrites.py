"""Level 2 PRE step — score the 7 pre-generated rewrites per attribution_eval prompt
with two probes side-by-side (Arditi direction + LR probe), pick the best rewrite per
sample (lowest refusal score), and dump everything for downstream concordance.

Pipeline:
  1. Load 832 Gemma refusal extracts → fit one full-data LR probe on mean-pool
     features at layer 32 (probe coefficients saved as `lr_probe.npz`).
  2. Load `arditi_direction.npy` from exp 11.
  3. Load Gemma 4-31B-it.
  4. For each (sample_id, rewrite_idx) in `rewrites_k7.json` and for each ORIGINAL
     attack_prompt: tokenize with chat template, run Gemma forward with a hook on
     layer 32, mean-pool the residuals over the attended positions, score with
     both probes.
  5. Save:
       - `scores.jsonl`  : (sample_id, rewrite_idx or "original", arditi_score,
                           lr_prob, n_tokens)
       - `best_rewrites.json` : per sample_id, the best rewrite by each scorer +
                                 the rewrite text itself, ready for Phase 3 (Gemma
                                 rollout + judge).
       - `summary.json`  : aggregate flip counts and probe-vs-probe agreement.

Long prompts (up to 8k tokens). Uses chunked SDPA for Gemma's wide head_dim=512.
"""
from __future__ import annotations
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "starter_code"))
from chunked_sdpa import chunked_sdpa_scope  # noqa: E402

ARDITI_PATH = REPO_ROOT / "experiments" / "11_refusal_probe_omar" / "arditi_direction.npy"
EXTRACTS_DIR = Path("/home/ubuntu/extracts/gemma_refusal_L32/gemma4_31b")
REWRITES_PATH = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "rewrites_k7.json"
ATTRS_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"
MODEL_PATH = Path("/home/ubuntu/models/Gemma-4-31B-it")

OUT_DIR = Path(__file__).parent
SCORES_PATH = OUT_DIR / "scores.jsonl"
BEST_PATH = OUT_DIR / "best_rewrites.json"
SUMMARY_PATH = OUT_DIR / "summary.json"
LR_PROBE_PATH = OUT_DIR / "lr_probe.npz"
TARGET_LAYER = 32


def append_jsonl(path: Path, obj):
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


def fit_lr_probe():
    """Refit a logistic regression probe on all 832 refusal extracts (mean-pool, L32).

    Saves coef_ + intercept_ to lr_probe.npz so we can score new rewrites with
    `sigmoid(W @ x + b)` without sklearn.
    """
    print("[1/4] Fitting full-data LR probe on 832 refusal extracts...", flush=True)
    means, labels, ids = [], [], []
    bad = 0
    for p in sorted(EXTRACTS_DIR.glob("*.pt")):
        ex = torch.load(str(p), weights_only=False)
        residuals = ex["residuals"]
        if residuals.dim() == 3 and residuals.shape[0] == 1:
            residuals = residuals.squeeze(0)
        residuals = residuals.float()
        mask = ex["attention_mask"].bool().squeeze()
        n = int(mask.sum().item())
        if n == 0:
            bad += 1
            continue
        m_f = mask.float().unsqueeze(-1)
        feat = ((residuals * m_f).sum(dim=0) / n).numpy()
        if not np.isfinite(feat).all():
            bad += 1
            continue
        means.append(feat)
        labels.append(int(ex["label"]))
        ids.append(p.stem)
    X = np.stack(means).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    print(f"  loaded {len(y)} ok, {bad} bad. pos={int((y==1).sum())} neg={int((y==0).sum())}", flush=True)

    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    clf.fit(X, y)
    coef = clf.coef_[0].astype(np.float32)   # (d,)
    bias = float(clf.intercept_[0])
    np.savez(LR_PROBE_PATH, coef=coef, bias=bias, sample_ids=np.asarray(ids))
    train_proba = clf.predict_proba(X)[:, 1]
    train_acc = float(((train_proba > 0.5).astype(int) == y).mean())
    print(f"  saved {LR_PROBE_PATH} (train acc = {train_acc:.4f}, |coef|_2 = {np.linalg.norm(coef):.2f})", flush=True)
    return coef, bias


def load_arditi():
    arr = np.load(ARDITI_PATH).astype(np.float32)
    print(f"[2/4] Arditi direction loaded: shape={arr.shape}, norm={np.linalg.norm(arr):.4f}", flush=True)
    return arr


def load_gemma():
    print(f"[3/4] Loading Gemma 4-31B-it from {MODEL_PATH}...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.float16, device_map="cuda:0",
        attn_implementation="sdpa", trust_remote_code=True,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s, dtype=fp16", flush=True)
    return model, tok


@torch.no_grad()
def gemma_residual_at_layer(model, tok, prompt, layer_idx, use_chat_template=True):
    """Returns the (n_tokens, d) fp32 residual at hidden-state index `layer_idx`,
    plus the attention_mask (n_tokens,). Uses `output_hidden_states=True` so we
    don't have to know the model's internal module-path (Gemma 4 uses
    `model.language_model.layers`, not `model.model.layers`).

    Matches extract_residuals.py's indexing: hidden_states[0] is the embedding
    output, hidden_states[i] for i ∈ [1..n_layers] is the post-block residual.
    """
    if use_chat_template:
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    else:
        text = prompt
    enc = tok(text, return_tensors="pt", truncation=True, max_length=8192)
    input_ids = enc["input_ids"].to("cuda:0")
    attn_mask = enc["attention_mask"].to("cuda:0")

    with chunked_sdpa_scope():
        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

    h = out.hidden_states[layer_idx]  # (1, n_tok, d)
    res = h.squeeze(0).float().cpu()  # (n_tok, d)
    mask = attn_mask.squeeze(0).bool().cpu()
    return res, mask


def mean_pool(res, mask):
    n = int(mask.sum().item())
    if n == 0:
        return None
    feat = (res[mask].float()).mean(dim=0).numpy()  # (d,)
    if not np.isfinite(feat).all():
        return None
    return feat


def score_one(feat, arditi, lr_coef, lr_bias):
    arditi_score = float(feat @ arditi)
    logit = float(feat @ lr_coef + lr_bias)
    lr_prob = 1.0 / (1.0 + np.exp(-logit))
    return arditi_score, float(lr_prob)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if SCORES_PATH.exists():
        SCORES_PATH.unlink()  # restart fresh

    lr_coef, lr_bias = fit_lr_probe()
    arditi = load_arditi()
    model, tok = load_gemma()

    print(f"[4/4] Scoring rewrites + originals...", flush=True)
    rewrites = json.loads(REWRITES_PATH.read_text())
    by_sample = {r["sample_id"]: r["rewrites"] for r in rewrites}
    sample_ids = list(by_sample.keys())
    print(f"  {len(sample_ids)} samples × 7 rewrites = {sum(len(v) for v in by_sample.values())} rewrites", flush=True)

    # Originals: keyed by sample_id from attacks_full.jsonl
    originals = {}
    with ATTRS_FULL.open() as f:
        for line in f:
            r = json.loads(line)
            originals[r["sample_id"]] = r["attack_prompt"]

    t_total = time.time()
    rows = []
    for si, sid in enumerate(sample_ids):
        # Score the original first (anchor for delta)
        for label, prompt in [("original", originals.get(sid))] + [
            (f"rewrite_{j}", rw) for j, rw in enumerate(by_sample[sid])
        ]:
            if prompt is None:
                continue
            t0 = time.time()
            try:
                res, mask = gemma_residual_at_layer(model, tok, prompt, TARGET_LAYER)
            except torch.cuda.OutOfMemoryError as e:
                print(f"  [OOM] {sid}/{label}: {e}", flush=True)
                torch.cuda.empty_cache(); gc.collect()
                continue
            feat = mean_pool(res, mask)
            del res
            if feat is None:
                print(f"  [skip-nan] {sid}/{label}", flush=True)
                continue
            arditi_s, lr_p = score_one(feat, arditi, lr_coef, lr_bias)
            row = {
                "sample_id": sid,
                "label": label,
                "n_tokens": int(mask.sum().item()),
                "arditi_score": arditi_s,
                "lr_prob": lr_p,
                "elapsed_s": round(time.time() - t0, 2),
            }
            rows.append(row)
            append_jsonl(SCORES_PATH, row)
        if (si + 1) % 5 == 0 or si == len(sample_ids) - 1:
            print(f"  [{si+1}/{len(sample_ids)}] {sid}: total elapsed {time.time()-t_total:.0f}s", flush=True)

    # Aggregate: for each sample_id, find best rewrite per scorer
    best = {}
    by_sid = {}
    for r in rows:
        by_sid.setdefault(r["sample_id"], []).append(r)
    arditi_flips = 0
    lr_flips = 0
    agree_best = 0
    n_with_orig = 0
    for sid, runs in by_sid.items():
        original = next((r for r in runs if r["label"] == "original"), None)
        rw_runs = [r for r in runs if r["label"].startswith("rewrite_")]
        if not rw_runs:
            continue
        best_arditi = min(rw_runs, key=lambda r: r["arditi_score"])
        best_lr = min(rw_runs, key=lambda r: r["lr_prob"])
        b = {
            "sample_id": sid,
            "best_by_arditi": {
                "label": best_arditi["label"],
                "arditi_score": best_arditi["arditi_score"],
                "lr_prob": best_arditi["lr_prob"],
                "rewrite_text": by_sample[sid][int(best_arditi["label"].split("_")[1])],
            },
            "best_by_lr": {
                "label": best_lr["label"],
                "arditi_score": best_lr["arditi_score"],
                "lr_prob": best_lr["lr_prob"],
                "rewrite_text": by_sample[sid][int(best_lr["label"].split("_")[1])],
            },
        }
        if original is not None:
            n_with_orig += 1
            b["original"] = {
                "arditi_score": original["arditi_score"],
                "lr_prob": original["lr_prob"],
            }
            if best_arditi["arditi_score"] < original["arditi_score"]:
                arditi_flips += 1
            if best_lr["lr_prob"] < 0.5:
                lr_flips += 1
        if best_arditi["label"] == best_lr["label"]:
            agree_best += 1
        best[sid] = b

    BEST_PATH.write_text(json.dumps(best, indent=2))
    summary = {
        "n_samples": len(by_sid),
        "n_with_original_baseline": n_with_orig,
        "arditi_flips_vs_original": arditi_flips,
        "lr_prob_below_0.5": lr_flips,
        "best_rewrite_agreement_arditi_lr": agree_best,
        "elapsed_total_s": round(time.time() - t_total, 1),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v}", flush=True)
    print(f"\nwrote {SCORES_PATH}, {BEST_PATH}, {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
