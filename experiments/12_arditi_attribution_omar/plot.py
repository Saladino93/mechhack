"""Plot summaries for the Arditi per-token attribution.

Produces:
  - per_prompt_attribution_strip.png
      For each of the 81 prompts: a horizontal strip showing per-token
      Arditi attribution as a heatmap (x=token position, color=score).
      Lets you eyeball where refusal triggers cluster within each prompt.

  - score_distribution.png
      Two-panel histogram: distribution of per-prompt max attribution and
      distribution of per-prompt min attribution across the 81 prompts.

  - top_tokens_overall.png
      Bar chart of the 30 most frequently-occurring "top-K" tokens across
      the 81 prompts, with bar height = mean attribution score when that
      token appears as a top-K position. Shows the *recurring* refusal
      triggers (e.g. 'simulates', 'detection', 'forensic').

  - rank_vs_score.png
      For each prompt's top 8 tokens, sorted by absolute score: shows how
      quickly the score drops from rank 1 → rank 8 (averaged across prompts).
      Tight curve = a few outlier tokens dominate; flat = the signal is
      distributed.

CPU only.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ATTRS = HERE / "attributions.jsonl"
TOP_TSV = HERE / "top_tokens.tsv"


def load_rows():
    return [json.loads(line) for line in ATTRS.read_text().splitlines() if line.strip()]


def fig_per_prompt_strip(rows, out_path):
    """Heatmap: rows = prompts, x = token position (normalised to [0,1]), color = attribution.

    We bin each prompt's 0..n_tokens range into 200 buckets, take the max
    absolute attribution per bucket. Visual story: across-prompt position
    bias of refusal-trigger tokens.
    """
    n = len(rows)
    n_buckets = 200
    M = np.zeros((n, n_buckets), dtype=np.float32)
    for i, r in enumerate(rows):
        n_tok = r["n_tokens_in_scope"]
        if n_tok == 0:
            continue
        for t in r["top_tokens"]:
            pos = t["position"]
            score = t["score"]
            b = min(int(pos / max(1, n_tok) * n_buckets), n_buckets - 1)
            if abs(score) > abs(M[i, b]):
                M[i, b] = score
    # Sort rows by max-token position (so the strip is visually ordered)
    sort_idx = np.argsort([np.argmax(np.abs(M[i])) for i in range(n)])
    M = M[sort_idx]
    fig, ax = plt.subplots(figsize=(11, 6))
    vmax = float(np.percentile(np.abs(M), 99))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest",
                   extent=[0.0, 1.0, n - 0.5, -0.5])
    ax.set_xlabel("Token position (normalised: 0 = start, 1 = end of prompt)")
    ax.set_ylabel("Prompt (sorted by position of max |Arditi score|)")
    ax.set_xticks(np.linspace(0.0, 1.0, 11))
    ax.set_title("Per-token Arditi attribution across 81 attribution_eval prompts\n"
                 "(top-8 tokens per prompt, bucketed; red=refusal-pulling, blue=compliance-pulling)")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Arditi-direction projection (residual · refusal_dir)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_score_distribution(rows, out_path):
    maxes = np.array([r["attribution_max"] for r in rows])
    mins = np.array([r["attribution_min"] for r in rows])
    means_abs = np.array([r["mean_abs_attribution"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].hist(maxes, bins=20, color="tab:red", alpha=0.85, edgecolor="black")
    axes[0].set_title(f"Per-prompt max attribution\n"
                      f"(mean={maxes.mean():.1f}, range {maxes.min():.1f}-{maxes.max():.1f})")
    axes[0].set_xlabel("Max Arditi score in prompt")
    axes[0].set_ylabel("# prompts")

    axes[1].hist(mins, bins=20, color="tab:blue", alpha=0.85, edgecolor="black")
    axes[1].set_title(f"Per-prompt min attribution\n"
                      f"(mean={mins.mean():.2f}, range {mins.min():.2f}-{mins.max():.2f})")
    axes[1].set_xlabel("Min Arditi score in prompt")

    axes[2].hist(means_abs, bins=20, color="tab:purple", alpha=0.85, edgecolor="black")
    axes[2].set_title(f"Per-prompt mean |attribution|\n"
                      f"(mean={means_abs.mean():.1f})")
    axes[2].set_xlabel("Mean |Arditi score| in prompt")

    fig.suptitle(f"Arditi attribution score distributions across {len(rows)} attribution_eval prompts",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_top_tokens_overall(rows, out_path, top_n=30):
    """Aggregate top-K tokens across prompts, sorted by frequency × mean score."""
    by_token = defaultdict(list)
    for r in rows:
        for t in r["top_tokens"]:
            tok = t["token"].strip()  # de-noise leading/trailing whitespace
            if not tok or tok in ("-", ":", ",", ".", "(", ")"):
                continue
            by_token[tok].append(t["score"])

    # Rank by frequency then mean score
    ranked = sorted(by_token.items(), key=lambda kv: (-len(kv[1]), -np.mean(kv[1])))
    top = ranked[:top_n]
    labels = [t for t, _ in top]
    counts = [len(s) for _, s in top]
    means = [float(np.mean(s)) for _, s in top]

    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(labels))
    ax.barh(y, counts, color="tab:red", alpha=0.85, edgecolor="black")
    for i, (c, m) in enumerate(zip(counts, means)):
        ax.text(c + 0.3, i, f"  mean={m:+.0f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("# of attribution_eval prompts where this token is in the top-8")
    ax.set_title(f"Recurring Arditi-attribution top-tokens across {len(rows)} prompts\n"
                 "(each row: frequency, plus mean Arditi score across appearances)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_rank_vs_score(rows, out_path):
    """For each prompt, sort top tokens by |score| desc → average across prompts.
    Shows whether attribution is concentrated or distributed."""
    K = max(len(r["top_tokens"]) for r in rows)
    M = np.full((len(rows), K), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        sorted_t = sorted(r["top_tokens"], key=lambda t: -abs(t["score"]))
        for j, t in enumerate(sorted_t):
            M[i, j] = abs(t["score"])

    means = np.nanmean(M, axis=0)
    stds = np.nanstd(M, axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(1, K + 1)
    ax.plot(x, means, marker="o", color="tab:purple")
    ax.fill_between(x, means - stds, means + stds, alpha=0.2, color="tab:purple")
    ax.set_xlabel("Rank within a prompt's top tokens (1 = most attributing)")
    ax.set_ylabel("|Arditi score| (mean ± 1σ across 81 prompts)")
    ax.set_title("How quickly per-prompt attribution decays from rank 1 to rank K\n"
                 "(flat curve = attribution distributed across many tokens; "
                 "steep = a few outliers dominate)")
    ax.grid(alpha=0.3)
    ax.set_xticks(x)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    rows = load_rows()
    print(f"Loaded {len(rows)} prompts.")
    fig_per_prompt_strip(rows, HERE / "per_prompt_attribution_strip.png")
    fig_score_distribution(rows, HERE / "score_distribution.png")
    fig_top_tokens_overall(rows, HERE / "top_tokens_overall.png")
    fig_rank_vs_score(rows, HERE / "rank_vs_score.png")


if __name__ == "__main__":
    main()
