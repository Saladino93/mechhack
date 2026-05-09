"""Experiment 01: Data Exploration

Produces summary stats and distribution plots for all datasets.
Run from repo root:
    python experiments/01_data_exploration/explore.py
"""
import json
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS = REPO_ROOT / "datasets"
OUT_DIR = Path(__file__).resolve().parent

# Try importing plotting libraries — graceful fallback if not available
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warn] matplotlib not installed — skipping plots")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Cyber probes ──────────────────────────────────────────────
def explore_cyber():
    print_header("CYBER PROBES")

    manifest = json.load(open(DATASETS / "cyber_probes" / "manifest.json"))
    print(f"\nManifest: {json.dumps(manifest, indent=2)}")

    train = load_jsonl(DATASETS / "cyber_probes" / "train.jsonl")
    test = load_jsonl(DATASETS / "cyber_probes" / "test.jsonl")
    all_data = train + test

    print(f"\nTotal: {len(all_data)} | Train: {len(train)} | Test: {len(test)}")

    # Category distribution
    cat_counts = Counter(r["category"] for r in all_data)
    cat_train = Counter(r["category"] for r in train)
    cat_test = Counter(r["category"] for r in test)

    print("\nCategory distribution:")
    print(f"  {'Category':<25} {'Total':>6} {'Train':>6} {'Test':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6}")
    for cat in ["benign", "dual_use", "high_risk_dual_use", "prohibited"]:
        print(f"  {cat:<25} {cat_counts[cat]:>6} {cat_train[cat]:>6} {cat_test[cat]:>6}")

    # Token length stats
    token_lengths = [r.get("n_tokens", 0) for r in all_data]
    char_lengths = [r.get("n_chars", 0) for r in all_data]
    print(f"\nToken lengths: min={min(token_lengths)} max={max(token_lengths)} "
          f"mean={sum(token_lengths)/len(token_lengths):.0f} "
          f"median={sorted(token_lengths)[len(token_lengths)//2]}")
    print(f"Char lengths:  min={min(char_lengths)} max={max(char_lengths)} "
          f"mean={sum(char_lengths)/len(char_lengths):.0f}")

    # Per-category token length stats
    print("\nPer-category token lengths:")
    for cat in ["benign", "dual_use", "high_risk_dual_use", "prohibited"]:
        toks = [r["n_tokens"] for r in all_data if r["category"] == cat]
        print(f"  {cat:<25} mean={sum(toks)/len(toks):>7.0f}  "
              f"median={sorted(toks)[len(toks)//2]:>5}  "
              f"min={min(toks):>4}  max={max(toks):>5}")

    # Probe task compositions
    print("\nProbe task compositions (test split):")
    for probe_id, pos, neg_cats in [
        (1, "dual_use", ["benign"]),
        (2, "high_risk_dual_use", ["dual_use", "benign"]),
        (3, "prohibited", ["high_risk_dual_use", "dual_use", "benign"]),
    ]:
        n_pos = sum(1 for r in test if r["category"] == pos)
        n_neg = sum(1 for r in test if r["category"] in neg_cats)
        print(f"  Probe-{probe_id}: pos={n_pos} ({pos}) neg={n_neg} ({'+'.join(neg_cats)}) "
              f"ratio={n_pos/(n_pos+n_neg):.2f}")

    # Example prompts (first 200 chars)
    print("\nExample prompts (first 200 chars):")
    for cat in ["benign", "dual_use", "high_risk_dual_use", "prohibited"]:
        sample = next(r for r in all_data if r["category"] == cat)
        prompt_preview = sample["prompt"][:200].replace("\n", " ")
        print(f"\n  [{cat}] {sample['sample_id']}:")
        print(f"    {prompt_preview}...")

    # Source distribution
    sources = Counter(r.get("source", "unknown") for r in all_data)
    print(f"\nSource distribution: {dict(sources)}")

    return all_data, train, test


# ── Refusal probes ────────────────────────────────────────────
def explore_refusal(model_key):
    print_header(f"REFUSAL PROBES — {model_key}")

    base = DATASETS / "refusal_probes" / model_key
    train = load_jsonl(base / "train_split.jsonl")
    test = load_jsonl(base / "test_split.jsonl")
    full = load_jsonl(base / "attacks_full.jsonl")
    attrib_eval = load_jsonl(base / "attribution_eval.jsonl")

    print(f"\nFull: {len(full)} | Train: {len(train)} | Test: {len(test)} | Attribution eval: {len(attrib_eval)}")

    # Refusal distribution
    for name, data in [("Full", full), ("Train", train), ("Test", test), ("Attrib eval", attrib_eval)]:
        n_ref = sum(1 for r in data if r["is_refusal"])
        n_comp = sum(1 for r in data if not r["is_refusal"])
        print(f"  {name:<12} refusal={n_ref:>4} compliance={n_comp:>4} "
              f"refusal_rate={n_ref/len(data):.2f}")

    # Token lengths
    token_lengths = [r["n_tokens"] for r in full]
    print(f"\nToken lengths: min={min(token_lengths)} max={max(token_lengths)} "
          f"mean={sum(token_lengths)/len(token_lengths):.0f} "
          f"median={sorted(token_lengths)[len(token_lengths)//2]}")

    # Refusal vs compliance token lengths
    ref_toks = [r["n_tokens"] for r in full if r["is_refusal"]]
    comp_toks = [r["n_tokens"] for r in full if not r["is_refusal"]]
    print(f"  Refusal tokens:    mean={sum(ref_toks)/len(ref_toks):.0f} median={sorted(ref_toks)[len(ref_toks)//2]}")
    print(f"  Compliance tokens: mean={sum(comp_toks)/len(comp_toks):.0f} median={sorted(comp_toks)[len(comp_toks)//2]}")

    # Judge score distribution
    scores = Counter(r.get("judge_score") for r in full)
    print(f"\nJudge score distribution: {dict(sorted((k, v) for k, v in scores.items() if k is not None))}")
    if None in scores:
        print(f"  (None scores: {scores[None]})")

    # Attribution eval details
    print(f"\nAttribution eval set ({len(attrib_eval)} samples):")
    ae_toks = [r["n_tokens"] for r in attrib_eval]
    print(f"  Token lengths: min={min(ae_toks)} max={max(ae_toks)} "
          f"mean={sum(ae_toks)/len(ae_toks):.0f}")
    print(f"  All refusals: {all(r['is_refusal'] for r in attrib_eval)}")

    # Example
    sample = full[0]
    print(f"\nExample ({sample['sample_id']}, refusal={sample['is_refusal']}):")
    print(f"  Prompt (first 200 chars): {sample['attack_prompt'][:200].replace(chr(10), ' ')}...")
    if sample.get("response"):
        print(f"  Response (first 200 chars): {sample['response'][:200].replace(chr(10), ' ')}...")

    return full, train, test, attrib_eval


# ── Plots ─────────────────────────────────────────────────────
def plot_cyber_distributions(all_data):
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Category counts
    cats = ["benign", "dual_use", "high_risk_dual_use", "prohibited"]
    cat_short = ["benign", "dual_use", "high_risk\ndual_use", "prohibited"]
    counts = [sum(1 for r in all_data if r["category"] == c) for c in cats]
    colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]
    axes[0].bar(cat_short, counts, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_title("Cyber Probes: Category Distribution")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts):
        axes[0].text(i, v + 20, str(v), ha="center", fontsize=9)

    # 2. Token length histogram by category
    for cat, color in zip(cats, colors):
        toks = [r["n_tokens"] for r in all_data if r["category"] == cat]
        axes[1].hist(toks, bins=50, alpha=0.5, label=cat, color=color)
    axes[1].set_title("Token Length Distribution by Category")
    axes[1].set_xlabel("n_tokens")
    axes[1].set_ylabel("Count")
    axes[1].legend(fontsize=8)

    # 3. Probe task class balance (test split)
    test_data = [r for r in all_data if r["split"] == "test"]
    probe_labels = []
    for pid, pos, neg_cats in [
        (1, "dual_use", ["benign"]),
        (2, "high_risk_dual_use", ["dual_use", "benign"]),
        (3, "prohibited", ["high_risk_dual_use", "dual_use", "benign"]),
    ]:
        n_pos = sum(1 for r in test_data if r["category"] == pos)
        n_neg = sum(1 for r in test_data if r["category"] in neg_cats)
        probe_labels.append((f"Probe-{pid}", n_pos, n_neg))

    x = range(len(probe_labels))
    width = 0.35
    axes[2].bar([i - width/2 for i in x], [p[1] for p in probe_labels],
                width, label="Positive", color="#F44336", alpha=0.7)
    axes[2].bar([i + width/2 for i in x], [p[2] for p in probe_labels],
                width, label="Negative", color="#2196F3", alpha=0.7)
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels([p[0] for p in probe_labels])
    axes[2].set_title("Probe Task Class Balance (Test)")
    axes[2].set_ylabel("Count")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "cyber_distributions.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUT_DIR / 'cyber_distributions.png'}")
    plt.close()


def plot_refusal_distributions(gemma_full, qwen_full):
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Refusal vs compliance counts
    models = ["Gemma 4-31B", "Qwen 3.6-27B"]
    ref_counts = [sum(1 for r in d if r["is_refusal"]) for d in [gemma_full, qwen_full]]
    comp_counts = [sum(1 for r in d if not r["is_refusal"]) for d in [gemma_full, qwen_full]]
    x = range(len(models))
    width = 0.35
    axes[0].bar([i - width/2 for i in x], ref_counts, width,
                label="Refusal", color="#F44336", alpha=0.7)
    axes[0].bar([i + width/2 for i in x], comp_counts, width,
                label="Compliance", color="#4CAF50", alpha=0.7)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(models)
    axes[0].set_title("Refusal vs Compliance")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # 2. Token length by refusal status (Gemma)
    ref_toks = [r["n_tokens"] for r in gemma_full if r["is_refusal"]]
    comp_toks = [r["n_tokens"] for r in gemma_full if not r["is_refusal"]]
    axes[1].hist(ref_toks, bins=40, alpha=0.5, label="Refusal", color="#F44336")
    axes[1].hist(comp_toks, bins=40, alpha=0.5, label="Compliance", color="#4CAF50")
    axes[1].set_title("Gemma: Token Length by Outcome")
    axes[1].set_xlabel("n_tokens")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    # 3. Judge score distribution (Gemma)
    scores_ref = [r.get("judge_score", 0) for r in gemma_full if r["is_refusal"]]
    scores_comp = [r.get("judge_score", 0) for r in gemma_full if not r["is_refusal"]]
    bins = range(0, 12)
    axes[2].hist(scores_ref, bins=bins, alpha=0.5, label="Refusal", color="#F44336")
    axes[2].hist(scores_comp, bins=bins, alpha=0.5, label="Compliance", color="#4CAF50")
    axes[2].set_title("Gemma: Judge Score Distribution")
    axes[2].set_xlabel("Judge Score")
    axes[2].set_ylabel("Count")
    axes[2].xaxis.set_major_locator(ticker.MultipleLocator(1))
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "refusal_distributions.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUT_DIR / 'refusal_distributions.png'}")
    plt.close()


# ── Summary JSON ──────────────────────────────────────────────
def save_summary(cyber_all, gemma_full, qwen_full):
    summary = {
        "cyber": {
            "total": len(cyber_all),
            "categories": dict(Counter(r["category"] for r in cyber_all)),
            "token_length_mean": sum(r["n_tokens"] for r in cyber_all) / len(cyber_all),
            "token_length_median": sorted(r["n_tokens"] for r in cyber_all)[len(cyber_all)//2],
        },
        "refusal_gemma": {
            "total": len(gemma_full),
            "refusal_count": sum(1 for r in gemma_full if r["is_refusal"]),
            "compliance_count": sum(1 for r in gemma_full if not r["is_refusal"]),
            "refusal_rate": sum(1 for r in gemma_full if r["is_refusal"]) / len(gemma_full),
            "token_length_mean": sum(r["n_tokens"] for r in gemma_full) / len(gemma_full),
        },
        "refusal_qwen": {
            "total": len(qwen_full),
            "refusal_count": sum(1 for r in qwen_full if r["is_refusal"]),
            "compliance_count": sum(1 for r in qwen_full if not r["is_refusal"]),
            "refusal_rate": sum(1 for r in qwen_full if r["is_refusal"]) / len(qwen_full),
            "token_length_mean": sum(r["n_tokens"] for r in qwen_full) / len(qwen_full),
        },
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'results.json'}")


# ── Main ──────────────────────────────────────────────────────
def main():
    cyber_all, cyber_train, cyber_test = explore_cyber()
    gemma_full, gemma_train, gemma_test, gemma_attrib = explore_refusal("gemma4_31b")
    qwen_full, qwen_train, qwen_test, qwen_attrib = explore_refusal("qwen36")

    plot_cyber_distributions(cyber_all)
    plot_refusal_distributions(gemma_full, qwen_full)
    save_summary(cyber_all, gemma_full, qwen_full)

    print_header("DONE")
    print("Check experiments/01_data_exploration/ for plots and results.json")


if __name__ == "__main__":
    main()
