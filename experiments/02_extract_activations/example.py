"""Example: how to use the extraction modules in Python.

This script demonstrates both the data loading and extraction APIs.
It can run locally (CPU) with a small model, or on the cluster (GPU)
with Gemma/Qwen.

Run from repo root:
    # CPU test (no real model, just shows data loading)
    python experiments/02_extract_activations/example.py --demo data

    # GPU extraction (cluster or local with model downloaded)
    python experiments/02_extract_activations/example.py --demo extract \
        --model_key gemma4_31b --limit 2

    # Load and inspect a saved .pt file
    python experiments/02_extract_activations/example.py --demo inspect \
        --pt_file ./extracts/gemma4_31b_refusal_gemma4_31b/ref_gemma4_31b_pos_0001.pt
"""
import argparse
import sys
from pathlib import Path

# Setup imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_dataset, list_datasets, get_tasks_for_dataset, get_label_for_task


def demo_data():
    """Demonstrate data loading — no GPU needed."""
    print("=" * 60)
    print("  DEMO: Data Loading")
    print("=" * 60)

    # 1. List all datasets
    print("\n--- Available datasets ---")
    for name, desc in list_datasets().items():
        tasks = get_tasks_for_dataset(name)
        print(f"  {name:<25} {desc}")
        print(f"    Probe tasks: {', '.join(tasks)}")

    # 2. Load cyber probes
    print("\n--- Loading cyber probes (test split) ---")
    cyber_test = load_dataset("cyber", split="test")
    print(f"  Loaded {len(cyber_test)} samples")
    print(f"  First sample keys: {list(cyber_test[0].keys())}")
    print(f"  Example sample_id: {cyber_test[0]['sample_id']}")
    print(f"  Example label:     {cyber_test[0]['label']}")
    print(f"  Prompt preview:    {cyber_test[0]['prompt'][:100]}...")

    # 3. Show label mapping for probe tasks
    print("\n--- Label mapping for cyber probe tasks ---")
    sample = cyber_test[0]
    for task in ["cyber_1", "cyber_2", "cyber_3"]:
        binary = get_label_for_task(sample, task)
        print(f"  {task}: raw='{sample['label']}' -> binary={binary}")

    # 4. Load refusal probes
    print("\n--- Loading Gemma refusal probes (train split) ---")
    gemma_train = load_dataset("refusal_gemma4_31b", split="train")
    print(f"  Loaded {len(gemma_train)} samples")
    refusals = sum(1 for s in gemma_train if s["label"])
    print(f"  Refusals: {refusals}, Compliance: {len(gemma_train) - refusals}")

    # 5. Load attribution eval (Level-2 subset)
    print("\n--- Loading attribution eval set ---")
    attrib = load_dataset("refusal_gemma4_31b", split="eval")
    print(f"  Loaded {len(attrib)} samples (all refusals, <=2048 tokens)")

    print("\n--- Data loading complete ---")


def demo_extract(model_key, limit):
    """Demonstrate extraction — needs GPU + model."""
    from extractor import ActivationExtractor, extract_dataset

    print("=" * 60)
    print(f"  DEMO: Extraction ({model_key}, limit={limit})")
    print("=" * 60)

    # 1. Load a few samples
    dataset_name = f"refusal_{model_key}"
    samples = load_dataset(dataset_name, split="train")
    print(f"\nLoaded {len(samples)} samples from {dataset_name}")

    # 2. Create extractor
    extractor = ActivationExtractor(
        model_key=model_key,
        layers="middle",        # single middle layer
        dtype="fp16",
        use_chat_template=True,
    )

    # 3. Load model
    print("\nLoading model...")
    extractor.load_model()

    # 4. Extract a single sample (manual way)
    print("\n--- Single sample extraction ---")
    sample = samples[0]
    print(f"Sample: {sample['sample_id']}")
    print(f"Prompt: {sample['prompt'][:100]}...")

    result = extractor.extract_single(sample["prompt"])
    print(f"Result keys: {list(result.keys())}")
    print(f"  residuals shape: {result['residuals'].shape}")
    print(f"  residuals dtype: {result['residuals'].dtype}")
    print(f"  n_tokens: {result['n_tokens']}")
    print(f"  layer_idxs: {result['layer_idxs']}")
    print(f"  fwd_seconds: {result['fwd_seconds']}")
    print(f"  peak_vram_gb: {result['peak_vram_gb']}")

    # 5. Batch extraction (the normal workflow)
    out_dir = REPO_ROOT / "extracts" / f"{model_key}_example"
    print(f"\n--- Batch extraction to {out_dir} ---")
    metadata = extract_dataset(extractor, samples, out_dir, limit=limit)
    print(f"\nExtraction metadata: {metadata['n_ok']} ok, "
          f"{metadata['n_skipped']} skipped, {metadata['n_errors']} errors")


def demo_inspect(pt_file):
    """Inspect a saved .pt extraction file."""
    import torch

    print("=" * 60)
    print(f"  DEMO: Inspect {pt_file}")
    print("=" * 60)

    data = torch.load(pt_file, weights_only=False)

    print(f"\nKeys: {list(data.keys())}")
    for key, val in data.items():
        if hasattr(val, "shape"):
            print(f"  {key:20s}  shape={str(val.shape):20s}  dtype={val.dtype}")
        elif isinstance(val, list):
            print(f"  {key:20s}  list len={len(val)}")
        else:
            print(f"  {key:20s}  {val}")

    # Show residuals info
    res = data["residuals"]
    print(f"\nResiduals tensor:")
    print(f"  Shape:  {res.shape}  (n_layers, n_tokens, d_model)")
    print(f"  Dtype:  {res.dtype}")
    print(f"  Memory: {res.element_size() * res.nelement() / 1024**2:.1f} MB")
    print(f"  Mean:   {res.float().mean():.4f}")
    print(f"  Std:    {res.float().std():.4f}")
    print(f"  Min:    {res.float().min():.4f}")
    print(f"  Max:    {res.float().max():.4f}")

    if "attention_mask" in data:
        mask = data["attention_mask"]
        n_real = mask.sum().item()
        print(f"\nAttention mask: {n_real}/{mask.shape[0]} real tokens")


def main():
    p = argparse.ArgumentParser(
        description="Example usage of the extraction modules.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--demo", required=True,
                   choices=["data", "extract", "inspect"],
                   help="Which demo to run")
    p.add_argument("--model_key", default="gemma4_31b",
                   choices=["gemma4_31b", "qwen36"])
    p.add_argument("--limit", type=int, default=2,
                   help="Max samples for extract demo (default: 2)")
    p.add_argument("--pt_file", type=str, default=None,
                   help="Path to .pt file for inspect demo")
    args = p.parse_args()

    if args.demo == "data":
        demo_data()
    elif args.demo == "extract":
        demo_extract(args.model_key, args.limit)
    elif args.demo == "inspect":
        if not args.pt_file:
            print("ERROR: --pt_file required for inspect demo")
            sys.exit(1)
        demo_inspect(args.pt_file)


if __name__ == "__main__":
    main()
