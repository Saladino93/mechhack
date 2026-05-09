"""Experiment 02: Extract residual activations from Gemma / Qwen.

CLI entry point for activation extraction. Handles all datasets
(cyber probes, refusal probes for both models) with a single command.

Usage:
    # Extract refusal probes for Gemma (middle layer, all samples)
    python experiments/02_extract_activations/run.py \
        --dataset refusal_gemma4_31b \
        --model_key gemma4_31b \
        --layers middle

    # Extract cyber probes (specific layers, limited samples)
    python experiments/02_extract_activations/run.py \
        --dataset cyber \
        --model_key gemma4_31b \
        --layers "10,30,50" \
        --limit 100

    # Extract with custom output directory
    python experiments/02_extract_activations/run.py \
        --dataset refusal_qwen36 \
        --model_key qwen36 \
        --out_dir ./extracts/qwen36_refusal

    # List available datasets
    python experiments/02_extract_activations/run.py --list
"""
import argparse
import sys
from pathlib import Path

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import from sibling modules (extractor deferred — needs torch)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_dataset, list_datasets, get_tasks_for_dataset


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract residual activations from transformer models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", type=str, default="refusal_gemma4_31b",
                   choices=["cyber", "refusal_gemma4_31b", "refusal_qwen36"],
                   help="Which dataset to extract (default: refusal_gemma4_31b)")
    p.add_argument("--split", type=str, default="full",
                   help="Dataset split: 'full', 'train', 'test', 'eval' (default: full)")
    p.add_argument("--model_key", type=str, default="gemma4_31b",
                   choices=["gemma4_31b", "qwen36"],
                   help="Which model to use (default: gemma4_31b)")
    p.add_argument("--model_path", type=str, default=None,
                   help="Override model path (auto-resolved if not set)")
    p.add_argument("--layers", type=str, default="middle",
                   help="Layer spec: 'middle', 'early', 'late', 'all', "
                        "'32', '10,30,50', '0:65:8' (default: middle)")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"],
                   help="Storage dtype (default: fp16)")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output directory (default: ./extracts/<model_key>_<dataset>)")
    p.add_argument("--limit", type=int, default=0,
                   help="Max samples to process, 0 = all (default: 0)")
    p.add_argument("--no_chat_template", action="store_true",
                   help="Don't apply chat template (raw prompt only)")
    p.add_argument("--list", action="store_true",
                   help="List available datasets and exit")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Device to run on (default: cuda:0)")
    return p.parse_args()


def main():
    args = parse_args()

    # List mode
    if args.list:
        print("Available datasets:")
        for name, desc in list_datasets().items():
            tasks = get_tasks_for_dataset(name)
            print(f"  {name:<25} {desc}")
            print(f"    {'Tasks:':<25} {', '.join(tasks)}")
        return

    # Lazy import — needs torch/transformers (GPU deps)
    from extractor import ActivationExtractor, extract_dataset

    # Determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = REPO_ROOT / "extracts" / f"{args.model_key}_{args.dataset}"

    # Load dataset
    print(f"=== Extract Activations ===", flush=True)
    print(f"  Dataset:  {args.dataset} (split={args.split})", flush=True)
    print(f"  Model:    {args.model_key}", flush=True)
    print(f"  Layers:   {args.layers}", flush=True)
    print(f"  Dtype:    {args.dtype}", flush=True)
    print(f"  Output:   {out_dir}", flush=True)
    print(f"  Limit:    {args.limit or 'all'}", flush=True)
    print()

    samples = load_dataset(args.dataset, split=args.split)
    print(f"Loaded {len(samples)} samples from {args.dataset}/{args.split}", flush=True)

    # Create extractor and load model
    extractor = ActivationExtractor(
        model_key=args.model_key,
        layers=args.layers,
        dtype=args.dtype,
        use_chat_template=not args.no_chat_template,
        model_path=args.model_path,
        device=args.device,
    )
    extractor.load_model()

    # Run extraction
    metadata = extract_dataset(extractor, samples, out_dir, limit=args.limit)

    print(f"\nDone. Extracts saved to: {out_dir}")
    print(f"  Samples OK:      {metadata['n_ok']}")
    print(f"  Samples skipped: {metadata['n_skipped']}")
    print(f"  Samples failed:  {metadata['n_errors']}")


if __name__ == "__main__":
    main()
