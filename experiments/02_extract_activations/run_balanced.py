"""Experiment 02: Extract residual activations from Gemma / Qwen.

CLI entry point for activation extraction. Handles all datasets
(cyber probes, refusal probes for both models) with a single command.

Adds optional task filtering and class balancing before extraction:
  --task cyber_1 --balance_classes   # balanced dual_use/benign
  --balance_classes                  # balance raw labels in the loaded split
"""
import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import from sibling modules (extractor deferred — needs torch)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_dataset, list_datasets, get_tasks_for_dataset, get_label_for_task


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
                   help=("Max samples to process. With --balance_classes, this is interpreted "
                         "as a total balanced budget and may be rounded down to keep exact balance."))
    p.add_argument("--no_chat_template", action="store_true",
                   help="Don't apply chat template (raw prompt only)")
    p.add_argument("--list", action="store_true",
                   help="List available datasets and exit")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Device to run on (default: cuda:0)")

    # New selection/balancing options
    p.add_argument("--task", type=str, default=None,
                   help=("Optional task filter before extraction, e.g. cyber_1, cyber_2, cyber_3. "
                         "Samples outside the task scope are removed."))
    p.add_argument("--balance_classes", action="store_true",
                   help=("Randomly balance selected samples so every class has the same count. "
                         "If --task is set, balances binary task labels; otherwise balances raw labels."))
    p.add_argument("--samples_per_class", type=int, default=0,
                   help=("Exact number of examples per class for --balance_classes. "
                         "If omitted, uses the largest balanced count allowed by --limit or the smallest class."))
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for task filtering / balancing (default: 0)")
    return p.parse_args()


def _label_name(x):
    """Stable printable key for labels that might be bool/int/str."""
    return str(x)


def filter_by_task(samples, task):
    """Keep only samples in a binary task's scope and attach _task_label."""
    out = []
    skipped = 0
    for s in samples:
        y = get_label_for_task(s, task)
        if y is None:
            skipped += 1
            continue
        ss = dict(s)
        ss["_task_label"] = int(y)
        out.append(ss)
    return out, skipped


def balance_samples(samples, *, key_fn, seed, limit=0, samples_per_class=0):
    """Return a shuffled class-balanced subset.

    If samples_per_class > 0, select exactly that many per class.
    Else if limit > 0, select floor(limit / n_classes) per class.
    Else select min class size per class.
    """
    groups = defaultdict(list)
    for s in samples:
        k = key_fn(s)
        groups[k].append(s)

    if not groups:
        raise ValueError("No samples available after filtering; cannot balance classes.")
    if len(groups) < 2:
        counts = {repr(k): len(v) for k, v in groups.items()}
        raise ValueError(f"Only one class is available after filtering: {counts}")

    rng = random.Random(seed)
    for g in groups.values():
        rng.shuffle(g)

    min_count = min(len(g) for g in groups.values())
    n_classes = len(groups)
    if samples_per_class and samples_per_class > 0:
        n_each = samples_per_class
    elif limit and limit > 0:
        n_each = limit // n_classes
        if n_each * n_classes != limit:
            print(f"  [warn] --limit {limit} is not divisible by {n_classes} classes; "
                  f"using {n_each * n_classes} samples for exact balance.", flush=True)
    else:
        n_each = min_count

    if n_each <= 0:
        raise ValueError(
            f"Balanced sample count per class would be {n_each}. "
            f"Increase --limit or remove --balance_classes."
        )
    if n_each > min_count:
        counts = {repr(k): len(v) for k, v in groups.items()}
        raise ValueError(
            f"Requested {n_each} samples per class but smallest class has {min_count}. "
            f"Class counts: {counts}"
        )

    selected = []
    for k in sorted(groups.keys(), key=_label_name):
        selected.extend(groups[k][:n_each])
    rng.shuffle(selected)
    return selected, {k: len(v) for k, v in groups.items()}, n_each


def print_raw_counts(samples, prefix=""):
    counts = Counter(s.get("label") for s in samples)
    if counts:
        joined = ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: str(kv[0])))
        print(f"{prefix}raw labels: {joined}", flush=True)


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
    print(f"  Task:     {args.task or 'none'}", flush=True)
    print(f"  Balance:  {args.balance_classes}", flush=True)
    print()

    samples = load_dataset(args.dataset, split=args.split)
    print(f"Loaded {len(samples)} samples from {args.dataset}/{args.split}", flush=True)
    print_raw_counts(samples, prefix="  ")

    if args.task:
        samples, skipped = filter_by_task(samples, args.task)
        print(f"Filtered to task={args.task}: kept={len(samples)} skipped={skipped}", flush=True)
        task_counts = Counter(s["_task_label"] for s in samples)
        print(f"  task labels: neg={task_counts.get(0, 0)} pos={task_counts.get(1, 0)}", flush=True)
        print_raw_counts(samples, prefix="  in-scope ")

    # If balancing, do it here and pass limit=0 to extract_dataset so it does not slice again.
    extract_limit = args.limit
    if args.balance_classes:
        if args.task:
            key_fn = lambda s: s["_task_label"]
            key_desc = f"binary task labels for {args.task}"
        else:
            key_fn = lambda s: s.get("label")
            key_desc = "raw labels"
        samples, pre_counts, n_each = balance_samples(
            samples,
            key_fn=key_fn,
            seed=args.seed,
            limit=args.limit,
            samples_per_class=args.samples_per_class,
        )
        print(f"Balanced by {key_desc}: {n_each} per class, total={len(samples)}", flush=True)
        print(f"  pre-balance counts: {dict(pre_counts)}", flush=True)
        if args.task:
            task_counts = Counter(s["_task_label"] for s in samples)
            print(f"  selected task labels: neg={task_counts.get(0, 0)} pos={task_counts.get(1, 0)}", flush=True)
        print_raw_counts(samples, prefix="  selected ")
        extract_limit = 0

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
    metadata = extract_dataset(extractor, samples, out_dir, limit=extract_limit)

    print(f"\nDone. Extracts saved to: {out_dir}")
    print(f"  Samples OK:      {metadata['n_ok']}")
    print(f"  Samples skipped: {metadata['n_skipped']}")
    print(f"  Samples failed:  {metadata['n_errors']}")


if __name__ == "__main__":
    main()
