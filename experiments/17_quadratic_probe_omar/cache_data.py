"""Pre-compute and cache pooled features for every (task, layer) we'll use.

Run this *once* before run_all.sh — saves ~5 min × 3 cyber tasks of repeated
disk IO. Without caching, each (task, d_pca) invocation re-reads 100+ GB of
.pt files from /home/ubuntu/extracts/cyber_all_omar/.

Outputs:
    cache/cyber_<task>_L<layer>.npz
    cache/refusal_gemma_L32.npz
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Re-use the loaders in train.py — they write the cache as a side effect.
import train  # noqa: E402


TASKS = [("cyber_1", 40), ("cyber_2", 40), ("cyber_3", 35),
         ("refusal_gemma", 32)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=[t for t, _ in TASKS] + ["all"], default="all")
    args = ap.parse_args()

    todo = TASKS if args.task == "all" else [(t, l) for t, l in TASKS if t == args.task]

    for task, layer in todo:
        cache_path = train.CACHE_DIR / (
            f"refusal_gemma_L32.npz" if task == "refusal_gemma"
            else f"cyber_{task}_L{layer}.npz")
        if cache_path.exists():
            print(f"[skip] {cache_path.name} already cached", flush=True)
            continue
        print(f"[load] {task} L{layer} ...", flush=True)
        t0 = time.time()
        if task == "refusal_gemma":
            train.load_refusal_L32()
        else:
            train.load_cyber(task, layer)
        print(f"  -> done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
