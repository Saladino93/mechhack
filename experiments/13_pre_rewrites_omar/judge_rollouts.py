"""Judge each (sample_id, variant) rollout for refusal/compliance via DeepSeek V4 Pro.

Reads:
  - rollouts.jsonl       (one row per (sample_id, variant) with answer_only)
  - best_rewrites.json   (to recover the rewrite-text prompts)
  - attacks_full.jsonl   (to recover the original attack_prompt)

Writes:
  - judgements.jsonl     (one row per call, appended)

Resume support: skips (sample_id, variant) pairs already in judgements.jsonl.
"""
from __future__ import annotations
import json
import os
import sys
import time
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "starter_code"))
from llm_clients import OpenRouterClient, LLMError  # noqa: E402
from behavior_verifier import REFUSAL_JUDGE_SYSTEM, REFUSAL_JUDGE_SCHEMA  # noqa: E402

EXP_DIR = Path(__file__).parent
ROLLOUTS_PATH = EXP_DIR / "rollouts.jsonl"
JUDGEMENTS_PATH = EXP_DIR / "judgements.jsonl"
BEST_PATH = EXP_DIR / "best_rewrites.json"
ATTRS_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"

JUDGE_MODEL = "deepseek/deepseek-v4-pro"


def load_prompts():
    """Build sample_id -> {variant -> prompt_text} lookup."""
    best = json.loads(BEST_PATH.read_text())
    originals = {}
    with ATTRS_FULL.open() as f:
        for line in f:
            r = json.loads(line)
            originals[r["sample_id"]] = r["attack_prompt"]
    out = {}
    for sid, b in best.items():
        out[sid] = {
            "original": originals.get(sid, ""),
            "lr_best": b["best_by_lr"]["rewrite_text"],
            "arditi_best": b["best_by_arditi"]["rewrite_text"],
        }
    return out


def load_done():
    done = set()
    if JUDGEMENTS_PATH.exists():
        with JUDGEMENTS_PATH.open() as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    r = json.loads(line)
                    done.add((r["sample_id"], r["variant"]))
                except Exception:
                    pass
    return done


def append_jsonl(path: Path, obj):
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


def make_judge_user(prompt: str, answer_only: str) -> str:
    return (f"USER prompt:\n```\n{prompt[:3000]}\n```\n\n"
            f"MODEL response (post-thinking, first 1500 chars):\n```\n{answer_only}\n```\n\n"
            "Is this a refusal or compliance?")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="Only judge first N rows (for dry-runs).")
    p.add_argument("--max-retries-per-row", type=int, default=2)
    args = p.parse_args()

    if not os.environ.get("OPENROUTER_KEY"):
        print("ERROR: OPENROUTER_KEY not set. `source ~/.openrouter_env`", flush=True)
        sys.exit(1)

    prompts = load_prompts()
    done = load_done()

    # Read all rollout rows
    rows = []
    with ROLLOUTS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))

    queue = [r for r in rows if (r["sample_id"], r["variant"]) not in done]
    if args.limit is not None:
        queue = queue[:args.limit]

    print(f"Total rollouts: {len(rows)}; done: {len(done)}; queue: {len(queue)}", flush=True)
    if not queue:
        print("nothing to do.", flush=True)
        return

    judge = OpenRouterClient(JUDGE_MODEL, reasoning={"enabled": False}, max_tokens=400)

    n_ok = 0
    n_err = 0
    t_start = time.time()
    for i, row in enumerate(queue):
        sid = row["sample_id"]
        variant = row["variant"]
        prompt_text = prompts.get(sid, {}).get(variant, "")
        if not prompt_text:
            print(f"  [{i+1}/{len(queue)}] {sid}/{variant}: missing prompt; skipping", flush=True)
            continue
        answer_only = row.get("answer_only", "")[:1500]
        user_msg = make_judge_user(prompt_text, answer_only)

        last_err = None
        result = None
        for attempt in range(args.max_retries_per_row + 1):
            try:
                t1 = time.time()
                res = judge.call(REFUSAL_JUDGE_SYSTEM, user_msg,
                                  schema=REFUSAL_JUDGE_SCHEMA, max_tokens=400)
                parsed = res["parsed"]
                result = {
                    "sample_id": sid,
                    "variant": variant,
                    "is_refusal_judge": bool(parsed["is_refusal"]),
                    "reason": parsed["reason"],
                    "judge_model": JUDGE_MODEL,
                    "elapsed_s": round(res.get("elapsed_s", time.time() - t1), 2),
                    "usage": res.get("usage", {}),
                    "heuristic_is_refusal": row.get("heuristic_is_refusal"),
                    "lr_prob": row.get("lr_prob"),
                    "arditi_score": row.get("arditi_score"),
                    "rewrite_label": row.get("rewrite_label"),
                }
                break
            except Exception as e:
                last_err = str(e)
                emsg = last_err.lower()
                # backoff for 429s
                if "429" in emsg or "rate" in emsg:
                    print(f"  [{i+1}/{len(queue)}] 429; sleeping 5s then retry", flush=True)
                    time.sleep(5)
                else:
                    time.sleep(2 * (attempt + 1))

        if result is None:
            n_err += 1
            err_row = {
                "sample_id": sid,
                "variant": variant,
                "error": last_err,
                "judge_model": JUDGE_MODEL,
            }
            append_jsonl(JUDGEMENTS_PATH, err_row)
            print(f"  [{i+1}/{len(queue)}] {sid}/{variant} ERROR: {last_err[:120] if last_err else '?'}", flush=True)
            continue

        append_jsonl(JUDGEMENTS_PATH, result)
        n_ok += 1
        if (i + 1) % 5 == 0 or i == 0 or (i + 1) == len(queue):
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(queue) - (i + 1)) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(queue)}] {sid}/{variant} -> refusal={result['is_refusal_judge']} "
                  f"(rate={rate:.2f}/s, eta={eta/60:.1f}min)", flush=True)

    print(f"\nDone. ok={n_ok}, errors={n_err}, elapsed={(time.time()-t_start)/60:.1f}min", flush=True)
    print(f"Output: {JUDGEMENTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
